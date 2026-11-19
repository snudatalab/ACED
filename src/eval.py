import argparse
import inspect
import json
import os
import random
import sys
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding


def _patch_transformers_cache_compat():
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return
    if not hasattr(DynamicCache, "get_max_length") and hasattr(DynamicCache, "get_max_cache_shape"):
        DynamicCache.get_max_length = DynamicCache.get_max_cache_shape


_patch_transformers_cache_compat()


def _accepts_kwarg(fn, name: str) -> bool:
    try:
        sig = inspect.signature(fn)
        for p in sig.parameters.values():
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                return True
        return name in sig.parameters
    except Exception:
        return False


@contextmanager
def suppress_load_balancing_loss(modeling_mixtral):
    original_fn = getattr(modeling_mixtral, "load_balancing_loss_func", None)
    if original_fn is None:
        yield
        return

    def _zero_loss(gate_logits=None, num_experts=None, top_k=2, attention_mask=None):
        ref = None
        if isinstance(gate_logits, tuple) and gate_logits:
            ref = gate_logits[0]
        elif torch.is_tensor(gate_logits):
            ref = gate_logits
        elif torch.is_tensor(attention_mask):
            ref = attention_mask
        device = ref.device if ref is not None else torch.device("cpu")
        dtype = ref.dtype if ref is not None else torch.float32
        return torch.zeros((), device=device, dtype=dtype)

    modeling_mixtral.load_balancing_loss_func = _zero_loss
    try:
        yield
    finally:
        modeling_mixtral.load_balancing_loss_func = original_fn


def _is_scalar_number(x: Any) -> bool:
    return isinstance(x, (float, int))


def _is_scalar_tensor(x: Any) -> bool:
    return torch.is_tensor(x) and x.numel() == 1


def _set_scalar_attr_zero(obj: Any, attr: str):
    if not hasattr(obj, attr):
        return None
    v = getattr(obj, attr)
    if _is_scalar_number(v):
        old = v
        try:
            setattr(obj, attr, 0.0)
            return (obj, attr, old)
        except Exception:
            return None
    if _is_scalar_tensor(v):
        old = v.detach().clone()
        try:
            with torch.no_grad():
                v.zero_()
            return (obj, attr, old)
        except Exception:
            return None
    return None


def _restore_scalar_attr(handle):
    obj, attr, old = handle
    cur = getattr(obj, attr, None)
    if _is_scalar_number(old):
        try:
            setattr(obj, attr, old)
        except Exception:
            pass
        return
    if torch.is_tensor(old) and _is_scalar_tensor(cur):
        try:
            with torch.no_grad():
                cur.copy_(old)
        except Exception:
            pass
        return
    try:
        setattr(obj, attr, old)
    except Exception:
        pass


@contextmanager
def suppress_moe_aux_loss(
    model,
    *,
    keys: Optional[Iterable[str]] = None,
    verbose: bool = False,
    scan_modules: bool = True,
):
    key_list = list(keys) if keys is not None else [
        "router_aux_loss_coef",
        "router_z_loss_coef",
        "z_loss_coef",
        "load_balancing_loss_coef",
        "aux_loss_alpha",
        "aux_loss_coef",
        "moe_aux_loss_coef",
        "gate_loss_coef",
        "expert_loss_coef",
        "auxiliary_loss_coef",
        "aux_coef",
        "alpha",
    ]
    restores = []

    def _apply(obj, prefix: str):
        for k in key_list:
            h = _set_scalar_attr_zero(obj, k)
            if h is not None:
                restores.append(h)
                if verbose:
                    old = h[2]
                    old_val = float(old.item()) if torch.is_tensor(old) else float(old)
                    print(f"[AuxZero] {prefix}.{k}: {old_val} -> 0")

    if hasattr(model, "config") and model.config is not None:
        _apply(model.config, "config")
    _apply(model, "model")

    if scan_modules:
        for m in model.modules():
            for k in key_list:
                if k == "alpha":
                    continue
                h = _set_scalar_attr_zero(m, k)
                if h is not None:
                    restores.append(h)
                    if verbose:
                        old = h[2]
                        old_val = float(old.item()) if torch.is_tensor(old) else float(old)
                        print(f"[AuxZero] module:{m.__class__.__name__}.{k}: {old_val} -> 0")

    try:
        yield
    finally:
        for h in reversed(restores):
            _restore_scalar_attr(h)


@torch.no_grad()
def get_ppl(
    model,
    testenc,
    device,
    modeling_mixtral=None,
    max_length=2048,
    stride=512,
    *,
    zero_aux_coef: bool = True,
    aux_verbose: bool = False,
    suppress_mixtral_lb: bool = True,
):
    input_ids = testenc.input_ids
    model.eval()

    use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = False

    supports_output_router = _accepts_kwarg(model.forward, "output_router_logits")

    total_nll = 0.0
    total_tokens = 0
    try:
        with ExitStack() as stack:
            if modeling_mixtral is not None and suppress_mixtral_lb:
                stack.enter_context(suppress_load_balancing_loss(modeling_mixtral))
            else:
                stack.enter_context(nullcontext())

            if zero_aux_coef:
                stack.enter_context(suppress_moe_aux_loss(model, verbose=aux_verbose, scan_modules=True))
            else:
                stack.enter_context(nullcontext())

            seq_len = input_ids.size(1)
            prev_end_loc = 0
            for begin_loc in tqdm(range(0, seq_len, stride), desc="(Eval) PPL"):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = end_loc - prev_end_loc

                input_chunk = input_ids[:, begin_loc:end_loc].to(device)
                if input_chunk.size(1) <= 1:
                    prev_end_loc = end_loc
                    if end_loc == seq_len:
                        break
                    continue

                labels = input_chunk.clone()
                labels[:, :-trg_len] = -100

                kwargs = {"labels": labels}
                if supports_output_router:
                    kwargs["output_router_logits"] = False

                outputs = model(input_chunk, **kwargs)
                loss = outputs.loss
                if not torch.isfinite(loss):
                    prev_end_loc = end_loc
                    if end_loc == seq_len:
                        break
                    continue

                num_valid = (labels != -100).sum().item()
                n_tokens = num_valid - labels.size(0)
                if n_tokens > 0:
                    total_nll += loss.item() * n_tokens
                    total_tokens += n_tokens

                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break
    finally:
        model.config.use_cache = use_cache

    if total_tokens == 0:
        return float("nan")
    return torch.exp(torch.tensor(total_nll / total_tokens)).item()


def get_tokenizer(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
    return tok


def _safe_load_dataset(*args, **kwargs):
    last_err = None
    for _ in range(5):
        try:
            return load_dataset(*args, **kwargs)
        except Exception as e:
            last_err = e
    raise last_err


def get_eval_enc(
    tokenizer,
    data_name: str,
    *,
    total_eval_tokens: Optional[int],
    c4_eval_num_docs: int,
    seed: int,
):
    _ = seed
    data_name = data_name.lower()
    if "wikitext" in data_name:
        ds = _safe_load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        enc = tokenizer("\n\n".join(ds["text"]), return_tensors="pt")
        ids_1d = enc.input_ids[0].contiguous()
    elif "c4" in data_name:
        ds = _safe_load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
        n_docs = min(c4_eval_num_docs, len(ds))
        text = " ".join(ds[:n_docs]["text"])
        enc = tokenizer(text, return_tensors="pt")
        ids_1d = enc.input_ids[0].contiguous()
    else:
        raise NotImplementedError(f"Unsupported ppl dataset: {data_name}")

    if total_eval_tokens is not None:
        ids_1d = ids_1d[:total_eval_tokens]
    return BatchEncoding({"input_ids": ids_1d.unsqueeze(0)})


def _to_dtype(dtype_name: str):
    name = dtype_name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    if name == "auto":
        return "auto"
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _run_lm_eval(args, model, tokenizer):
    if not args.lm_eval_repo:
        raise ValueError("--run_lm_eval requires --lm_eval_repo <path to a local lm-evaluation-harness checkout>.")
    lm_eval_repo = os.path.abspath(args.lm_eval_repo)
    if not os.path.isdir(lm_eval_repo):
        raise FileNotFoundError(f"lm-evaluation-harness repo not found: {lm_eval_repo}")
    if lm_eval_repo not in sys.path:
        sys.path.insert(0, lm_eval_repo)

    from lm_eval import evaluator, utils
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.lm_eval_batch_size,
        device=args.device,
        trust_remote_code=True,
        mixed_precision_dtype=(None if args.dtype == "auto" else args.dtype),
    )
    lm._device = torch.device(args.device)

    tasks = [t.strip() for t in args.lm_eval_tasks.split(",") if t.strip()]
    if not tasks:
        raise ValueError("No lm-eval tasks specified.")

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.lm_eval_batch_size,
        device=args.device,
        log_samples=args.log_samples,
        limit=args.limit,
        confirm_run_unsafe_code=args.confirm_run_unsafe_code,
    )
    if results is None:
        raise RuntimeError("lm_eval returned None results")

    print(utils.make_table(results))
    if "groups" in results:
        print(utils.make_table(results, "groups"))
    return results


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--attn_implementation", type=str, default="eager")
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--run_ppl", action="store_true")
    ap.add_argument("--ppl_data_name", type=str, default="c4", choices=["wikitext2", "c4"])
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument("--total_eval_tokens", type=int, default=300000)
    ap.add_argument("--c4_eval_num_docs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zero_aux_coef", action="store_true")
    ap.add_argument("--aux_verbose", action="store_true")
    ap.add_argument("--suppress_mixtral_lb", action="store_true")

    ap.add_argument("--run_lm_eval", action="store_true")
    ap.add_argument("--lm_eval_repo", type=str, default=None)
    ap.add_argument("--lm_eval_tasks", type=str, default="piqa,boolq,arc_challenge,arc_easy,hellaswag,winogrande,mmlu")
    ap.add_argument("--lm_eval_batch_size", type=str, default="auto")
    ap.add_argument("--num_fewshot", type=int, default=0)
    ap.add_argument("--limit", type=float, default=None)
    ap.add_argument("--log_samples", action="store_true")
    ap.add_argument("--confirm_run_unsafe_code", action="store_true")

    ap.add_argument("--output_json", type=str, default="./results/eval_results.json")
    return ap.parse_args()


def main():
    args = parse_args()

    if not args.run_ppl and not args.run_lm_eval:
        args.run_ppl = True
        args.run_lm_eval = True

    torch_dtype = _to_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    tokenizer = get_tokenizer(args.model_path)

    out: Dict[str, Any] = {
        "model_path": args.model_path,
        "device": args.device,
        "dtype": args.dtype,
    }

    if args.run_ppl:
        testenc = get_eval_enc(
            tokenizer,
            args.ppl_data_name,
            total_eval_tokens=args.total_eval_tokens,
            c4_eval_num_docs=args.c4_eval_num_docs,
            seed=args.seed,
        )
        modeling_mixtral = None
        try:
            from transformers.models.mixtral import modeling_mixtral as mm  # type: ignore
            modeling_mixtral = mm
        except Exception:
            modeling_mixtral = None

        ppl = get_ppl(
            model,
            testenc,
            args.device,
            modeling_mixtral=modeling_mixtral,
            max_length=args.max_length,
            stride=args.stride,
            zero_aux_coef=args.zero_aux_coef,
            aux_verbose=args.aux_verbose,
            suppress_mixtral_lb=args.suppress_mixtral_lb,
        )
        out["ppl"] = {
            "dataset": args.ppl_data_name,
            "value": float(ppl),
            "max_length": args.max_length,
            "stride": args.stride,
            "total_eval_tokens": args.total_eval_tokens,
            "c4_eval_num_docs": args.c4_eval_num_docs,
            "zero_aux_coef": bool(args.zero_aux_coef),
            "suppress_mixtral_lb": bool(args.suppress_mixtral_lb),
        }
        print(f"[PPL] {args.ppl_data_name}: {ppl:.6f}")

    if args.run_lm_eval:
        results = _run_lm_eval(args, model, tokenizer)
        out["lm_eval"] = results

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
