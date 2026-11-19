# data_utils.py
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
from torch import Tensor
from datasets import load_dataset
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizer


# =========================
# Config / Helpers
# =========================

@dataclass
class EvalConfig:
    seqlen: int = 2048
    total_eval_tokens: Optional[int] = None
    c4_eval_num_docs: int = 1100
    seed: int = 0
    cache_dir: Optional[str] = None


def _ensure_pad_token(tokenizer: PreTrainedTokenizer) -> None:
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})


def _safe_load_dataset(*args, retry: int = 5, **kwargs):
    last_err = None
    for _ in range(retry):
        try:
            return load_dataset(*args, **kwargs)
        except Exception as e:
            last_err = e
    raise last_err


def _cache_path(cache_dir: str, name: str) -> Path:
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / name


def _save_cached_ids(path: Path, input_ids_1d: torch.LongTensor, meta: Dict[str, Any]) -> None:
    obj = {
        "input_ids": input_ids_1d.cpu(),
        "meta": meta,
    }
    torch.save(obj, path)


def _load_cached_ids(path: Path) -> Optional[torch.LongTensor]:
    if not path.exists():
        return None
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "input_ids" in obj:
        return obj["input_ids"]
    return None


# =========================
# WikiText2
# =========================

def get_wikitext2_calib(
    tokenizer: PreTrainedTokenizer,
    num_samples: int,
    seqlen: int,
    seed: int,
) -> List[Tensor]:
    ds = _safe_load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    enc = tokenizer("\n\n".join(ds["text"]), return_tensors="pt")
    ids: torch.LongTensor = enc.input_ids  # [1, T]
    T = ids.size(1)
    if T < seqlen + 1:
        raise ValueError(f"WikiText2 train tokens too short: T={T}, seqlen={seqlen}")

    rng = random.Random(seed)
    samples: List[Tensor] = []
    for _ in range(num_samples):
        i = rng.randint(0, T - seqlen - 1)
        j = i + seqlen
        samples.append(ids[:, i:j].clone())  # [1, seqlen]
    return samples


def get_wikitext2_eval_enc(
    tokenizer: PreTrainedTokenizer,
    cfg: EvalConfig,
) -> BatchEncoding:
    cache_ids = None
    cache_key = None
    if cfg.cache_dir is not None:
        cache_key = f"wikitext2_test_tok_seed{cfg.seed}_tot{cfg.total_eval_tokens}_tok.pt"
        cache_file = _cache_path(cfg.cache_dir, cache_key)
        cache_ids = _load_cached_ids(cache_file)

    if cache_ids is None:
        ds = _safe_load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        enc = tokenizer("\n\n".join(ds["text"]), return_tensors="pt")
        ids_1d = enc.input_ids[0].contiguous()  # [T]
        if cfg.total_eval_tokens is not None:
            ids_1d = ids_1d[: cfg.total_eval_tokens]

        if cfg.cache_dir is not None:
            meta = {
                "dataset": "wikitext",
                "name": "wikitext-2-raw-v1",
                "split": "test",
                "seed": cfg.seed,
                "total_eval_tokens": cfg.total_eval_tokens,
            }
            _save_cached_ids(_cache_path(cfg.cache_dir, cache_key), ids_1d, meta)
    else:
        ids_1d = cache_ids
        if cfg.total_eval_tokens is not None:
            ids_1d = ids_1d[: cfg.total_eval_tokens]

    return BatchEncoding({"input_ids": ids_1d.unsqueeze(0)})  # [1, T]


# =========================
# C4
# =========================

def get_c4_calib(
    tokenizer: PreTrainedTokenizer,
    num_samples: int,
    seqlen: int,
    seed: int,
) -> List[Tensor]:
    rng = random.Random(seed)

    ds = _safe_load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )

    samples: List[Tensor] = []
    for _ in range(num_samples):
        while True:
            idx = rng.randint(0, len(ds) - 1)
            text = ds[idx]["text"]
            enc = tokenizer(text, return_tensors="pt")
            ids = enc.input_ids  # [1, L]
            L = ids.size(1)
            if L >= seqlen + 1:
                break
        start = rng.randint(0, L - seqlen - 1)
        end = start + seqlen
        samples.append(ids[:, start:end].clone())  # [1, seqlen]
    return samples

# =========================
# The Pile
# =========================

def get_pile_calib(
    tokenizer: PreTrainedTokenizer,
    num_samples: int,
    seqlen: int,
    seed: int,
) -> List[Tensor]:
    rng = random.Random(seed)

    try:
        ds = _safe_load_dataset("mit-han-lab/pile-val-backup", split="validation")
    except Exception as e:
        raise RuntimeError(f"Failed to load mit-han-lab/pile-val-backup validation split: {e}")

    # Without replacement over document indices.
    all_indices = list(range(len(ds)))
    rng.shuffle(all_indices)

    samples: List[Tensor] = []
    for idx in all_indices:
        text = ds[idx]["text"]
        enc = tokenizer(text, return_tensors="pt")
        ids = enc.input_ids  # [1, L]
        L = ids.size(1)
        if L < seqlen + 1:
            continue
        start = rng.randint(0, L - seqlen - 1)
        end = start + seqlen
        samples.append(ids[:, start:end].clone())  # [1, seqlen]
        if len(samples) >= num_samples:
            break

    if len(samples) < num_samples:
        raise RuntimeError(
            f"pile-val-backup validation does not have enough usable docs: got={len(samples)}, need={num_samples}, seqlen={seqlen}"
        )
    return samples

def get_c4_eval_enc(
    tokenizer: PreTrainedTokenizer,
    cfg: EvalConfig,
) -> BatchEncoding:
    cache_ids = None
    cache_key = None
    if cfg.cache_dir is not None:
        cache_key = (
            f"c4_val00000_tok_seed{cfg.seed}_docs{cfg.c4_eval_num_docs}"
            f"_tot{cfg.total_eval_tokens}_tok.pt"
        )
        cache_file = _cache_path(cfg.cache_dir, cache_key)
        cache_ids = _load_cached_ids(cache_file)

    if cache_ids is None:
        ds = _safe_load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )

        n_docs = min(cfg.c4_eval_num_docs, len(ds))
        text = " ".join(ds[:n_docs]["text"])  
        enc = tokenizer(text, return_tensors="pt")
        ids_1d = enc.input_ids[0].contiguous()

        if cfg.total_eval_tokens is not None:
            ids_1d = ids_1d[: cfg.total_eval_tokens]

        if cfg.cache_dir is not None:
            meta = {
                "dataset": "allenai/c4",
                "split": "validation",
                "shard": "en/c4-validation.00000-of-00008.json.gz",
                "seed": cfg.seed,
                "c4_eval_num_docs": cfg.c4_eval_num_docs,
                "total_eval_tokens": cfg.total_eval_tokens,
            }
            _save_cached_ids(_cache_path(cfg.cache_dir, cache_key), ids_1d, meta)
    else:
        ids_1d = cache_ids
        if cfg.total_eval_tokens is not None:
            ids_1d = ids_1d[: cfg.total_eval_tokens]

    return BatchEncoding({"input_ids": ids_1d.unsqueeze(0)})  # [1, T]


# =========================
# Public API
# =========================

def get_tokenizer(model_path: str, use_fast: bool = False) -> PreTrainedTokenizer:
    try:
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast, trust_remote_code=True)
    except Exception as e:
        msg = str(e).lower()
        has_tokenizer_json = (Path(model_path) / "tokenizer.json").is_file()
        should_fallback_fast = (
            (not use_fast)
            and has_tokenizer_json
            and ("protobuf" in msg or "not a string" in msg or "sentencepiece" in msg)
        )
        if not should_fallback_fast:
            raise
        print(
            f"[Info] Tokenizer slow-load failed for {model_path}; "
            "retrying with use_fast=True"
        )
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    _ensure_pad_token(tok)
    return tok


def get_calib_loader(
    tokenizer: PreTrainedTokenizer,
    data_name: str,
    num_samples: int = 128,
    seqlen: int = 2048,
    seed: int = 0,
) -> List[Tensor]:
    data_name = data_name.lower()
    if "wikitext2" in data_name or "wikitext" in data_name:
        return get_wikitext2_calib(tokenizer, num_samples, seqlen, seed)
    if "pile" in data_name:
        return get_pile_calib(tokenizer, num_samples, seqlen, seed)
    if "c4" in data_name:
        return get_c4_calib(tokenizer, num_samples, seqlen, seed)
    raise NotImplementedError(f"Unknown data_name: {data_name}")


def get_eval_enc(
    tokenizer: PreTrainedTokenizer,
    data_name: str,
    cfg: EvalConfig,
) -> BatchEncoding:
    data_name = data_name.lower()
    if "wikitext2" in data_name or "wikitext" in data_name:
        return get_wikitext2_eval_enc(tokenizer, cfg)
    if "c4" in data_name:
        return get_c4_eval_enc(tokenizer, cfg)
    raise NotImplementedError(f"Unknown data_name: {data_name}")


def get_loaders(
    model_path: str,
    data_name: str,
    num_samples: int = 128,
    seqlen: int = 2048,
    seed: int = 0,
    total_eval_tokens: Optional[int] = None,
    c4_eval_num_docs: int = 1100,
    cache_dir: Optional[str] = None,
    use_fast: bool = False,
) -> Tuple[PreTrainedTokenizer, List[Tensor], BatchEncoding]:
    tok = get_tokenizer(model_path, use_fast=use_fast)

    calib = get_calib_loader(
        tokenizer=tok,
        data_name=data_name,
        num_samples=num_samples,
        seqlen=seqlen,
        seed=seed,
    )

    cfg = EvalConfig(
        seqlen=seqlen,
        total_eval_tokens=total_eval_tokens,
        c4_eval_num_docs=c4_eval_num_docs,
        seed=seed,
        cache_dir=cache_dir,
    )
    testenc = get_eval_enc(tok, data_name=data_name, cfg=cfg)

    return tok, calib, testenc
