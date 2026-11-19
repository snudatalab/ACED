import re
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from weight_io import _load_expert_weight_stack


def _parse_layer_idx(name: str) -> int:
    m = re.search(r'\.layers\.(\d+)\.', name)
    if m:
        return int(m.group(1))
    m = re.search(r'layers\.(\d+)', name)
    return int(m.group(1)) if m else -1

def _find_moe_modules(model) -> Dict[int, torch.nn.Module]:
    out = {}
    for name, module in model.named_modules():
        if not (hasattr(module, 'experts') and (hasattr(module, 'gate') or hasattr(module, 'router') or hasattr(module, 'gate_proj'))):
            continue
        li = _parse_layer_idx(name)
        if li >= 0:
            out[li] = module
    return out

def _resolve_router_module(moe_layer: torch.nn.Module) -> torch.nn.Module:
    for attr in ('gate', 'router'):
        if hasattr(moe_layer, attr):
            return getattr(moe_layer, attr)
    raise AttributeError(f'Could not resolve router module for {type(moe_layer)}')

def _normalize_calib_data_name(data_name: str) -> str:
    return 'wikitext2' if str(data_name) == 'wiki2' else str(data_name)

def _collect_layer_inputs_and_router_info(
    *,
    model_path: str,
    layers: List[int],
    calib_num_samples: int,
    calib_seqlen: int,
    seed: int,
    calib_data_name: str,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, Dict[str, torch.Tensor]]]:
    print(f'[I3][Cache] loading model for calibration: {model_path}')
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map='auto',
        trust_remote_code=True,
    )
    model.eval()
    moe_by_layer = _find_moe_modules(model)
    missing = sorted(set(int(x) for x in layers) - set(moe_by_layer.keys()))
    if missing:
        raise ValueError(f'MoE layer not found for I3 cache. missing_layers={missing}')
    from utils.data_utils import get_calib_loader, get_tokenizer  # type: ignore
    tok = get_tokenizer(model_path, use_fast=False)
    calib = get_calib_loader(
        tokenizer=tok,
        data_name=_normalize_calib_data_name(calib_data_name),
        num_samples=int(calib_num_samples),
        seqlen=int(calib_seqlen),
        seed=int(seed),
    )
    x_buckets = {int(li): [] for li in layers}
    topk_buckets = {int(li): [] for li in layers}
    logits_buckets = {int(li): [] for li in layers}

    def input_hook(layer_idx: int):
        def hook(_module, inputs):
            x = inputs[0].detach().reshape(-1, inputs[0].size(-1)).cpu().to(torch.float32)
            x_buckets[layer_idx].append(x)
        return hook

    def router_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            logits = None
            topk_idx = None
            if isinstance(output, dict):
                logits = output.get('logits', None)
                topk_idx = output.get('topk_idx', None)
            elif isinstance(output, (tuple, list)):
                tensors = [t for t in output if torch.is_tensor(t)]
                ints = [t for t in tensors if t.dtype in (torch.int32, torch.int64, torch.long)]
                floats = [t for t in tensors if t.is_floating_point()]
                if ints:
                    topk_idx = next((t for t in ints if t.dim() >= 2), ints[0])
                if floats:
                    logits = max(floats, key=lambda t: int(t.size(-1)) if t.dim() > 0 else -1)
            elif torch.is_tensor(output):
                logits = output
            if torch.is_tensor(topk_idx):
                if topk_idx.dim() == 3:
                    topk_idx = topk_idx.reshape(-1, topk_idx.size(-1))
                topk_buckets[layer_idx].append(topk_idx.detach().cpu().to(torch.long))
            if torch.is_tensor(logits):
                if logits.dim() == 3:
                    logits = logits.reshape(-1, logits.size(-1))
                logits_buckets[layer_idx].append(logits.detach().cpu().to(torch.float32))
        return hook

    handles = []
    for li in layers:
        moe = moe_by_layer[int(li)]
        handles.append(moe.register_forward_pre_hook(input_hook(int(li))))
        handles.append(_resolve_router_module(moe).register_forward_hook(router_hook(int(li))))

    try:
        with torch.no_grad():
            for batch in calib:
                input_ids = batch[0] if isinstance(batch, (tuple, list)) else batch
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
                input_ids = input_ids.to(next(model.parameters()).device)
                model(input_ids=input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    layer_inputs = {li: torch.cat(x_buckets[li], dim=0) for li in layers}
    router_info = {}
    for li in layers:
        info = {}
        if topk_buckets[li]:
            info['topk_idx'] = torch.cat(topk_buckets[li], dim=0)
        if logits_buckets[li]:
            info['logits'] = torch.cat(logits_buckets[li], dim=0)
        router_info[li] = info
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return layer_inputs, router_info

def _infer_topk(base_model_path: Path) -> int:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(str(base_model_path), trust_remote_code=True)
    for attr in ('num_experts_per_tok', 'num_experts_per_token', 'top_k'):
        val = getattr(cfg, attr, None)
        if isinstance(val, int) and val > 0:
            return int(val)
    return 2

def _router_topk_indices_from_info(router_info: Dict[str, torch.Tensor], topk: int) -> torch.Tensor:
    if 'topk_idx' in router_info:
        idx = router_info['topk_idx'].to(torch.long)
        return idx.unsqueeze(1) if idx.dim() == 1 else idx
    if 'logits' not in router_info:
        raise RuntimeError('router_info has neither topk_idx nor logits for I3 cache.')
    logits = router_info['logits'].to(torch.float32)
    _, idx = torch.topk(torch.softmax(logits, dim=-1), k=min(int(topk), int(logits.size(-1))), dim=-1)
    return idx.to(torch.long)

def build_i3_sparse_activation_cache(
    *,
    model_path: str,
    layers: List[int],
    index_data: dict,
    base_model_path: Path,
    activation: str,
    calib_data_name: str,
    calib_num_samples: int,
    calib_seqlen: int,
    seed: int = 0,
    cache_kinds_by_layer: Optional[Dict[int, List[str]]] = None,
) -> Dict[int, Dict[str, Dict[int, torch.Tensor]]]:
    print(f'[I3][Cache] build start data={calib_data_name} samples={calib_num_samples} seqlen={calib_seqlen} layers={layers}')
    layer_inputs, router_info = _collect_layer_inputs_and_router_info(
        model_path=model_path,
        layers=[int(x) for x in layers],
        calib_num_samples=int(calib_num_samples),
        calib_seqlen=int(calib_seqlen),
        seed=int(seed),
        calib_data_name=calib_data_name,
    )
    topk = _infer_topk(Path(model_path))
    out = {}
    with torch.no_grad():
        for li in layers:
            li = int(li)
            requested = set(cache_kinds_by_layer.get(li, ['upgate', 'down'])) if cache_kinds_by_layer else {'upgate', 'down'}
            x = layer_inputs[li].to(torch.float32).cpu()
            top_idx = _router_topk_indices_from_info(router_info[li], topk=topk)
            n_tok = min(int(x.size(0)), int(top_idx.size(0)))
            x, top_idx = x[:n_tok], top_idx[:n_tok]
            gate_w, eids = _load_expert_weight_stack(index_data, base_model_path, li, 'gate_proj')
            up_w = None
            if 'down' in requested:
                up_w, _ = _load_expert_weight_stack(index_data, base_model_path, li, 'up_proj')
            upgate_cache, down_cache = {}, {}
            for local_ei, expert_id in enumerate(eids):
                mask = (top_idx == int(expert_id))
                if not mask.any():
                    if 'upgate' in requested:
                        upgate_cache[int(expert_id)] = torch.empty(0, int(x.size(1)), dtype=torch.bfloat16)
                    if 'down' in requested:
                        down_cache[int(expert_id)] = torch.empty(0, int(gate_w.size(1)), dtype=torch.bfloat16)
                    continue
                token_idx, _slot = torch.where(mask)
                xin = x[token_idx.to(torch.long)].to(torch.float32)
                if 'upgate' in requested:
                    upgate_cache[int(expert_id)] = xin.cpu().to(torch.bfloat16).contiguous()
                if 'down' in requested:
                    gout = F.linear(xin, gate_w[local_ei])
                    uout = F.linear(xin, up_w[local_ei])
                    h = (F.silu(gout) if activation == 'silu' else torch.tanh(gout)) * uout
                    down_cache[int(expert_id)] = h.cpu().to(torch.bfloat16).contiguous()
            out[li] = {}
            if 'upgate' in requested:
                out[li]['upgate'] = upgate_cache
            if 'down' in requested:
                out[li]['down'] = down_cache
            active = sum(int(t.size(0)) for d in out[li].values() for t in d.values())
            print(f'[I3][Cache] layer={li} kinds={sorted(requested)} routed_entries={active}')
    print('[I3][Cache] build done')
    return out

def _apply_basis_activation_for_i3(b: torch.Tensor, activation: str) -> torch.Tensor:
    return F.silu(b) if activation == 'silu' else torch.tanh(b)

def _sample_i3_indices(x_cache: torch.Tensor, tokens_batch: int) -> torch.Tensor:
    t = int(x_cache.size(0))
    if t <= 0:
        raise ValueError('Cannot sample from empty I3 cache.')
    k = int(tokens_batch)
    if k <= 0:
        k = t
    if t >= k:
        return torch.randperm(t, device=x_cache.device)[:k]
    return torch.randint(0, t, (k,), device=x_cache.device)

def _pcc_stats_from_y_batch(y_batch: torch.Tensor, eps: float = 1e-12):
    e, m = int(y_batch.size(0)), int(y_batch.size(1))
    if e <= 0 or m <= 1:
        z = y_batch.sum() * 0.0
        empty = torch.empty(0, device=y_batch.device, dtype=torch.float32)
        return z, empty, empty
    v = y_batch.reshape(e, m, -1).float()
    v = v - v.mean(dim=-1, keepdim=True)
    v = v / v.norm(dim=-1, keepdim=True).clamp_min(float(eps))
    corr = torch.matmul(v, v.transpose(-1, -2))
    mask = ~torch.eye(m, dtype=torch.bool, device=y_batch.device)
    off = corr[:, mask]
    sq = (off * off).mean(dim=1)
    abs_off = off.detach().abs()
    return sq, abs_off.mean(dim=1), abs_off.max(dim=1).values

def _i3_pcc_loss_for_module(
    *,
    A: torch.Tensor,
    B: torch.Tensor,
    mode: str,
    activation: str,
    module_cache: Optional[List[torch.Tensor]],
    tokens_batch: int,
    expert_chunk_size: int,
):
    if module_cache is None:
        return None, {'pcc_abs_mean': float('nan'), 'pcc_abs_max': float('nan'), 'pcc_sq_mean': float('nan')}
    active = [(i, x) for i, x in enumerate(module_cache) if torch.is_tensor(x) and int(x.size(0)) > 0]
    if not active:
        z = B.sum() * 0.0
        return z, {'pcc_abs_mean': float('nan'), 'pcc_abs_max': float('nan'), 'pcc_sq_mean': float('nan')}
    device = B.device
    B_used = _apply_basis_activation_for_i3(B, activation)
    if int(tokens_batch) <= 0 or int(expert_chunk_size) < 0:
        expert_chunk_size = 1
    elif int(expert_chunk_size) == 0:
        expert_chunk_size = len(active)
    expert_chunk_size = max(1, int(expert_chunk_size))
    all_sq, all_abs_mean, all_abs_max = [], [], []
    for s0 in range(0, len(active), expert_chunk_size):
        chunk = active[s0:s0 + expert_chunk_size]
        xs, eis = [], []
        for ei, x_cache in chunk:
            idx = _sample_i3_indices(x_cache, int(tokens_batch))
            xs.append(x_cache[idx].to(device=device, dtype=torch.float32, non_blocking=True))
            eis.append(int(ei))
        same_t = len({int(x.size(0)) for x in xs}) == 1
        if not same_t:
            for x, ei in zip(xs, eis):
                xb = x.unsqueeze(0)
                if mode == 'right':
                    yb = torch.einsum('etd,mkd->emtk', xb, B_used)
                else:
                    ae = A[torch.tensor([ei], device=device, dtype=torch.long)]
                    z = torch.bmm(xb, ae.transpose(1, 2)).detach()
                    yb = torch.einsum('etk,mrk->emtr', z, B_used)
                sq, am, ax = _pcc_stats_from_y_batch(yb)
                all_sq.append(sq); all_abs_mean.append(am); all_abs_max.append(ax)
            continue
        x_batch = torch.stack(xs, dim=0)
        if mode == 'right':
            y_batch = torch.einsum('etd,mkd->emtk', x_batch, B_used)
        elif mode == 'left':
            ae = A[torch.tensor(eis, device=device, dtype=torch.long)]
            z_batch = torch.bmm(x_batch, ae.transpose(1, 2)).detach()
            y_batch = torch.einsum('etk,mrk->emtr', z_batch, B_used)
        else:
            raise ValueError(mode)
        sq, am, ax = _pcc_stats_from_y_batch(y_batch)
        all_sq.append(sq); all_abs_mean.append(am); all_abs_max.append(ax)
    sq_all = torch.cat(all_sq, dim=0)
    am_all = torch.cat(all_abs_mean, dim=0)
    ax_all = torch.cat(all_abs_max, dim=0)
    loss = sq_all.mean()
    return loss, {
        'pcc_abs_mean': float(am_all.mean().item()),
        'pcc_abs_max': float(ax_all.max().item()),
        'pcc_sq_mean': float(loss.detach().item()),
    }

def _reindex_i3_module_cache(module_cache, expert_ids: List[int]) -> Optional[List[torch.Tensor]]:
    if module_cache is None:
        return None
    if isinstance(module_cache, list):
        if len(module_cache) != len(expert_ids):
            raise ValueError(f'I3 cache list length mismatch: {len(module_cache)} vs {len(expert_ids)}')
        return module_cache
    if not isinstance(module_cache, dict):
        raise TypeError(f'Unsupported I3 module cache type: {type(module_cache)}')
    sample = next((v for v in module_cache.values() if torch.is_tensor(v)), None)
    if sample is None:
        return [torch.empty(0, 0, dtype=torch.bfloat16) for _ in expert_ids]
    empty_shape = list(sample.shape); empty_shape[0] = 0
    return [module_cache.get(int(eid), torch.empty(*empty_shape, dtype=sample.dtype)) for eid in expert_ids]

def _move_i3_module_cache_to_device(module_cache: Optional[List[torch.Tensor]], device: torch.device) -> Optional[List[torch.Tensor]]:
    if module_cache is None:
        return None
    return [t.to(device=device, non_blocking=True) for t in module_cache]
