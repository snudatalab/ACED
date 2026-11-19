import torch
import torch.nn.functional as F
from typing import List, Tuple


MODULES = ['up_proj', 'gate_proj', 'down_proj']

MATRIX_ALIASES = {
    'gate_proj': ['w1'],
    'up_proj': ['w3'],
    'down_proj': ['w2'],
    'w1': ['gate_proj'],
    'w3': ['up_proj'],
    'w2': ['down_proj'],
}

# Fill these in with your own local checkpoint paths to use --model {mixtral,qwen3,deepseek}
# as a shorthand; left unset by default so a missing path fails loudly instead of pointing at
# someone else's machine. You can always skip this and use --base_model <path> directly instead.
DEFAULT_BASE_MODELS = {
    'mixtral': None,
    'qwen3': None,
    'deepseek': None,
}

def _build_group_spans(n: int, num_b: int) -> List[Tuple[int, int]]:
    if num_b <= 0:
        raise ValueError(f'num_b must be > 0, got {num_b}')
    if num_b > n:
        raise ValueError(f'num_b={num_b} cannot exceed num_experts={n}')
    base = n // num_b
    rem = n % num_b
    spans = []
    s = 0
    for gi in range(num_b):
        size = base + (1 if gi < rem else 0)
        spans.append((s, s + size))
        s += size
    return spans

def _groups_by_index(n: int, num_b: int) -> List[List[int]]:
    return [list(range(s, e)) for s, e in _build_group_spans(n, num_b)]

def _mode_by_shape(w: torch.Tensor) -> str:
    d_out, d_in = int(w.size(0)), int(w.size(1))
    if d_out > d_in:
        return 'left'
    if d_out < d_in:
        return 'right'
    return 'right'

def _estimate_compression_ratio(n: int, r: int, d: int, num_b: int, rank: int, mode: str) -> float:
    original = float(n * r * d)
    if mode == 'right':
        compressed = float((n * r * rank) + (num_b * rank * d) + (n * num_b))
    else:
        compressed = float((n * d * rank) + (num_b * r * rank) + (n * num_b))
    return 1.0 - (compressed / max(original, 1.0))

def _select_rank_from_compression_ratio(n: int, r: int, d: int, num_b: int, target_compression_ratio: float, mode: str) -> int:
    max_rank = min(r, d)
    target = float(target_compression_ratio)
    keep_ratio = 1.0 - target
    numer = (keep_ratio * n * r * d) - (n * num_b)
    denom = ((n * r) + (num_b * d)) if mode == 'right' else ((n * d) + (num_b * r))
    rank = max(1, min(int(numer // max(denom, 1.0)), max_rank))
    while rank < max_rank and _estimate_compression_ratio(n, r, d, num_b, rank + 1, mode) >= target:
        rank += 1
    while rank > 1 and _estimate_compression_ratio(n, r, d, num_b, rank, mode) < target:
        rank -= 1
    return int(rank)

def _forward_from_params(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, mode: str, activation: str) -> torch.Tensor:
    alpha = torch.softmax(w, dim=1)
    if mode == 'right':
        weighted_b = torch.einsum('nm,mkd->nkd', alpha, b)
        c = F.silu(weighted_b) if activation == 'silu' else torch.tanh(weighted_b)
        return torch.bmm(a, c)
    weighted_b = torch.einsum('nm,mrk->nrk', alpha, b)
    c = F.silu(weighted_b) if activation == 'silu' else torch.tanh(weighted_b)
    return torch.bmm(c, a)

def _forward_from_params_indices(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, indices: torch.Tensor, mode: str, activation: str) -> torch.Tensor:
    a_i = a[indices]
    alpha_i = torch.softmax(w[indices], dim=1)
    if mode == 'right':
        weighted_b = torch.einsum('nm,mkd->nkd', alpha_i, b)
        c = F.silu(weighted_b) if activation == 'silu' else torch.tanh(weighted_b)
        return torch.bmm(a_i, c)
    weighted_b = torch.einsum('nm,mrk->nrk', alpha_i, b)
    c = F.silu(weighted_b) if activation == 'silu' else torch.tanh(weighted_b)
    return torch.bmm(c, a_i)

def _init_params_grouped(target_w: torch.Tensor, groups: List[List[int]], num_b: int, rank: int, mode: str):
    n, r, d = int(target_w.size(0)), int(target_w.size(1)), int(target_w.size(2))
    if len(groups) != int(num_b):
        raise ValueError(f'groups({len(groups)}) != num_b({num_b})')
    initial_w = torch.zeros(n, num_b, device=target_w.device, dtype=torch.float32)
    if mode == 'right':
        initial_a = torch.zeros(n, r, rank, device=target_w.device, dtype=torch.float32)
        basis = []
        for gi, group in enumerate(groups):
            mat = target_w[group].reshape(-1, d).to(torch.float32)
            u, s, vt = torch.linalg.svd(mat, full_matrices=False)
            group_a = u[:, :rank] @ torch.diag(s[:rank])
            basis.append(vt[:rank, :].unsqueeze(0))
            for wi, ei in enumerate(group):
                initial_a[int(ei)] = group_a[wi * r:(wi + 1) * r, :]
                initial_w[int(ei), gi] = 1.0
        return initial_a, torch.cat(basis, dim=0), initial_w
    initial_a = torch.zeros(n, rank, d, device=target_w.device, dtype=torch.float32)
    basis = []
    for gi, group in enumerate(groups):
        gm = target_w[group]
        mat = torch.cat([gm[i] for i in range(int(gm.size(0)))], dim=1).to(torch.float32)
        u, s, vt = torch.linalg.svd(mat, full_matrices=False)
        basis.append(u[:, :rank].unsqueeze(0))
        svt = torch.diag(s[:rank]) @ vt[:rank, :]
        for wi, ei in enumerate(group):
            initial_a[int(ei)] = svt[:, wi * d:(wi + 1) * d]
            initial_w[int(ei), gi] = 1.0
    return initial_a, torch.cat(basis, dim=0), initial_w

def _train_and_reconstruct_grouped(
    target_w: torch.Tensor,
    groups: List[List[int]],
    num_b: int,
    rank: int,
    mode: str,
    activation: str,
    epochs: int,
    lr: float,
    init_std: bool,
):
    target_w_f = target_w.to(torch.float32)
    target_std = target_w_f.std().clamp_min(1e-8)
    init_target_w = (target_w_f / target_std) if bool(init_std) else target_w_f
    z = target_w_f / target_std
    a0, b0, w0 = _init_params_grouped(init_target_w, groups, num_b, rank, mode)
    a = torch.nn.Parameter(a0)
    b = torch.nn.Parameter(b0)
    w = torch.nn.Parameter(w0)
    optim = torch.optim.Adam([a, b, w], lr=lr)
    best_real = float('inf')
    best_w_hat = None
    for _ep in range(int(epochs)):
        optim.zero_grad()
        out = _forward_from_params(a, b, w, mode, activation)
        loss = F.mse_loss(out, z)
        loss.backward()
        optim.step()
        with torch.no_grad():
            w_hat = out * target_std
            real = float(F.mse_loss(w_hat, target_w_f).item())
            if real < best_real:
                best_real = real
                best_w_hat = w_hat.detach().clone()
    if best_w_hat is None:
        with torch.no_grad():
            best_w_hat = _forward_from_params(a, b, w, mode, activation).detach() * target_std
            best_real = float(F.mse_loss(best_w_hat, target_w_f).item())
    return best_real, best_w_hat

def _candidate_m_values(num_experts: int, num_candidates: int) -> List[int]:
    k = max(1, int(num_candidates))
    if k >= int(num_experts):
        return list(range(1, int(num_experts) + 1))
    step = max(1, int(num_experts) // k)
    vals = list(range(step, int(num_experts) + 1, step))
    if vals[-1] != int(num_experts):
        vals.append(int(num_experts))
    return sorted(set(vals))

def _select_search_layers_arithmetic(layers: List[int], num_search_layers: int) -> List[int]:
    if not layers:
        return []
    k = max(1, int(num_search_layers))
    if k >= len(layers):
        return list(layers)
    if k == 1:
        return [layers[0]]
    idxs = [int(round(i * (len(layers) - 1) / (k - 1))) for i in range(k)]
    idxs[0] = 0
    idxs[-1] = len(layers) - 1
    idxs = sorted(set(idxs))
    if len(idxs) < k:
        used = set(idxs)
        for i in range(len(layers)):
            if i not in used:
                idxs.append(i)
                used.add(i)
            if len(idxs) >= k:
                break
        idxs = sorted(idxs)
    return [layers[i] for i in idxs]
