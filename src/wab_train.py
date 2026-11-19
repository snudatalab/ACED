from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from wab_core import _forward_from_params, _forward_from_params_indices, _init_params_grouped
from calibration import _i3_pcc_loss_for_module


def _train_grouped_export_wab(
    target_w: torch.Tensor,
    groups: List[List[int]],
    num_b: int,
    rank: int,
    mode: str,
    activation: str,
    epochs: int,
    lr: float,
    init_std: bool,
    log_interval: int,
    log_tag: str = '',
    expert_batch_size: int = 0,
    i3_module_cache: Optional[List[torch.Tensor]] = None,
    i3_lambda_pcc: float = 0.0,
    i3_tokens_batch: int = 0,
    i3_expert_chunk_size: int = 8,
):
    target_w_f = target_w.to(torch.float32)
    target_std = target_w_f.std().clamp_min(1e-8)
    init_target_w = (target_w_f / target_std) if bool(init_std) else target_w_f
    z = target_w_f / target_std
    a0, b0, w0 = _init_params_grouped(init_target_w, groups, num_b, rank, mode)
    a = torch.nn.Parameter(a0)
    b = torch.nn.Parameter(b0)
    w = torch.nn.Parameter(w0)
    opt = torch.optim.Adam([a, b, w], lr=float(lr))
    n_experts = int(target_w_f.size(0))
    expert_batch_size = int(expert_batch_size)
    if expert_batch_size <= 0 or expert_batch_size >= n_experts:
        expert_batch_size = n_experts
    chunked = expert_batch_size < n_experts
    use_pcc = i3_module_cache is not None and float(i3_lambda_pcc) > 0.0
    hist = []
    best_real = float('inf')
    best_state = None
    log_interval = max(1, int(log_interval))
    num_epochs = int(epochs)
    for ep in range(num_epochs):
        opt.zero_grad()
        rec_sum = real_sum = mae_sum = 0.0
        for s0 in range(0, n_experts, expert_batch_size):
            e0 = min(s0 + expert_batch_size, n_experts)
            idx = torch.arange(s0, e0, device=target_w_f.device)
            out = _forward_from_params_indices(a, b, w, idx, mode, activation) if chunked else _forward_from_params(a, b, w, mode, activation)
            rec = F.mse_loss(out, z[idx])
            (rec * (float(e0 - s0) / max(float(n_experts), 1.0))).backward()
            with torch.no_grad():
                w_hat = out * target_std
                real = F.mse_loss(w_hat, target_w_f[idx])
                mae = F.l1_loss(w_hat, target_w_f[idx])
                rec_sum += float(rec.item()) * (e0 - s0)
                real_sum += float(real.item()) * (e0 - s0)
                mae_sum += float(mae.item()) * (e0 - s0)
        pcc_value = 0.0
        pcc_stats = {'pcc_abs_mean': float('nan'), 'pcc_abs_max': float('nan'), 'pcc_sq_mean': float('nan')}
        if use_pcc:
            pcc_loss, pcc_stats = _i3_pcc_loss_for_module(
                A=a, B=b, mode=mode, activation=activation,
                module_cache=i3_module_cache,
                tokens_batch=int(i3_tokens_batch),
                expert_chunk_size=int(i3_expert_chunk_size),
            )
            if pcc_loss is not None:
                (float(i3_lambda_pcc) * pcc_loss).backward()
                pcc_value = float(pcc_loss.detach().item())
        opt.step()
        rec_loss = rec_sum / max(n_experts, 1)
        real_mse = real_sum / max(n_experts, 1)
        real_mae = mae_sum / max(n_experts, 1)
        total_loss = rec_loss + (float(i3_lambda_pcc) * pcc_value if use_pcc else 0.0)
        hist.append({
            'epoch': ep + 1,
            'scaled_loss': float(total_loss),
            'total_loss': float(total_loss),
            'rec_loss': float(rec_loss),
            'pcc_loss': float(pcc_value) if use_pcc else float('nan'),
            'lambda_pcc': float(i3_lambda_pcc) if use_pcc else 0.0,
            'pcc_abs_mean': float(pcc_stats['pcc_abs_mean']),
            'pcc_abs_max': float(pcc_stats['pcc_abs_max']),
            'pcc_sq_mean': float(pcc_stats['pcc_sq_mean']),
            'real_mse': float(real_mse),
            'real_mae': float(real_mae),
        })
        if (ep + 1) % log_interval == 0:
            prefix = f'[{log_tag}] ' if log_tag else ''
            print(f'{prefix}Epoch {ep + 1}/{epochs} total={total_loss:.10f} rec={rec_loss:.10f} pcc={pcc_value:.10f} real_mse={real_mse:.10f}')
            if real_mse < best_real:
                best_real = float(real_mse)
                best_state = {
                    'A_params': a.detach().clone(),
                    'B_params': b.detach().clone(),
                    'w_params': w.detach().clone(),
                }

    with torch.no_grad():
        out = _forward_from_params(a, b, w, mode, activation)
        final_real = float(F.mse_loss(out * target_std, target_w_f).item())
        if final_real < best_real:
            best_real = final_real
            best_state = {
                'A_params': a.detach().clone(),
                'B_params': b.detach().clone(),
                'w_params': w.detach().clone(),
            }
    if best_state is None:
        best_state = {
            'A_params': a.detach().clone(),
            'B_params': b.detach().clone(),
            'w_params': w.detach().clone(),
        }

    best_state_cpu = {
        'A_params': best_state['A_params'].detach().cpu().clone(),
        'B_params': best_state['B_params'].detach().cpu().clone(),
        'w_params': best_state['w_params'].detach().cpu().clone(),
    }
    wab_state = {
        'A_params': best_state_cpu['A_params'],
        'B_params': best_state_cpu['B_params'],
        'w_params': best_state_cpu['w_params'],
        'target_std': target_std.detach().cpu(),
        'mobe_mode': mode,
    }
    print(f'[{log_tag}] [Done] best_real_mse={best_real:.10f}')
    return best_real, hist, wab_state

def _reconstruct_from_wab_state(wab_state: Dict[str, torch.Tensor], activation: str, device: torch.device) -> torch.Tensor:
    for req in ('A_params', 'B_params', 'w_params', 'target_std', 'mobe_mode'):
        if req not in wab_state:
            raise KeyError(f"Missing '{req}' in WAB state.")

    mode = str(wab_state['mobe_mode'])
    a = wab_state['A_params'].to(torch.float32).to(device)
    b = wab_state['B_params'].to(torch.float32).to(device)
    w = wab_state['w_params'].to(torch.float32).to(device)
    target_std = wab_state['target_std']
    if not torch.is_tensor(target_std):
        target_std = torch.tensor(float(target_std), dtype=torch.float32)
    target_std = target_std.to(torch.float32).to(device).clamp_min(1e-12)

    w_soft = torch.softmax(w, dim=1)
    if mode == 'right':
        weighted_b = torch.einsum('nm,mkd->nkd', w_soft, b)
        weighted_b = F.silu(weighted_b) if activation == 'silu' else torch.tanh(weighted_b)
        w_hat_scaled = torch.bmm(a, weighted_b)
    elif mode == 'left':
        weighted_b = torch.einsum('nm,mrk->nrk', w_soft, b)
        weighted_b = F.silu(weighted_b) if activation == 'silu' else torch.tanh(weighted_b)
        w_hat_scaled = torch.bmm(weighted_b, a)
    else:
        raise ValueError(f'Unknown mobe_mode: {mode}')
    return (w_hat_scaled * target_std).detach().cpu()
