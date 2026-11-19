import os

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN

# Relative import: at save time train.py copies the base model's own
# modeling_deepseek.py / configuration_deepseek.py next to this file so this
# resolves under HF's trust_remote_code dynamic module loading.
from .modeling_deepseek import DeepseekConfig, DeepseekForCausalLM, DeepseekMoE

WAB_MODULES = ('gate_proj', 'up_proj', 'down_proj')

# Apply A/B to the activations instead of first rebuilding the dense (r, d) expert
# weight. Mathematically identical; set ACED_WAB_FACTORED=1 to enable. Off by default
# so the default numerics stay bit-for-bit what verify_reconstruction checked.
WAB_FACTORED_FORWARD = os.environ.get('ACED_WAB_FACTORED', '0') == '1'

# Run every activated expert in one batched pass (bmm over the expert dimension) instead of
# one python-loop iteration per expert. Mathematically identical; set ACED_WAB_BATCHED=1.
# At decode each expert owns only a handful of tokens, so the padding this needs is nearly
# free and the win is the removed per-expert kernel launches.
WAB_BATCHED_FORWARD = os.environ.get('ACED_WAB_BATCHED', '0') == '1'
# fall back to the per-expert loop when padding would inflate the token count more than this
WAB_BATCHED_MAX_PAD = float(os.environ.get('ACED_WAB_BATCHED_MAX_PAD', '2.0'))
# ...and whenever the busiest expert holds more than this many tokens. Decode gives each expert
# only a handful (batch*top_k/n_experts), while prefill gives it hundreds, where the padded
# (E, t_max, d) buffers get large enough to cost more than the saved launches.
WAB_BATCHED_MAX_TOKENS = int(os.environ.get('ACED_WAB_BATCHED_MAX_TOKENS', '32'))


class DeepseekWABProjection(nn.Module):
    def __init__(self, n_experts: int, r: int, d: int, num_b: int, rank: int, mode: str, activation: str):
        super().__init__()
        if mode not in ('left', 'right'):
            raise ValueError(f'Unknown wab_mode: {mode}')
        self.mode = mode
        self.activation = activation
        if mode == 'right':
            self.A_params = nn.Parameter(torch.zeros(n_experts, r, rank))
            self.B_params = nn.Parameter(torch.zeros(num_b, rank, d))
        else:
            self.A_params = nn.Parameter(torch.zeros(n_experts, rank, d))
            self.B_params = nn.Parameter(torch.zeros(num_b, r, rank))
        self.w_params = nn.Parameter(torch.zeros(n_experts, num_b))
        self.register_buffer('target_std', torch.tensor(1.0))

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x) if self.activation == 'silu' else torch.tanh(x)

    def expert_weight(self, expert_idx: int) -> torch.Tensor:
        alpha = torch.softmax(self.w_params[expert_idx], dim=0)
        if self.mode == 'right':
            weighted_b = torch.einsum('m,mkd->kd', alpha, self.B_params)
            c = self._activate(weighted_b)
            w_hat = self.A_params[expert_idx] @ c
        else:
            weighted_b = torch.einsum('m,mrk->rk', alpha, self.B_params)
            c = self._activate(weighted_b)
            w_hat = c @ self.A_params[expert_idx]
        return w_hat * self.target_std

    def apply_experts_batched(self, expert_ids: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Batched form of apply_expert: x is (E, T, d) for the E experts in `expert_ids`.

        The per-expert basis mix, activation and both GEMMs are each issued once for all E
        experts instead of once per expert, which is what removes the launch overhead.
        """
        alpha = torch.softmax(self.w_params[expert_ids], dim=-1)          # (E, m)
        A = self.A_params[expert_ids]
        if self.mode == 'right':
            c = self._activate(torch.einsum('em,mkd->ekd', alpha, self.B_params))   # (E, rank, d)
            h = torch.bmm(x, c.transpose(1, 2))                                     # (E, T, rank)
            y = torch.bmm(h, A.transpose(1, 2))                                     # (E, T, r)
        else:
            c = self._activate(torch.einsum('em,mrk->erk', alpha, self.B_params))   # (E, r, rank)
            h = torch.bmm(x, A.transpose(1, 2))                                     # (E, T, rank)
            y = torch.bmm(h, c.transpose(1, 2))                                     # (E, T, r)
        return y * self.target_std

    def apply_expert(self, expert_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Same result as F.linear(x, expert_weight(idx)) without ever materialising the
        (r, d) weight: the two factors are applied to the activations instead.

        right: W = A_e C  =>  x W^T = (x C^T) A_e^T
        left : W = C A_e  =>  x W^T = (x A_e^T) C^T
        """
        alpha = torch.softmax(self.w_params[expert_idx], dim=0)
        if self.mode == 'right':
            c = self._activate(torch.einsum('m,mkd->kd', alpha, self.B_params))   # (rank, d)
            h = F.linear(x, c)                                                    # (T, rank)
            y = F.linear(h, self.A_params[expert_idx])                            # (T, r)
        else:
            c = self._activate(torch.einsum('m,mrk->rk', alpha, self.B_params))   # (r, rank)
            h = F.linear(x, self.A_params[expert_idx])                            # (T, rank)
            y = F.linear(h, c)                                                    # (T, r)
        return y * self.target_std


class DeepseekWABMoE(nn.Module):
    def __init__(self, config: DeepseekConfig, old_moe: DeepseekMoE):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = int(config.num_experts_per_tok)
        self.n_routed_experts = int(config.n_routed_experts)
        self.gate = old_moe.gate
        self.shared_experts = getattr(old_moe, 'shared_experts', None)

        hidden_size = int(config.hidden_size)
        ffn_dim = int(config.moe_intermediate_size)
        num_b = int(config.num_b)
        rank = config.wab_rank
        mode = config.wab_mode
        activation = str(config.wab_activation)
        for name in WAB_MODULES:
            if name not in rank or name not in mode:
                raise KeyError(f'config.wab_rank/wab_mode missing entry for {name}')

        self.gate_proj_wab = DeepseekWABProjection(self.n_routed_experts, ffn_dim, hidden_size, num_b, int(rank['gate_proj']), mode['gate_proj'], activation)
        self.up_proj_wab = DeepseekWABProjection(self.n_routed_experts, ffn_dim, hidden_size, num_b, int(rank['up_proj']), mode['up_proj'], activation)
        self.down_proj_wab = DeepseekWABProjection(self.n_routed_experts, hidden_size, ffn_dim, num_b, int(rank['down_proj']), mode['down_proj'], activation)
        self.act_fn = ACT2FN[config.hidden_act]

    def _forward_batched(self, hs_flat, flat_topk_idx, flat_expert_weights, h):
        """One bmm per projection over all activated experts, instead of a loop over experts.

        Tokens are padded to the busiest expert's count; returns None (so the caller falls back
        to the per-expert loop) when that padding would cost more than it saves.
        """
        counts = flat_topk_idx.bincount(minlength=self.n_routed_experts)
        active = torch.nonzero(counts, as_tuple=True)[0]
        if active.numel() == 0:
            return torch.zeros_like(hs_flat)
        cnt = counts[active]
        t_max = int(cnt.max())
        if (t_max > WAB_BATCHED_MAX_TOKENS
                or t_max * active.numel() > WAB_BATCHED_MAX_PAD * int(flat_topk_idx.numel())):
            return None

        order = flat_topk_idx.argsort()
        token_idxs = order // self.num_experts_per_tok
        starts = (torch.cumsum(counts, 0) - counts)[active]
        ar = torch.arange(t_max, device=hs_flat.device)
        pos = (starts[:, None] + ar[None, :]).clamp_(max=int(order.numel()) - 1)   # (E, T)
        valid = (ar[None, :] < cnt[:, None])                                       # (E, T)

        tok = token_idxs[pos]                                                      # (E, T)
        x = hs_flat[tok]                                                           # (E, T, h)
        w = flat_expert_weights[order][pos]                                        # (E, T, 1)

        g = self.gate_proj_wab.apply_experts_batched(active, x)
        u = self.up_proj_wab.apply_experts_batched(active, x)
        y = self.down_proj_wab.apply_experts_batched(active, self.act_fn(g) * u)
        y = y * w * valid.unsqueeze(-1)          # padded slots contribute nothing

        out = torch.zeros_like(hs_flat)
        out.index_add_(0, tok.reshape(-1), y.reshape(-1, h).to(hs_flat.dtype))
        return out

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        bsz, seq_len, h = hidden_states.shape
        hs_flat = hidden_states.view(-1, h)

        logits = F.linear(hs_flat, self.gate.weight, None)
        scores = logits.softmax(dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.num_experts_per_tok, dim=-1, sorted=False)
        if self.num_experts_per_tok > 1 and self.gate.norm_topk_prob:
            denom = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denom
        topk_weight = topk_weight.to(hs_flat.dtype)

        flat_topk_idx = topk_idx.view(-1)
        flat_expert_weights = topk_weight.view(-1, 1)

        if WAB_BATCHED_FORWARD:
            y = self._forward_batched(hs_flat, flat_topk_idx, flat_expert_weights, h)
            if y is not None:
                y = y.view(*orig_shape)
                if self.shared_experts is not None:
                    y = y + self.shared_experts(identity)
                return y
        expert_cache = torch.zeros_like(hs_flat)
        idxs = flat_topk_idx.argsort()
        tokens_per_expert = flat_topk_idx.bincount(minlength=self.n_routed_experts).cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_experts_per_tok
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = hs_flat[exp_token_idx]
            if WAB_FACTORED_FORWARD:
                g = self.gate_proj_wab.apply_expert(i, expert_tokens)
                u = self.up_proj_wab.apply_expert(i, expert_tokens)
                expert_out = self.down_proj_wab.apply_expert(i, self.act_fn(g) * u)
            else:
                w_gate = self.gate_proj_wab.expert_weight(i)
                w_up = self.up_proj_wab.expert_weight(i)
                w_down = self.down_proj_wab.expert_weight(i)
                expert_out = F.linear(self.act_fn(F.linear(expert_tokens, w_gate)) * F.linear(expert_tokens, w_up), w_down)
            expert_out = expert_out * flat_expert_weights[idxs[start_idx:end_idx]]
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, h), expert_out.to(hs_flat.dtype), reduce='sum')

        y = expert_cache.view(*orig_shape)
        if self.shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y


class DeepseekWABForCausalLM(DeepseekForCausalLM):
    def __init__(self, config: DeepseekConfig):
        super().__init__(config)
        for layer in self.model.layers:
            if isinstance(layer.mlp, DeepseekMoE):
                layer.mlp = DeepseekWABMoE(config, layer.mlp)
