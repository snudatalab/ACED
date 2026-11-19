import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN
from transformers.models.mixtral.modeling_mixtral import MixtralConfig, MixtralForCausalLM

WAB_MODULES = ('gate_proj', 'up_proj', 'down_proj')


class MixtralWABProjection(nn.Module):
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


class MixtralSparseWABBlock(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.hidden_dim = int(config.hidden_size)
        self.ffn_dim = int(config.intermediate_size)
        self.num_experts = int(config.num_local_experts)
        self.top_k = int(config.num_experts_per_tok)

        num_b = int(config.num_b)
        rank = config.wab_rank
        mode = config.wab_mode
        activation = str(config.wab_activation)
        for name in WAB_MODULES:
            if name not in rank or name not in mode:
                raise KeyError(f'config.wab_rank/wab_mode missing entry for {name}')

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.gate_proj_wab = MixtralWABProjection(self.num_experts, self.ffn_dim, self.hidden_dim, num_b, int(rank['gate_proj']), mode['gate_proj'], activation)
        self.up_proj_wab = MixtralWABProjection(self.num_experts, self.ffn_dim, self.hidden_dim, num_b, int(rank['up_proj']), mode['up_proj'], activation)
        self.down_proj_wab = MixtralWABProjection(self.num_experts, self.hidden_dim, self.ffn_dim, num_b, int(rank['down_proj']), mode['down_proj'], activation)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        expert_mask = nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            w1 = self.gate_proj_wab.expert_weight(expert_idx)
            w3 = self.up_proj_wab.expert_weight(expert_idx)
            w2 = self.down_proj_wab.expert_weight(expert_idx)
            current_hidden_states = F.linear(self.act_fn(F.linear(current_state, w1)) * F.linear(current_state, w3), w2)
            current_hidden_states = current_hidden_states * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


class MixtralWABForCausalLM(MixtralForCausalLM):
    def __init__(self, config: MixtralConfig):
        super().__init__(config)
        for layer in self.model.layers:
            layer.block_sparse_moe = MixtralSparseWABBlock(config)
