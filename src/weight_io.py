import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple

from wab_core import MATRIX_ALIASES, MODULES


def _load_index(index_path: Path) -> dict:
    with index_path.open('r', encoding='utf-8') as f:
        return json.load(f)

def _resolve_matrix_type(requested: str, available: set) -> str:
    if requested in available:
        return requested
    for alt in MATRIX_ALIASES.get(requested, []):
        if alt in available:
            return alt
    raise KeyError(f'matrix_type={requested} unavailable. available={sorted(available)}')

def _discover_layers(index_data: dict) -> Dict[str, set]:
    available = {u: set() for u in MODULES}
    for key in index_data.get('weight_map', {}).keys():
        if '.experts.' not in key or not key.endswith('.weight') or 'model.layers.' not in key:
            continue
        li = int(key.split('model.layers.')[1].split('.', 1)[0])
        if '.gate_proj.weight' in key or '.w1.weight' in key:
            available['gate_proj'].add(li)
        if '.up_proj.weight' in key or '.w3.weight' in key:
            available['up_proj'].add(li)
        if '.down_proj.weight' in key or '.w2.weight' in key:
            available['down_proj'].add(li)
    return available

def _is_dense_layer(index_data: dict, layer_i: int) -> bool:
    prefixes = [
        f'model.layers.{layer_i}.mlp.experts.',
        f'model.layers.{layer_i}.block_sparse_moe.experts.',
    ]
    return not any(any(k.startswith(p) for p in prefixes) for k in index_data.get('weight_map', {}))

def _get_layer_proj_dict(index_data: dict, base_dir: Path, layer_i: int, module_type: str):
    expert_prefixes = [
        f'model.layers.{layer_i}.mlp.experts.',
        f'model.layers.{layer_i}.block_sparse_moe.experts.',
    ]
    matched_prefix = None
    available_proj = set()
    for weight_name in index_data['weight_map'].keys():
        for pref in expert_prefixes:
            if weight_name.startswith(pref) and weight_name.endswith('.weight'):
                matched_prefix = pref
                available_proj.add(weight_name[len(pref):].split('.', 1)[1].replace('.weight', ''))
    if matched_prefix is None:
        raise KeyError(f'No expert weights found for layer={layer_i}.')
    resolved = _resolve_matrix_type(module_type, available_proj)
    layer_dict = {}
    shard_cache = {}
    for weight_name, shard_name in index_data['weight_map'].items():
        if not weight_name.startswith(matched_prefix) or not weight_name.endswith(f'.{resolved}.weight'):
            continue
        path = base_dir / shard_name
        if path.suffix == '.safetensors':
            from safetensors import safe_open
            with safe_open(str(path), framework='pt', device='cpu') as f:
                layer_dict[weight_name] = f.get_tensor(weight_name)
        else:
            if path not in shard_cache:
                shard_cache[path] = torch.load(str(path), map_location='cpu')
            layer_dict[weight_name] = shard_cache[path][weight_name]
    return layer_dict, matched_prefix, resolved

def _load_expert_weight_stack(index_data: dict, base_dir: Path, layer_i: int, module_type: str) -> Tuple[torch.Tensor, List[int]]:
    sd, pfx, res = _get_layer_proj_dict(index_data, base_dir, int(layer_i), module_type)
    eids = sorted(set(int(k[len(pfx):].split('.', 1)[0]) for k in sd if k.startswith(pfx) and k.endswith(f'.{res}.weight')))
    weights = torch.stack([sd[f'{pfx}{ei}.{res}.weight'].to(torch.float32) for ei in eids], dim=0).contiguous()
    return weights, eids
