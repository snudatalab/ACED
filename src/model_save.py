import csv
import importlib
import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from wab_train import _reconstruct_from_wab_state

_THIS_DIR = Path(__file__).resolve().parent
_ACED_ROOT = _THIS_DIR.parent
if str(_ACED_ROOT) not in sys.path:
    sys.path.insert(0, str(_ACED_ROOT))


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def _dtype_name(dtype_name: str) -> str:
    return dtype_name if dtype_name in ('float16', 'bfloat16', 'float32') else 'bfloat16'

def _detect_model_family(base_model: str, config) -> str:
    mt = str(getattr(config, 'model_type', '')).lower()
    path = str(base_model).lower()
    if 'mixtral' in mt or 'mixtral' in path:
        return 'mixtral'
    if 'qwen3' in mt or 'qwen3' in path:
        return 'qwen3'
    if 'deepseek' in mt or 'deepseek' in path:
        return 'deepseek'
    return 'other'

WAB_NATIVE_FAMILY_INFO = {
    'mixtral': {'module': 'models.modeling_mixtral_wab', 'class_name': 'MixtralWABForCausalLM', 'expert_attr': 'block_sparse_moe'},
    'qwen3': {'module': 'models.modeling_qwen3_wab', 'class_name': 'Qwen3MoeWABForCausalLM', 'expert_attr': 'mlp'},
    'deepseek': {'module': 'models.modeling_deepseek_wab', 'class_name': 'DeepseekWABForCausalLM', 'expert_attr': 'mlp'},
}

def _merge_wab_to_model(*, base_model: Path, wab_dir: Path, save_dir: Path, modules: List[str], layers: List[int], dtype_name: str, trust_remote_code: bool, activation: str, verify_reconstruction: bool = False, verify_tol: float = 1e-5):
    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
    torch_dtype = dtype_map[_dtype_name(dtype_name)]
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained(str(base_model), trust_remote_code=trust_remote_code)
    family = _detect_model_family(str(base_model), cfg)
    expert_module = 'block_sparse_moe' if family == 'mixtral' else 'mlp'
    proj_to_hf = {'gate_proj': 'w1', 'up_proj': 'w3', 'down_proj': 'w2'} if family == 'mixtral' else {'gate_proj': 'gate_proj', 'up_proj': 'up_proj', 'down_proj': 'down_proj'}
    print(f'[Merge] loading base model: {base_model}')
    model = AutoModelForCausalLM.from_pretrained(str(base_model), torch_dtype=torch_dtype, trust_remote_code=trust_remote_code, low_cpu_mem_usage=False)
    tokenizer = AutoTokenizer.from_pretrained(str(base_model), trust_remote_code=trust_remote_code)
    sd = model.state_dict()
    hf_proj_names = set(proj_to_hf.values())
    all_moe_expert_weight_keys = {
        key for key in sd.keys()
        if f'.{expert_module}.experts.' in key
        and key.endswith('.weight')
        and key.rsplit('.', 2)[-2] in hf_proj_names
    }
    replacement_keys = set()
    verification_tensors = {} if bool(verify_reconstruction) else None
    expected_replacements = 0
    for li in layers:
        for module in modules:
            wab_path = wab_dir / f'model_layers_{li}_mlp_{module}_WAB.pth'
            if not wab_path.exists():
                raise FileNotFoundError(f'Missing WAB for merge: {wab_path}')
            wab = torch.load(str(wab_path), map_location='cpu', weights_only=True)
            w_hat = _reconstruct_from_wab_state(wab, activation=activation, device=torch.device('cpu'))
            hf_proj = proj_to_hf[module]
            expected_replacements += int(w_hat.size(0))
            for eid in range(int(w_hat.size(0))):
                weight_key = f'model.layers.{li}.{expert_module}.experts.{eid}.{hf_proj}.weight'
                if weight_key not in sd:
                    raise KeyError(f'Missing target expert weight in base model state_dict: {weight_key}')
                if tuple(sd[weight_key].shape) != tuple(w_hat[eid].shape):
                    raise RuntimeError(
                        f'Shape mismatch for {weight_key}: base={tuple(sd[weight_key].shape)} wab={tuple(w_hat[eid].shape)}'
                    )
                replacement = w_hat[eid].to(dtype=sd[weight_key].dtype).contiguous()
                sd[weight_key] = replacement
                replacement_keys.add(weight_key)
                if verification_tensors is not None:
                    verification_tensors[weight_key] = replacement.detach().cpu().to(torch.float32)
    if len(replacement_keys) != expected_replacements:
        raise RuntimeError(f'Duplicate or missing replacement keys while merging WABs: expected={expected_replacements} actual={len(replacement_keys)}')
    missing_replacements = sorted(all_moe_expert_weight_keys - replacement_keys)
    unexpected_replacements = sorted(replacement_keys - all_moe_expert_weight_keys)
    if missing_replacements or unexpected_replacements:
        preview_missing = missing_replacements[:5]
        preview_unexpected = unexpected_replacements[:5]
        raise RuntimeError(
            'WAB merge coverage mismatch: '
            f'missing={len(missing_replacements)} preview_missing={preview_missing} '
            f'unexpected={len(unexpected_replacements)} preview_unexpected={preview_unexpected}'
        )
    try:
        model.load_state_dict(sd, assign=True)
    except TypeError:
        model.load_state_dict(sd)
    if bool(verify_reconstruction):
        loaded_sd = model.state_dict()
        verify_limit = float(verify_tol)
        max_abs_by_module = {}
        for weight_key, expected in verification_tensors.items():
            actual = loaded_sd[weight_key].detach().cpu().to(torch.float32)
            max_abs = float((actual - expected).abs().max().item())
            if max_abs > verify_limit:
                raise RuntimeError(f'Model parameter was not replaced by WAB tensor: {weight_key} max_abs_diff={max_abs:.6e}')
            parts = weight_key.split('.')
            layer = parts[2] if len(parts) > 2 else '?'
            proj = parts[-2] if len(parts) > 1 else '?'
            group_key = (layer, proj)
            max_abs_by_module[group_key] = max(max_abs_by_module.get(group_key, 0.0), max_abs)
        for (layer, proj), max_abs in sorted(max_abs_by_module.items(), key=lambda x: (int(x[0][0]) if str(x[0][0]).isdigit() else -1, str(x[0][1]))):
            print(f'[Merge][Verify] layer={layer} proj={proj} max_abs_diff={max_abs:.3e}')
    print(f'[Merge] replaced expert weights: {len(replacement_keys)}')
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(save_dir))
    print(f'[Merge] model saved: {save_dir}')

def _rank_from_wab_state(wab_state: Dict[str, torch.Tensor]) -> int:
    mode = str(wab_state['mobe_mode'])
    a = wab_state['A_params']
    return int(a.size(2)) if mode == 'right' else int(a.size(1))

def _import_wab_native_class(*, family: str, base_model: Path, save_dir: Path, trust_remote_code: bool):
    info = WAB_NATIVE_FAMILY_INFO.get(family)
    if info is None:
        raise ValueError(f'wab_native save format is not supported for model family: {family}')

    if family != 'deepseek':
        module = importlib.import_module(info['module'])
        return module, getattr(module, info['class_name']), info['expert_attr'], None

    # DeepSeek's modeling code is trust_remote_code, not part of the transformers package.
    # Our modeling_deepseek_wab.py relative-imports the base model's own modeling_deepseek.py,
    # so both must sit side by side before we import it. This staging dir is deliberately NOT
    # save_dir: save_pretrained's custom_object_save later copies the class's source file (and,
    # by following its relative imports, modeling_deepseek.py/configuration_deepseek.py too)
    # from wherever it was imported from into save_dir -- if that "wherever" were save_dir
    # itself, the copy would be a no-op self-copy and raise shutil.SameFileError.
    if not trust_remote_code:
        raise ValueError('DeepSeek wab_native save requires --trust_remote_code.')
    stage_dir = save_dir.parent / f'.{save_dir.name}_wab_import_stage'
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    for fname in ('modeling_deepseek.py', 'configuration_deepseek.py'):
        src = base_model / fname
        if not src.exists():
            raise FileNotFoundError(f'DeepSeek wab_native save requires {src} next to the base checkpoint.')
        shutil.copy2(str(src), str(stage_dir / fname))
    shutil.copy2(str(_ACED_ROOT / 'models' / 'modeling_deepseek_wab.py'), str(stage_dir / 'modeling_deepseek_wab.py'))
    init_path = stage_dir / '__init__.py'
    init_path.write_text('', encoding='utf-8')

    pkg_name = f'{stage_dir.name.lstrip(".")}_wab_native_ckpt'
    for mod_name in list(sys.modules):
        if mod_name == pkg_name or mod_name.startswith(pkg_name + '.'):
            del sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(pkg_name, str(init_path), submodule_search_locations=[str(stage_dir)])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = pkg
    spec.loader.exec_module(pkg)
    module = importlib.import_module(f'{pkg_name}.modeling_deepseek_wab')
    return module, getattr(module, info['class_name']), info['expert_attr'], stage_dir

def _save_wab_native_model(
    *, base_model: Path, wab_dir: Path, save_dir: Path, modules: List[str], layers: List[int],
    dtype_name: str, trust_remote_code: bool, activation: str, mode_star: Dict[str, str], num_b: int,
    verify_reconstruction: bool = False, verify_tol: float = 1e-5,
):
    dtype_map = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}
    torch_dtype = dtype_map[_dtype_name(dtype_name)]
    from transformers import AutoConfig, AutoTokenizer
    base_cfg = AutoConfig.from_pretrained(str(base_model), trust_remote_code=trust_remote_code)
    family = _detect_model_family(str(base_model), base_cfg)

    print(f'[WABNative] loading WAB checkpoints and validating per-module rank/mode consistency: family={family}')
    wab_states: Dict[Tuple[int, str], Dict[str, torch.Tensor]] = {}
    rank_per_module: Dict[str, int] = {}
    for u in modules:
        for li in layers:
            wab_path = wab_dir / f'model_layers_{li}_mlp_{u}_WAB.pth'
            if not wab_path.exists():
                raise FileNotFoundError(f'Missing WAB for wab_native save: {wab_path}')
            wab = torch.load(str(wab_path), map_location='cpu', weights_only=True)
            mode = str(wab['mobe_mode'])
            if mode != mode_star[u]:
                raise RuntimeError(f'mobe_mode mismatch for module={u} layer={li}: wab={mode} aced_result={mode_star[u]}')
            rank = _rank_from_wab_state(wab)
            if u not in rank_per_module:
                rank_per_module[u] = rank
            elif rank_per_module[u] != rank:
                raise RuntimeError(
                    f'Rank mismatch across layers for module={u}: layer={li} rank={rank} != '
                    f'expected {rank_per_module[u]} (wab_native save assumes one fixed rank per module across all layers).'
                )
            wab_states[(int(li), u)] = wab

    module, WABClass, expert_attr, stage_dir = _import_wab_native_class(
        family=family, base_model=base_model, save_dir=save_dir, trust_remote_code=trust_remote_code,
    )

    if family == 'deepseek':
        ConfigClass = module.DeepseekConfig
        cfg = ConfigClass(**base_cfg.to_dict())
        ConfigClass.register_for_auto_class()
    else:
        cfg = base_cfg
    cfg.num_b = int(num_b)
    cfg.wab_rank = {u: int(rank_per_module[u]) for u in modules}
    cfg.wab_mode = dict(mode_star)
    cfg.wab_activation = str(activation)
    cfg.architectures = [WABClass.__name__]
    WABClass.register_for_auto_class('AutoModelForCausalLM')

    print(f'[WABNative] loading base model with WAB-native architecture: {base_model}')
    # The base checkpoint's dense expert weights and the model's *_wab parameters are expected
    # to mismatch here by design (that's the whole point -- experts get replaced with WAB
    # params right below) -- suppress transformers' verbose missing/unexpected key dump for it.
    import transformers as _tf
    _prev_verbosity = _tf.logging.get_verbosity()
    _tf.logging.set_verbosity_error()
    try:
        model = WABClass.from_pretrained(
            str(base_model), config=cfg, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code, low_cpu_mem_usage=False,
        )
    finally:
        _tf.logging.set_verbosity(_prev_verbosity)
    tokenizer = AutoTokenizer.from_pretrained(str(base_model), trust_remote_code=trust_remote_code)

    print(f'[WABNative] copying trained WAB parameters into {len(wab_states)} (layer, module) submodules')
    verification: Dict[Tuple[int, str], Dict[str, torch.Tensor]] = {} if bool(verify_reconstruction) else None
    for (li, u), wab in wab_states.items():
        layer_module = model.model.layers[li]
        expert_block = getattr(layer_module, expert_attr)
        proj = getattr(expert_block, f'{u}_wab')
        with torch.no_grad():
            proj.A_params.copy_(wab['A_params'].to(torch_dtype))
            proj.B_params.copy_(wab['B_params'].to(torch_dtype))
            proj.w_params.copy_(wab['w_params'].to(torch_dtype))
            target_std = wab['target_std']
            if not torch.is_tensor(target_std):
                target_std = torch.tensor(float(target_std))
            proj.target_std.copy_(target_std.to(torch_dtype))
        if verification is not None:
            verification[(li, u)] = wab

    if verification is not None:
        verify_limit = float(verify_tol)
        max_abs_by_module = {}
        for (li, u), wab in verification.items():
            # Round the reference reconstruction to the same storage dtype as what actually got
            # saved, so this check isolates real wiring bugs from expected dtype-rounding noise
            # (both sides are computed in fp32 arithmetic either way).
            target_std = wab['target_std']
            if not torch.is_tensor(target_std):
                target_std = torch.tensor(float(target_std))
            expected_wab = {
                'A_params': wab['A_params'].to(torch_dtype).to(torch.float32),
                'B_params': wab['B_params'].to(torch_dtype).to(torch.float32),
                'w_params': wab['w_params'].to(torch_dtype).to(torch.float32),
                'target_std': target_std.to(torch_dtype).to(torch.float32),
                'mobe_mode': wab['mobe_mode'],
            }
            expected = _reconstruct_from_wab_state(expected_wab, activation=activation, device=torch.device('cpu')).to(torch.float32)

            layer_module = model.model.layers[li]
            proj = getattr(getattr(layer_module, expert_attr), f'{u}_wab')
            actual_wab = {
                'A_params': proj.A_params.detach().to(torch.float32).cpu(),
                'B_params': proj.B_params.detach().to(torch.float32).cpu(),
                'w_params': proj.w_params.detach().to(torch.float32).cpu(),
                'target_std': proj.target_std.detach().to(torch.float32).cpu(),
                'mobe_mode': wab['mobe_mode'],
            }
            actual = _reconstruct_from_wab_state(actual_wab, activation=activation, device=torch.device('cpu')).to(torch.float32)

            max_abs = float((actual - expected).abs().max().item())
            if max_abs > verify_limit:
                raise RuntimeError(f'WAB-native reconstruction mismatch: layer={li} module={u} max_abs_diff={max_abs:.6e}')
            max_abs_by_module[(li, u)] = max_abs
        for (li, u), max_abs in sorted(max_abs_by_module.items(), key=lambda x: (x[0][0], x[0][1])):
            print(f'[WABNative][Verify] layer={li} proj={u} max_abs_diff={max_abs:.3e}')

    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(save_dir))
    if stage_dir is not None and stage_dir.exists():
        shutil.rmtree(stage_dir)
    wab_native_params = sum(p.numel() for n, p in model.named_parameters() if n.endswith(('_wab.A_params', '_wab.B_params', '_wab.w_params')))
    print(f'[WABNative] wab parameter count (all layers/modules): {wab_native_params}')
    print(f'[WABNative] model saved: {save_dir}')
