import argparse
import gc
import json
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import torch

_THIS_DIR = Path(__file__).resolve().parent
_ACED_ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from wab_core import MODULES, _groups_by_index, _select_rank_from_compression_ratio
from weight_io import _load_expert_weight_stack, _load_index
from calibration import _move_i3_module_cache_to_device, _reindex_i3_module_cache, build_i3_sparse_activation_cache
from wab_train import _train_grouped_export_wab
from model_save import _merge_wab_to_model, _save_wab_native_model, _write_csv


def parse_args():
    ap = argparse.ArgumentParser(description='Final ACED training from aced_result.json.')
    ap.add_argument('--aced_result_path', type=str, required=True)
    ap.add_argument('--out_root', type=str, default=None)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--save_model', action='store_true')
    ap.add_argument('--save_format', type=str, default='wab_native', choices=['dense', 'wab_native', 'both'],
                     help="'wab_native' saves only the compressed A/B/w WAB parameters in a trust_remote_code "
                          "custom model that reconstructs each expert on demand (default). "
                          "'dense' merges WAB back into full-size expert weights. 'both' saves both.")
    ap.add_argument('--verify_reconstruction', action='store_true')
    ap.add_argument('--verify_tol', type=float, default=1e-5)
    ap.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    ap.add_argument('--trust_remote_code', action='store_true')
    ap.add_argument('--down_batch_size', type=int, default=0)
    ap.add_argument('--skip_existing_wab', action='store_true')
    ap.add_argument('--m', type=int, default=None, help='Override I2_selected_m and apply the same basis count to all modules.')
    ap.add_argument('--I3_use_pcc', action='store_true')
    ap.add_argument('--I3_calib_data_name', type=str, default=None, choices=['c4', 'wiki2', 'pile'])
    ap.add_argument('--I3_calib_num_samples', type=int, default=None)
    ap.add_argument('--I3_calib_seqlen', type=int, default=None)
    ap.add_argument('--I3_lambda_pcc', type=float, default=None)
    ap.add_argument('--I3_tokens_batch', type=int, default=None)
    ap.add_argument('--I3_expert_chunk_size', type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    result_path = Path(args.aced_result_path).resolve()
    cfg = json.loads(result_path.read_text(encoding='utf-8'))
    base_model_path = Path(cfg['base_model']).resolve()
    index_path = Path(cfg['index_path']).resolve()
    save_root = Path(args.out_root).resolve() if args.out_root else Path(cfg['save_root']).resolve()
    out_wab = save_root / 'wab'
    out_hist = save_root / 'history'
    out_wab.mkdir(parents=True, exist_ok=True)
    out_hist.mkdir(parents=True, exist_ok=True)
    index_data = _load_index(index_path)
    layers = [int(x) for x in cfg['layers']]
    mode_star = {k: str(v) for k, v in cfg['I1_mode_star'].items()}
    selected_m_json = int(cfg['I2_selected_m'])
    if args.m is not None:
        if int(args.m) <= 0:
            raise ValueError(f'--m must be a positive integer, got {args.m}')
        selected_m = int(args.m)
        print(f'[Setup] overriding I2_selected_m with --m={int(args.m)}')
    else:
        selected_m = int(selected_m_json)
    activation = str(cfg['activation'])
    epochs = int(cfg['final_epochs'])
    lr = float(cfg['lr'])
    log_interval = int(cfg.get('log_interval', 100))
    init_std = bool(cfg.get('init_std', False))
    target_compression_ratio = float(cfg['target_compression_ratio'])
    if not (0.0 <= target_compression_ratio < 1.0):
        raise ValueError(f'target_compression_ratio must be in [0, 1), got {target_compression_ratio}')
    use_pcc = bool(args.I3_use_pcc or cfg.get('I3_use_pcc', False))
    calib_data = str(args.I3_calib_data_name if args.I3_calib_data_name is not None else cfg.get('I3_calib_data_name', 'pile'))
    calib_samples = int(args.I3_calib_num_samples if args.I3_calib_num_samples is not None else cfg.get('I3_calib_num_samples', 128))
    calib_seqlen = int(args.I3_calib_seqlen if args.I3_calib_seqlen is not None else cfg.get('I3_calib_seqlen', 1024))
    calib_seed = int(cfg.get('I3_calib_seed', 0))
    lambda_pcc = float(args.I3_lambda_pcc if args.I3_lambda_pcc is not None else cfg.get('I3_lambda_pcc', 0.0))
    tokens_batch = int(args.I3_tokens_batch if args.I3_tokens_batch is not None else cfg.get('I3_tokens_batch', 0))
    expert_chunk_size = int(args.I3_expert_chunk_size if args.I3_expert_chunk_size is not None else cfg.get('I3_expert_chunk_size', 8))
    print(f'[Setup] layers={len(layers)} m={selected_m} I3_use_pcc={use_pcc} lambda={lambda_pcc}')

    pending = []
    for u in MODULES:
        for li in layers:
            wab_path = out_wab / f'model_layers_{li}_mlp_{u}_WAB.pth'
            if args.skip_existing_wab and wab_path.exists():
                continue
            pending.append((u, int(li)))
    print(f'[Resume] pending={len(pending)} skip_existing_wab={bool(args.skip_existing_wab)}')

    i3_cache = None
    if use_pcc and lambda_pcc > 0 and pending:
        kinds_by_layer: Dict[int, List[str]] = {}
        for u, li in pending:
            kinds_by_layer.setdefault(int(li), [])
            kinds_by_layer[int(li)].append('down' if u == 'down_proj' else 'upgate')
        i3_cache = build_i3_sparse_activation_cache(
            model_path=str(base_model_path), layers=sorted(kinds_by_layer.keys()), index_data=index_data,
            base_model_path=base_model_path, activation=activation, calib_data_name=calib_data,
            calib_num_samples=calib_samples, calib_seqlen=calib_seqlen, seed=calib_seed,
            cache_kinds_by_layer=kinds_by_layer,
        )

    optimize_start_time = time.time()
    optimize_started_at = datetime.now().isoformat(timespec='seconds')

    final_rows = []
    trained = skipped = 0
    for u in MODULES:
        mode = mode_star[u]
        m = int(selected_m)
        print(f'[Final] module={u} mode={mode} m={m}')
        for idx_li, li in enumerate(layers, start=1):
            wab_path = out_wab / f'model_layers_{li}_mlp_{u}_WAB.pth'
            hist_path = out_hist / f'model_layers_{li}_mlp_{u}_history.json'
            if args.skip_existing_wab and wab_path.exists():
                skipped += 1
                final_rows.append({'module': u, 'layer': int(li), 'mode': mode, 'm': int(m), 'rank_ab': None, 'best_real_mse': None, 'wab_path': str(wab_path), 'history_path': str(hist_path), 'skipped_existing_wab': True})
                print(f'[Final][Skip] module={u} layer={li}')
                continue
            tw_cpu, eids = _load_expert_weight_stack(index_data, base_model_path, int(li), u)
            n, r, d = int(tw_cpu.size(0)), int(tw_cpu.size(1)), int(tw_cpu.size(2))
            rank = _select_rank_from_compression_ratio(n, r, d, int(m), target_compression_ratio, mode)
            groups = _groups_by_index(n, int(m))
            module_cache = None
            if i3_cache is not None:
                kind = 'down' if u == 'down_proj' else 'upgate'
                module_cache = _reindex_i3_module_cache(i3_cache.get(int(li), {}).get(kind, None), eids)
                module_cache = _move_i3_module_cache_to_device(module_cache, device)
            best_real, hist, wab_state = _train_grouped_export_wab(
                tw_cpu.to(device=device, dtype=torch.float32), groups, int(m), int(rank), mode, activation,
                epochs, lr, init_std, log_interval, log_tag=f'Final {u} L{li}',
                expert_batch_size=(int(args.down_batch_size) if u == 'down_proj' else 0),
                i3_module_cache=module_cache,
                i3_lambda_pcc=lambda_pcc if use_pcc else 0.0,
                i3_tokens_batch=tokens_batch,
                i3_expert_chunk_size=expert_chunk_size,
            )
            torch.save(wab_state, str(wab_path))
            hist_path.write_text(json.dumps({'layer': int(li), 'module': u, 'mode': mode, 'm': int(m), 'rank_ab': int(rank), 'best_real_mse': float(best_real), 'history': hist, 'wab_path': str(wab_path), 'I3_use_pcc': bool(use_pcc), 'I3_lambda_pcc': float(lambda_pcc)}, indent=2), encoding='utf-8')
            trained += 1
            final_rows.append({'module': u, 'layer': int(li), 'mode': mode, 'm': int(m), 'rank_ab': int(rank), 'best_real_mse': float(best_real), 'wab_path': str(wab_path), 'history_path': str(hist_path), 'skipped_existing_wab': False})
            print(f'[Final] module={u} layer={li} progress={idx_li}/{len(layers)} rank={rank} best_real_mse={best_real:.10f}')
            del tw_cpu, module_cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    optimize_end_time = time.time()
    optimize_finished_at = datetime.now().isoformat(timespec='seconds')
    optimize_elapsed_seconds = float(optimize_end_time - optimize_start_time)
    optimize_elapsed_hms = str(timedelta(seconds=int(round(optimize_elapsed_seconds))))
    summary = {
        'aced_result_path': str(result_path),
        'save_root': str(save_root),
        'output_wab_dir': str(out_wab),
        'output_history_dir': str(out_hist),
        'final_trained_rows': int(trained),
        'final_skipped_existing_wab': int(skipped),
        'optimize_started_at': optimize_started_at,
        'optimize_finished_at': optimize_finished_at,
        'optimize_elapsed_seconds': optimize_elapsed_seconds,
        'optimize_elapsed_hms': optimize_elapsed_hms,
        'I1_mode_star': mode_star,
        'I2_selected_m_json': selected_m_json,
        'I2_selected_m': selected_m,
        'm_override': (int(args.m) if args.m is not None else None),
        'm_overridden': bool(args.m is not None),
        'verify_reconstruction': bool(args.verify_reconstruction),
        'verify_tol': float(args.verify_tol),
        'target_compression_ratio': float(target_compression_ratio),
        'I3_use_pcc': bool(use_pcc),
        'I3_calib_data_name': calib_data,
        'I3_calib_num_samples': int(calib_samples),
        'I3_calib_seqlen': int(calib_seqlen),
        'I3_lambda_pcc': float(lambda_pcc),
        'I3_tokens_batch': int(tokens_batch),
        'I3_expert_chunk_size': int(expert_chunk_size),
        'I3_basis_activation': 'post',
    }
    (save_root / 'train_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    _write_csv(save_root / 'final_aced_trials.csv', final_rows)
    if bool(args.save_model):
        save_format = str(args.save_format)
        if save_format in ('dense', 'both'):
            out_model = save_root / 'model'
            _merge_wab_to_model(base_model=base_model_path, wab_dir=out_wab, save_dir=out_model, modules=MODULES, layers=layers, dtype_name=str(args.dtype), trust_remote_code=bool(args.trust_remote_code), activation=activation, verify_reconstruction=bool(args.verify_reconstruction), verify_tol=float(args.verify_tol))
            if 'deepseek' in str(base_model_path).lower():
                src_modeling = base_model_path / 'modeling_deepseek.py'
                dst_modeling = out_model / 'modeling_deepseek.py'
                if src_modeling.exists():
                    shutil.copy2(str(src_modeling), str(dst_modeling))
                    print(f'[Info] copied {src_modeling} -> {dst_modeling}')
                else:
                    print(f'[Warn] DeepSeek modeling file not found: {src_modeling}')
        if save_format in ('wab_native', 'both'):
            out_model_wab = save_root / 'model_wab'
            _save_wab_native_model(
                base_model=base_model_path, wab_dir=out_wab, save_dir=out_model_wab, modules=MODULES, layers=layers,
                dtype_name=str(args.dtype), trust_remote_code=bool(args.trust_remote_code), activation=activation,
                mode_star=mode_star, num_b=int(selected_m),
                verify_reconstruction=bool(args.verify_reconstruction), verify_tol=float(args.verify_tol),
            )
    print(f'[Done][train] outputs at: {save_root}')


if __name__ == '__main__':
    main()
