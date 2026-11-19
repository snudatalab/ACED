import argparse
import gc
import json
import math
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_ACED_ROOT = _THIS_DIR.parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from wab_core import (
    MODULES,
    DEFAULT_BASE_MODELS,
    _candidate_m_values,
    _groups_by_index,
    _mode_by_shape,
    _select_rank_from_compression_ratio,
    _select_search_layers_arithmetic,
    _train_and_reconstruct_grouped,
)
from weight_io import _discover_layers, _is_dense_layer, _load_expert_weight_stack, _load_index
from search import _can_reach_compression_ratio, _candidate_m_values_from_min, _min_feasible_m


def parse_args():
    ap = argparse.ArgumentParser(description='ACED search: I1 mode selection + I2 basis-count selection.')
    ap.add_argument('--model', type=str, default='deepseek', choices=['mixtral', 'qwen3', 'deepseek'])
    ap.add_argument('--base_model', type=str, default=None)
    ap.add_argument('--index_path', type=str, default=None)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--activation', type=str, default='silu', choices=['silu', 'tanh'])
    ap.add_argument('--out_root', type=str, default=str(_ACED_ROOT / 'results'))
    ap.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    ap.add_argument('--target_compression_ratio', type=float, default=0.5)
    ap.add_argument('--final_epochs', type=int, required=True)
    ap.add_argument('--lr', type=float, default=0.07)
    ap.add_argument('--log_interval', type=int, default=100)
    ap.add_argument('--init_std', action='store_true')
    ap.add_argument('--I2_num_candidates', type=int, default=8)
    ap.add_argument('--I2_search_ratio', type=float, default=0.2)
    ap.add_argument('--I2_num_search_layers', type=int, default=5)
    ap.add_argument('--I3_use_pcc', action='store_true')
    ap.add_argument('--I3_calib_data_name', type=str, default='pile', choices=['c4', 'wiki2', 'pile'])
    ap.add_argument('--I3_calib_num_samples', type=int, default=128)
    ap.add_argument('--I3_calib_seqlen', type=int, default=1024)
    ap.add_argument('--I3_calib_seed', type=int, default=0)
    ap.add_argument('--I3_lambda_pcc', type=float, default=0.0)
    ap.add_argument('--I3_tokens_batch', type=int, default=128)
    ap.add_argument('--I3_expert_chunk_size', type=int, default=8)
    ap.add_argument('--aced_result_path', type=str, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    if int(args.I2_num_candidates) <= 0 or (int(args.I2_num_candidates) & (int(args.I2_num_candidates) - 1)) != 0:
        raise ValueError(f'--I2_num_candidates must be a positive power of two, got {args.I2_num_candidates}')
    if not (0.0 <= float(args.target_compression_ratio) < 1.0):
        raise ValueError(f'--target_compression_ratio must be in [0, 1), got {args.target_compression_ratio}')
    if args.base_model is not None:
        base_model_path = Path(args.base_model).resolve()
    elif args.model is not None:
        default_path = DEFAULT_BASE_MODELS[args.model]
        if not default_path:
            raise ValueError(
                f"--model {args.model} has no local checkpoint path configured "
                f"(DEFAULT_BASE_MODELS['{args.model}'] in src/wab_core.py is unset). "
                f"Either pass --base_model <path> directly, or fill in DEFAULT_BASE_MODELS for this machine."
            )
        base_model_path = Path(default_path).resolve()
    else:
        raise ValueError('Provide --base_model or --model')
    index_path = Path(args.index_path).resolve() if args.index_path else base_model_path / 'model.safetensors.index.json'
    if not index_path.exists():
        index_path = base_model_path / 'pytorch_model.bin.index.json'
    if args.aced_result_path:
        result_path = Path(args.aced_result_path).resolve()
        save_root = result_path.parent
    else:
        save_root = Path(args.out_root).resolve() / f"{(args.model or base_model_path.name)}_I1I2_{datetime.now().strftime('%Y%m%d_%H%M')}"
        result_path = save_root / 'aced_result.json'
    save_root.mkdir(parents=True, exist_ok=True)

    index_data = _load_index(index_path)
    available = _discover_layers(index_data)
    layers = sorted(set.intersection(*[available[u] for u in MODULES]))
    if layers and _is_dense_layer(index_data, layers[0]):
        layers = layers[1:]
    if not layers:
        raise ValueError('No common MoE layers found.')
    search_layers = _select_search_layers_arithmetic(layers, int(args.I2_num_search_layers))
    low_epochs = max(1, int(math.ceil(float(args.I2_search_ratio) * int(args.final_epochs))))
    print(f'[Setup] layers={len(layers)} search_layers={search_layers} low_epochs={low_epochs}')

    expert_counts = {}
    mode_star = {}
    module_shapes = {}
    for u in MODULES:
        tw, _eids = _load_expert_weight_stack(index_data, base_model_path, search_layers[0], u)
        expert_counts[u] = int(tw.size(0))
        mode_star[u] = _mode_by_shape(tw[0])
        module_shapes[u] = (int(tw.size(1)), int(tw.size(2)))
        print(f'[I1] module={u} mode={mode_star[u]} num_experts={expert_counts[u]}')
        del tw
        gc.collect()

    model_num_experts = int(expert_counts[MODULES[0]])
    inconsistent_expert_counts = {u: n for u, n in expert_counts.items() if int(n) != model_num_experts}
    if inconsistent_expert_counts:
        raise ValueError(f'I2 expects a single model-level expert count, got {expert_counts}')

    I2_m_search_start_time = time.time()
    I2_m_search_started_at = datetime.now().isoformat(timespec='seconds')

    candidate_m_default = _candidate_m_values(model_num_experts, int(args.I2_num_candidates))
    infeasible_default = [
        int(m)
        for m in candidate_m_default
        if not all(
            _can_reach_compression_ratio(
                model_num_experts,
                int(module_shapes[u][0]),
                int(module_shapes[u][1]),
                int(m),
                float(args.target_compression_ratio),
                mode_star[u],
            )
            for u in MODULES
        )
    ]
    if infeasible_default:
        min_feasible_m = _min_feasible_m(
            model_num_experts,
            module_shapes,
            mode_star,
            float(args.target_compression_ratio),
        )
        candidate_m = _candidate_m_values_from_min(
            model_num_experts,
            int(args.I2_num_candidates),
            int(min_feasible_m),
        )
        print(
            f'[I2][Candidates] default_m={candidate_m_default} infeasible_m={infeasible_default} '
            f'min_feasible_m={min_feasible_m} candidate_m={candidate_m}'
        )
    else:
        min_feasible_m = None
        candidate_m = candidate_m_default
        print(f'[I2][Candidates] candidate_m={candidate_m}')
    score_sums = {int(m): 0.0 for m in candidate_m}
    rank_sums = {int(m): 0.0 for m in candidate_m}
    pair_counts = {int(m): 0 for m in candidate_m}
    for u in MODULES:
        mode = mode_star[u]
        for li in search_layers:
            tw_cpu, _eids = _load_expert_weight_stack(index_data, base_model_path, li, u)
            tw = tw_cpu.to(device=device, dtype=torch.float32)
            n, r, d = int(tw.size(0)), int(tw.size(1)), int(tw.size(2))
            for m in candidate_m:
                rank = _select_rank_from_compression_ratio(n, r, d, int(m), float(args.target_compression_ratio), mode)
                groups = _groups_by_index(n, int(m))
                mse, _w_hat = _train_and_reconstruct_grouped(
                    tw, groups, int(m), int(rank), mode, args.activation, low_epochs, float(args.lr), bool(args.init_std)
                )
                score_sums[int(m)] += float(mse)
                rank_sums[int(m)] += float(rank)
                pair_counts[int(m)] += 1
                del _w_hat
                gc.collect()
            print(f'[I2][Partial] module={u} layer={li} evaluated_m={len(candidate_m)}')
            del tw, tw_cpu
            gc.collect()

    best = None
    summary_rows = []
    for m in candidate_m:
        count = max(int(pair_counts[int(m)]), 1)
        d_low = float(score_sums[int(m)] / count)
        avg_rank = float(rank_sums[int(m)] / count)
        row = {
            'candidate_m': int(m),
            'D_low': d_low,
            'score_real_mse': d_low,
            'avg_rank_ab': avg_rank,
            'num_layer_module_pairs': int(pair_counts[int(m)]),
            'low_epochs': int(low_epochs),
        }
        summary_rows.append(row)
        print(f'[I2] m={m} D_low={d_low:.10f} avg_rank={avg_rank:.2f} pairs={pair_counts[int(m)]}')
        if best is None or d_low < best['D_low']:
            best = row
    selected_m = int(best['candidate_m'])
    I2_m_search_end_time = time.time()
    I2_m_search_finished_at = datetime.now().isoformat(timespec='seconds')
    I2_m_search_elapsed_seconds = float(I2_m_search_end_time - I2_m_search_start_time)
    I2_m_search_elapsed_hms = str(timedelta(seconds=int(round(I2_m_search_elapsed_seconds))))
    print(f'[I2][Select] m_star={selected_m} D_low={best["D_low"]:.10f}')

    aced_result = {
        'base_model': str(base_model_path),
        'index_path': str(index_path),
        'save_root': str(save_root),
        'layers': [int(x) for x in layers],
        'search_layers': [int(x) for x in search_layers],
        'activation': str(args.activation),
        'dtype': str(args.dtype),
        'target_compression_ratio': float(args.target_compression_ratio),
        'final_epochs': int(args.final_epochs),
        'lr': float(args.lr),
        'log_interval': int(args.log_interval),
        'init_std': bool(args.init_std),
        'I1_mode_star': {k: str(v) for k, v in mode_star.items()},
        'I2_num_candidates': int(args.I2_num_candidates),
        'I2_search_ratio': float(args.I2_search_ratio),
        'I2_num_search_layers': int(args.I2_num_search_layers),
        'I2_low_epochs': int(low_epochs),
        'I2_m_search_started_at': I2_m_search_started_at,
        'I2_m_search_finished_at': I2_m_search_finished_at,
        'I2_m_search_elapsed_seconds': I2_m_search_elapsed_seconds,
        'I2_m_search_elapsed_hms': I2_m_search_elapsed_hms,
        'I2_model_num_experts': int(model_num_experts),
        'I2_candidate_m_default': [int(x) for x in candidate_m_default],
        'I2_candidate_m_infeasible_default': [int(x) for x in infeasible_default],
        'I2_min_feasible_m': int(min_feasible_m) if min_feasible_m is not None else None,
        'I2_candidate_m': [int(x) for x in candidate_m],
        'I2_selected_m': int(selected_m),
        'I2_m_selection_summary': summary_rows,
        'I3_use_pcc': bool(args.I3_use_pcc),
        'I3_calib_data_name': str(args.I3_calib_data_name),
        'I3_calib_num_samples': int(args.I3_calib_num_samples),
        'I3_calib_seqlen': int(args.I3_calib_seqlen),
        'I3_calib_seed': int(args.I3_calib_seed),
        'I3_lambda_pcc': float(args.I3_lambda_pcc),
        'I3_tokens_batch': int(args.I3_tokens_batch),
        'I3_expert_chunk_size': int(args.I3_expert_chunk_size),
        'I3_basis_activation': 'post',
    }
    result_path.write_text(json.dumps(aced_result, indent=2), encoding='utf-8')
    print(f'[Done][aced] result saved: {result_path}')


if __name__ == '__main__':
    main()
