from typing import Dict, List, Tuple

from wab_core import _estimate_compression_ratio


def _can_reach_compression_ratio(n: int, r: int, d: int, num_b: int, target_compression_ratio: float, mode: str) -> bool:
    max_rank = min(int(r), int(d))
    min_compression = _estimate_compression_ratio(n, r, d, num_b, max_rank, mode)
    return bool(min_compression <= float(target_compression_ratio) + 1e-12)

def _min_feasible_m(
    num_experts: int,
    module_shapes: Dict[str, Tuple[int, int]],
    mode_star: Dict[str, str],
    target_compression_ratio: float,
) -> int:
    for m in range(1, int(num_experts) + 1):
        ok = True
        for u, (r, d) in module_shapes.items():
            if not _can_reach_compression_ratio(num_experts, int(r), int(d), int(m), target_compression_ratio, mode_star[u]):
                ok = False
                break
        if ok:
            return int(m)
    return int(num_experts)

def _candidate_m_values_from_min(num_experts: int, num_candidates: int, min_m: int) -> List[int]:
    e = int(num_experts)
    k = max(1, int(num_candidates))
    q = max(1, min(int(min_m), e))
    if q >= e:
        return [e]
    rem = (e - q) % k
    if rem:
        q = min(e, q + rem)
    if q >= e:
        return [e]
    step = max(1, (e - q) // k)
    vals = list(range(q, e + 1, step))
    if vals[-1] != e:
        vals.append(e)
    vals = sorted(set(vals))
    if len(vals) > k:
        vals = vals[:k]
    return vals
