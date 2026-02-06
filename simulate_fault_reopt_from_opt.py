
# simulate_fault_reopt.py
"""
Simulate a single random (or specified) node failure during an irrigation cycle, then re-optimize the remaining
(irrigable) laterals with group sizes 2-4.

Problem setting (as you described)
- Before failure: use fixed 4-lateral groups (design regime).
- After failure: allow mixed group sizes {2,3,4}.
- Failure at node Jk makes that node and its downstream subtree (w.r.t. flow direction from root) unavailable.
  If any of those nodes/laterals have NOT been irrigated before the failure, they are removed from the remaining
  irrigation tasks ("service loss" due to topology cut).
- Remaining not-yet-irrigated laterals are re-optimized to minimize:
    J = w_mean * mu + w_std * std
  where mu/std are computed over group-level mean surplus head:
    surplus = pressure_head - Hmin
    m_g = mean surplus of group g
    mu = mean(m_g), std = std(m_g) across groups

Outputs (per scenario)
- schedule.json (full details)
- group_metrics.csv
- lateral_metrics.csv
- scenario_summary.csv (one row)

Notes
- Requires your tree_evaluator.py (same folder or on PYTHONPATH) and filled GBT_COEFS.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import tree_evaluator as te
except ImportError as e:
    raise ImportError(
        "Cannot import tree_evaluator.py. Put this script in the same folder as tree_evaluator.py, "
        "or add that folder to PYTHONPATH."
    ) from e


# ----------------------------
# Basic stats helpers
# ----------------------------
def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _var_population(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    mu = _mean(xs)
    return sum((x - mu) ** 2 for x in xs) / len(xs)


def _std_population(xs: Sequence[float]) -> float:
    v = _var_population(xs)
    return math.sqrt(v) if v == v else float("nan")


# ----------------------------
# Group eval / schedule eval
# ----------------------------
@dataclass(frozen=True)
class GroupEval:
    ok: bool
    laterals: Tuple[str, ...]
    nodes: Tuple[str, ...]
    pressures: Tuple[float, ...]   # node pressure head, aligned to laterals
    surplus: Tuple[float, ...]     # pressure - Hmin, aligned to laterals
    min_pressure: float
    min_margin: float              # min(pressure - Hmin) over opened laterals

    @property
    def mean_surplus(self) -> float:
        return _mean(self.surplus)

    @property
    def std_surplus(self) -> float:
        return _std_population(self.surplus)


@dataclass
class ScheduleEval:
    ok: bool
    groups: List[GroupEval]

    # objective uses group-level means m_g
    mu_group_mean: float
    std_group_mean: float
    objective: float


def objective(mu: float, std: float, w_mean: float, w_std: float) -> float:
    return w_mean * mu + w_std * std


class CachedGroupEvaluator:
    def __init__(
        self,
        evaluator: te.TreeHydraulicEvaluator,
        lateral_to_node: Dict[str, str],
        Hmin: float,
        q_lateral: float,
        cache_max: int = 500_000,
    ) -> None:
        self.evaluator = evaluator
        self.lateral_to_node = lateral_to_node
        self.Hmin = float(Hmin)
        self.q_lateral = float(q_lateral)
        self.cache_max = int(cache_max)
        self._cache: Dict[Tuple[str, ...], GroupEval] = {}
        self.calls = 0
        self.hits = 0

    def eval_group(self, laterals: Iterable[str]) -> GroupEval:
        key = tuple(sorted(laterals))
        if key in self._cache:
            self.hits += 1
            return self._cache[key]

        self.calls += 1
        res = self.evaluator.evaluate_group(key, lateral_to_node=self.lateral_to_node, q_lateral=self.q_lateral)

        nodes = tuple(self.lateral_to_node[lid] for lid in key)
        pressures: List[float] = []
        surplus: List[float] = []
        for nid in nodes:
            p = float(res.pressures[nid])
            pressures.append(p)
            surplus.append(p - self.Hmin)

        ge = GroupEval(
            ok=bool(res.ok),
            laterals=key,
            nodes=nodes,
            pressures=tuple(pressures),
            surplus=tuple(surplus),
            min_pressure=float(res.min_pressure_head),
            min_margin=float(res.min_margin),
        )

        # simple eviction to cap memory
        if len(self._cache) >= self.cache_max:
            for _ in range(2000):
                self._cache.pop(next(iter(self._cache)))
                if len(self._cache) < self.cache_max * 0.9:
                    break

        self._cache[key] = ge
        return ge


def evaluate_schedule(groups: List[List[str]], geval: CachedGroupEvaluator, w_mean: float, w_std: float) -> ScheduleEval:
    g_evals: List[GroupEval] = []
    group_means: List[float] = []

    for g in groups:
        gr = geval.eval_group(g)
        if not gr.ok:
            return ScheduleEval(False, [], float("inf"), float("inf"), float("inf"))
        g_evals.append(gr)
        group_means.append(gr.mean_surplus)

    mu = _mean(group_means)
    sd = _std_population(group_means)
    obj = objective(mu, sd, w_mean, w_std)
    return ScheduleEval(True, g_evals, mu, sd, obj)


# ----------------------------
# Construct + anneal optimizer
# ----------------------------
def build_feasible_schedule(
    laterals: List[str],
    geval: CachedGroupEvaluator,
    rng: random.Random,
    allowed_sizes: Tuple[int, ...],
    max_tries_per_group: int = 500,
) -> Optional[List[List[str]]]:
    remaining = laterals[:]
    rng.shuffle(remaining)
    groups: List[List[str]] = []

    # Prefer larger groups first (to reduce total groups), but constrained by allowed_sizes order
    size_order = tuple(sorted(allowed_sizes, reverse=True))

    while remaining:
        created = False
        for k in size_order:
            if len(remaining) < k:
                continue
            for _ in range(max_tries_per_group):
                g = rng.sample(remaining, k)
                if geval.eval_group(g).ok:
                    for lid in g:
                        remaining.remove(lid)
                    groups.append(g)
                    created = True
                    break
            if created:
                break

        if not created:
            return None

    return groups


def propose_neighbor(groups: List[List[str]], rng: random.Random, allowed_sizes: Tuple[int, ...]) -> List[List[str]]:
    new_groups = [g[:] for g in groups]
    n = len(new_groups)
    if n < 2:
        return new_groups

    i, j = rng.sample(range(n), 2)
    gi, gj = new_groups[i], new_groups[j]

    op = rng.random()
    if op < 0.60:
        # swap one element
        a = rng.choice(gi)
        b = rng.choice(gj)
        gi[gi.index(a)] = b
        gj[gj.index(b)] = a
    else:
        # move one element if sizes stay within allowed_sizes
        allowed = set(allowed_sizes)
        if rng.random() < 0.5:
            src, dst = gi, gj
        else:
            src, dst = gj, gi

        if len(src) > min(allowed) and len(dst) < max(allowed):
            x = rng.choice(src)
            if (len(src) - 1) in allowed and (len(dst) + 1) in allowed:
                src.remove(x)
                dst.append(x)
        else:
            # fallback to swap
            a = rng.choice(gi)
            b = rng.choice(gj)
            gi[gi.index(a)] = b
            gj[gj.index(b)] = a

    return new_groups


def anneal(
    init_groups: List[List[str]],
    geval: CachedGroupEvaluator,
    rng: random.Random,
    allowed_sizes: Tuple[int, ...],
    w_mean: float,
    w_std: float,
    steps: int = 30000,
    T0: float = 1.0,
    Tend: float = 1e-3,
) -> ScheduleEval:
    cur_groups = [g[:] for g in init_groups]
    cur_eval = evaluate_schedule(cur_groups, geval, w_mean, w_std)
    if not cur_eval.ok:
        raise RuntimeError("Initial schedule must be feasible.")

    best = cur_eval

    for t in range(steps):
        frac = t / max(steps - 1, 1)
        T = T0 * ((Tend / T0) ** frac)

        cand_groups = propose_neighbor(cur_groups, rng, allowed_sizes)
        cand_eval = evaluate_schedule(cand_groups, geval, w_mean, w_std)
        if not cand_eval.ok:
            continue

        delta = cand_eval.objective - cur_eval.objective
        if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
            cur_groups = cand_groups
            cur_eval = cand_eval
            if cur_eval.objective < best.objective:
                best = cur_eval

    return best


# ----------------------------
# Failure / reachability
# ----------------------------
def subtree_nodes(children: Dict[str, List[str]], start: str) -> List[str]:
    """Return start + all descendants in the rooted tree."""
    out: List[str] = []
    stack = [start]
    seen = set()
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
        for v in children.get(u, []):
            stack.append(v)
    return out


# ----------------------------
# IO helpers
# ----------------------------
def write_csv(path: Path, header: List[str], rows: List[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def export_all(
    out_dir: Path,
    H0: float,
    Hmin: float,
    q_lateral: float,
    w_mean: float,
    w_std: float,
    meta: Dict[str, object],
    prefault_executed: List[GroupEval],
    postfault: Optional[ScheduleEval],
    overall: Optional[ScheduleEval],
    lateral_to_node: Dict[str, str],
    blocked_laterals: List[str],
    irrigated_before: List[str],
    remaining_after_fault: List[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # group_metrics.csv
    group_rows: List[List[object]] = []
    gid = 0
    for phase, groups in [("prefault", prefault_executed), ("postfault", (postfault.groups if postfault and postfault.ok else []))]:
        for g in groups:
            gid += 1
            group_rows.append([
                H0, phase, gid, len(g.laterals),
                g.min_pressure, g.min_margin,
                g.mean_surplus, g.std_surplus,
                " ".join(g.laterals),
            ])

    write_csv(
        out_dir / "group_metrics.csv",
        ["H0", "phase", "group_id", "group_size",
         "min_pressure_head", "min_margin",
         "group_mean_surplus", "group_std_surplus",
         "laterals"],
        group_rows,
    )

    # lateral_metrics.csv
    lateral_rows: List[List[object]] = []
    gid = 0
    for phase, groups in [("prefault", prefault_executed), ("postfault", (postfault.groups if postfault and postfault.ok else []))]:
        for g in groups:
            gid += 1
            for lid, nid, p, s in zip(g.laterals, g.nodes, g.pressures, g.surplus):
                lateral_rows.append([H0, phase, gid, lid, nid, p, s])
    write_csv(
        out_dir / "lateral_metrics.csv",
        ["H0", "phase", "group_id", "lateral_id", "node_id", "pressure_head", "surplus_head"],
        lateral_rows,
    )

    payload = {
        "H0": H0,
        "Hmin": Hmin,
        "q_lateral": q_lateral,
        "weights": {"w_mean": w_mean, "w_std": w_std},
        "meta": meta,
        "fault_effect": {
            "blocked_laterals": blocked_laterals,
            "n_blocked_laterals": len(blocked_laterals),
            "irrigated_before_fault": irrigated_before,
            "n_irrigated_before_fault": len(irrigated_before),
            "remaining_after_fault": remaining_after_fault,
            "n_remaining_after_fault": len(remaining_after_fault),
        },
        "prefault_executed": [
            {
                "laterals": list(g.laterals),
                "nodes": list(g.nodes),
                "pressures": list(g.pressures),
                "surplus": list(g.surplus),
                "min_margin": g.min_margin,
                "min_pressure": g.min_pressure,
                "group_mean_surplus": g.mean_surplus,
                "group_std_surplus": g.std_surplus,
            } for g in prefault_executed
        ],
        "postfault_reopt": None if not postfault else {
            "ok": postfault.ok,
            "mu_group_mean": postfault.mu_group_mean,
            "std_group_mean": postfault.std_group_mean,
            "objective": postfault.objective,
            "groups": [
                {
                    "laterals": list(g.laterals),
                    "nodes": list(g.nodes),
                    "pressures": list(g.pressures),
                    "surplus": list(g.surplus),
                    "min_margin": g.min_margin,
                    "min_pressure": g.min_pressure,
                    "group_mean_surplus": g.mean_surplus,
                    "group_std_surplus": g.std_surplus,
                } for g in (postfault.groups if postfault.ok else [])
            ],
        },
        "overall_executed": None if not overall else {
            "ok": overall.ok,
            "mu_group_mean": overall.mu_group_mean,
            "std_group_mean": overall.std_group_mean,
            "objective": overall.objective,
            "n_groups_executed": len(overall.groups) if overall.ok else None,
        },
    }
    (out_dir / "schedule.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # scenario_summary.csv
    row = [
        H0,
        meta.get("fault_node"),
        meta.get("fault_after_groups"),
        len(irrigated_before),
        len(blocked_laterals),
        len(remaining_after_fault),
        len(prefault_executed),
        (len(postfault.groups) if postfault and postfault.ok else ""),
        (overall.mu_group_mean if overall and overall.ok else ""),
        (overall.std_group_mean if overall and overall.ok else ""),
        (overall.objective if overall and overall.ok else ""),
        (postfault.mu_group_mean if postfault and postfault.ok else ""),
        (postfault.std_group_mean if postfault and postfault.ok else ""),
        (postfault.objective if postfault and postfault.ok else ""),
        meta.get("cache_calls"),
        meta.get("cache_hits"),
        meta.get("elapsed_sec"),
        ("OK" if (postfault and postfault.ok) else "FAILED"),
    ]
    write_csv(
        out_dir / "scenario_summary.csv",
        ["H0", "fault_node", "fault_after_groups",
         "n_irrigated_before_fault", "n_blocked_laterals", "n_remaining_after_fault",
         "n_prefault_groups_executed", "n_postfault_groups",
         "overall_mu_group_mean", "overall_std_group_mean", "overall_objective",
         "post_mu_group_mean", "post_std_group_mean", "post_objective",
         "eval_calls", "cache_hits", "elapsed_sec", "status"],
        [row],
    )



def load_prefault_groups_from_json(path: str) -> List[List[str]]:
    """Load groups from a schedule.json produced by rotation_optimize_groupstats / simulate scripts."""
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    groups: List[List[str]] = []

    # Try common shapes
    if isinstance(obj, dict) and "groups" in obj and isinstance(obj["groups"], list):
        # rotation_optimize_groupstats style: groups: [{group_id, laterals,...}, ...]
        for g in obj["groups"]:
            if isinstance(g, dict) and "laterals" in g:
                groups.append(list(g["laterals"]))
            elif isinstance(g, list):
                groups.append(list(g))
    elif isinstance(obj, dict) and "prefault_executed" in obj:
        # simulate_fault_reopt style: prefault_executed: [{laterals: [...]}, ...]
        for g in obj["prefault_executed"]:
            groups.append(list(g["laterals"]))
    else:
        raise ValueError(f"Unrecognized schedule.json format: {p}")

    if not groups:
        raise ValueError(f"No groups found in {p}")

    return groups

# ----------------------------
# CLI + main
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", required=True)
    p.add_argument("--pipes", required=True)
    p.add_argument("--root", default="J0")
    p.add_argument("--H0", type=float, required=True)
    p.add_argument("--Hmin", type=float, default=11.59)
    p.add_argument("--q_lateral", type=float, default=0.012)
    p.add_argument("--out", required=True, help="Output directory for this scenario")

    # prefault schedule control
    p.add_argument("--prefault_seed", type=int, default=123)
    p.add_argument("--prefault_max_tries_per_group", type=int, default=800)
    p.add_argument("--prefault_sa_steps", type=int, default=15000)
    p.add_argument("--prefault_schedule_json", default="",
                   help="Optional: load a pre-fault schedule from a schedule.json (e.g., from runs_opt_groupstats). "
                        "If empty, a feasible 4-lateral schedule will be constructed automatically.")

    # fault control
    p.add_argument("--fault_node", default="", help="e.g., J12. If empty, random field node will be sampled.")
    p.add_argument("--fault_after_groups", type=int, default=-1,
                   help="How many 4-lateral groups are executed BEFORE fault. If -1, random index is sampled.")

    # postfault optimizer control
    p.add_argument("--seed", type=int, default=456)
    p.add_argument("--n_init", type=int, default=800)
    p.add_argument("--sa_steps", type=int, default=30000)
    p.add_argument("--max_tries_per_group", type=int, default=600)
    p.add_argument("--w_mean", type=float, default=0.5)
    p.add_argument("--w_std", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    nodes = te.load_nodes_xlsx(args.nodes)
    edges = te.load_pipes_xlsx(args.pipes)

    field_nodes = [nid for nid in nodes.keys() if te.is_field_node_id(nid)]
    lateral_ids, lateral_to_node = te.build_lateral_ids_for_field_nodes(field_nodes)

    evaluator = te.TreeHydraulicEvaluator(nodes=nodes, edges=edges, root=args.root, H0=args.H0, Hmin=args.Hmin)
    geval = CachedGroupEvaluator(evaluator, lateral_to_node, args.Hmin, args.q_lateral)

    # --------
    # 1) pre-fault schedule:
    #    - If --prefault_schedule_json is provided: use it (group sizes typically 2-4).
    #    - Else: construct + light anneal to get a feasible fixed-4 schedule (design regime).
    #    - If --prefault_schedule_json is provided: use it (assumed fixed-4).
    #    - Else: construct + light anneal to get a feasible fixed-4 schedule.
    # --------
    if args.prefault_schedule_json:
        pre_groups_best = load_prefault_groups_from_json(args.prefault_schedule_json)

        # sanity: allow typical sizes 2-4 (matching your offline optimizer)
        for g in pre_groups_best:
            if len(g) not in (2, 3, 4):
                raise ValueError(
                    "Pre-fault schedule must have group sizes in {2,3,4} when loading from schedule.json. "
                    f"Found group size={len(g)}."
                )

        # feasibility check
        for g in pre_groups_best:
            if not geval.eval_group(g).ok:
                raise RuntimeError("Provided pre-fault schedule contains infeasible group(s) at this H0.")
    else:
        rng_pref = random.Random(args.prefault_seed)
        pre_groups = build_feasible_schedule(
            lateral_ids, geval, rng_pref,
            allowed_sizes=(4,),
            max_tries_per_group=args.prefault_max_tries_per_group,
        )
        if pre_groups is None:
            raise RuntimeError("Cannot construct a feasible pre-fault schedule with fixed 4 laterals per group at this H0.")

        # a light anneal for pre-fault schedule (still fixed size=4)
        pre_eval0 = anneal(
            pre_groups, geval, random.Random(args.prefault_seed + 999),
            allowed_sizes=(4,),
            w_mean=args.w_mean, w_std=args.w_std,
            steps=args.prefault_sa_steps,
        )
        pre_groups_best = [list(g.laterals) for g in pre_eval0.groups]  # laterals within group are sorted

    # Choose fault node
    rng = random.Random(args.seed)
    fault_node = args.fault_node.strip().upper() if args.fault_node else rng.choice(field_nodes)

    # Choose fault time (number of groups executed before fault)
    if args.fault_after_groups >= 0:
        fault_after = args.fault_after_groups
    else:
        # random from [0, n_groups-1]
        fault_after = rng.randrange(0, len(pre_groups_best))

    fault_after = max(0, min(fault_after, len(pre_groups_best)))  # clamp

    executed_groups = pre_groups_best[:fault_after]
    irrigated_before = sorted({lid for g in executed_groups for lid in g})

    # Determine blocked subtree nodes (fault node + descendants)
    blocked_nodes = set(subtree_nodes(evaluator.children, fault_node))
    blocked_laterals = sorted([lid for lid, nid in lateral_to_node.items() if nid in blocked_nodes and lid not in irrigated_before])

    # Remaining laterals after fault (need reopt)
    remaining_after_fault = sorted([lid for lid in lateral_ids if lid not in irrigated_before and lid not in blocked_laterals])

    # Evaluate executed pre-fault groups (for detailed export)
    prefault_executed_evals: List[GroupEval] = []
    for g in executed_groups:
        gr = geval.eval_group(g)
        if not gr.ok:
            # should not happen because schedule is feasible
            raise RuntimeError("Unexpected: a pre-fault executed group became infeasible.")
        prefault_executed_evals.append(gr)

    # --------
    # 2) post-fault re-optimization on remaining laterals (sizes 2-4)
    # --------
    postfault_eval: Optional[ScheduleEval] = None
    overall_eval: Optional[ScheduleEval] = None

    if remaining_after_fault:
        best_eval: Optional[ScheduleEval] = None
        best_groups: Optional[List[List[str]]] = None

        for _ in range(args.n_init):
            g0 = build_feasible_schedule(
                remaining_after_fault, geval, random.Random(rng.randrange(1 << 30)),
                allowed_sizes=(2, 3, 4),
                max_tries_per_group=args.max_tries_per_group,
            )
            if g0 is None:
                continue
            se = evaluate_schedule(g0, geval, args.w_mean, args.w_std)
            if se.ok and (best_eval is None or se.objective < best_eval.objective):
                best_eval = se
                best_groups = g0

        if best_eval is None or best_groups is None:
            postfault_eval = ScheduleEval(False, [], float("inf"), float("inf"), float("inf"))
        else:
            postfault_eval = anneal(
                best_groups, geval, random.Random(rng.randrange(1 << 30)),
                allowed_sizes=(2, 3, 4),
                w_mean=args.w_mean, w_std=args.w_std,
                steps=args.sa_steps,
            )

        # overall executed = pre executed + postfault
        if postfault_eval.ok:
            overall_groups = executed_groups + [list(g.laterals) for g in postfault_eval.groups]
            overall_eval = evaluate_schedule(overall_groups, geval, args.w_mean, args.w_std)
        else:
            overall_eval = None
    else:
        # no remaining tasks (either fault happens late or subtree removes all rest)
        overall_groups = executed_groups
        overall_eval = evaluate_schedule(overall_groups, geval, args.w_mean, args.w_std) if overall_groups else None

    # Prefault group size stats (for reporting)
    pref_sizes = [len(g) for g in pre_groups_best]
    pref_hist = {2: 0, 3: 0, 4: 0}
    for sz in pref_sizes:
        if sz in pref_hist:
            pref_hist[sz] += 1

    elapsed = time.time() - t0

    meta = {
        "root": args.root,
        "fault_node": fault_node,
        "fault_after_groups": fault_after,
        "n_total_groups_prefault_plan": len(pre_groups_best),
        "prefault_schedule_source": ("loaded" if args.prefault_schedule_json else "constructed_fixed4"),
        "prefault_group_size_hist": pref_hist,

        "cache_calls": geval.calls,
        "cache_hits": geval.hits,
        "elapsed_sec": elapsed,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_field_nodes": len(field_nodes),
        "n_total_laterals": len(lateral_ids),
        "n_init_postfault": args.n_init,
        "sa_steps_postfault": args.sa_steps,
        "max_tries_per_group_postfault": args.max_tries_per_group,
    }

    export_all(
        out_dir=out_dir,
        H0=args.H0,
        Hmin=args.Hmin,
        q_lateral=args.q_lateral,
        w_mean=args.w_mean,
        w_std=args.w_std,
        meta=meta,
        prefault_executed=prefault_executed_evals,
        postfault=postfault_eval,
        overall=overall_eval,
        lateral_to_node=lateral_to_node,
        blocked_laterals=blocked_laterals,
        irrigated_before=irrigated_before,
        remaining_after_fault=remaining_after_fault,
    )


if __name__ == "__main__":
    main()
