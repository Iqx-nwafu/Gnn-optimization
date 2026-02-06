# rotation_optimize_groupstats.py
"""
Generate near-optimal rotation schedules (轮灌制度近最优解) for a tree irrigation network
under varying source head H0, using feasibility-constrained metaheuristics.

Key definitions (as agreed with user):
- Surplus head for a lateral's node under a given group: s = pressure_head - Hmin.
- Statistics are computed on a *per-group* basis:
    group_surplus_mean[g] = mean_{laterals in group g}(s)
  Then across all groups in a full cycle:
    mu = mean(group_surplus_mean)
    sigma = std(group_surplus_mean)
- Objective (equal weights by default):
    J = w_mean * mu + w_std * sigma

Outputs per H0:
- schedule.json (full detail)
- group_metrics.csv (per-group detail)
- lateral_metrics.csv (per-lateral detail)
- summary.csv (across all H0)

Dependencies:
- Your provided tree_evaluator.py (same folder or on PYTHONPATH).
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


@dataclass(frozen=True)
class GroupEval:
    ok: bool
    min_margin: float
    min_pressure: float
    laterals: Tuple[str, ...]
    pressures: Tuple[float, ...]  # per-lateral node pressure head
    surplus: Tuple[float, ...]    # per-lateral (pressure - Hmin)

    @property
    def mean_surplus(self) -> float:
        return sum(self.surplus) / len(self.surplus)

    @property
    def var_surplus(self) -> float:
        mu = self.mean_surplus
        return sum((x - mu) ** 2 for x in self.surplus) / len(self.surplus)


@dataclass
class ScheduleEval:
    ok: bool
    groups: List[GroupEval]
    # group-level statistics
    group_means: List[float]
    mu_group_mean: float
    std_group_mean: float
    objective: float


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _variance(values: Sequence[float]) -> float:
    mu = _mean(values)
    return sum((x - mu) ** 2 for x in values) / len(values)


def _objective(mu: float, std: float, w_mean: float, w_std: float) -> float:
    return w_mean * mu + w_std * std


class CachedGroupEvaluator:
    """Memoized wrapper around TreeHydraulicEvaluator.evaluate_group."""
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

        pressures: List[float] = []
        surplus: List[float] = []
        for lat in key:
            nid = self.lateral_to_node[lat]
            p = float(res.pressures[nid])
            pressures.append(p)
            surplus.append(p - self.Hmin)

        ge = GroupEval(
            ok=bool(res.ok),
            min_margin=float(res.min_margin),
            min_pressure=float(res.min_pressure_head),
            laterals=key,
            pressures=tuple(pressures),
            surplus=tuple(surplus),
        )

        # simple eviction
        if len(self._cache) >= self.cache_max:
            for _ in range(2000):
                self._cache.pop(next(iter(self._cache)))
                if len(self._cache) < self.cache_max * 0.9:
                    break

        self._cache[key] = ge
        return ge


def evaluate_schedule(
    groups: List[List[str]],
    geval: CachedGroupEvaluator,
    w_mean: float,
    w_std: float,
) -> ScheduleEval:
    g_evals: List[GroupEval] = []
    group_means: List[float] = []
    for g in groups:
        gr = geval.eval_group(g)
        if not gr.ok:
            return ScheduleEval(False, [], [], float("inf"), float("inf"), float("inf"))
        g_evals.append(gr)
        group_means.append(gr.mean_surplus)

    mu = _mean(group_means)
    var = _variance(group_means)
    std = math.sqrt(max(var, 0.0))
    obj = _objective(mu, std, w_mean, w_std)
    return ScheduleEval(True, g_evals, group_means, mu, std, obj)


def build_feasible_schedule(
    all_laterals: List[str],
    geval: CachedGroupEvaluator,
    rng: random.Random,
    max_tries_per_group: int = 600,
    size_order: Tuple[int, int, int] = (4, 3, 2),
) -> Optional[List[List[str]]]:
    remaining = all_laterals[:]
    rng.shuffle(remaining)
    groups: List[List[str]] = []

    while remaining:
        created = False
        for k in size_order:
            if len(remaining) < k:
                continue
            for _ in range(max_tries_per_group):
                g = rng.sample(remaining, k)
                if geval.eval_group(g).ok:
                    for lat in g:
                        remaining.remove(lat)
                    groups.append(g)
                    created = True
                    break
            if created:
                break

        if not created:
            return None

    return groups


def propose_neighbor(groups: List[List[str]], rng: random.Random) -> List[List[str]]:
    """Neighbor via swap or move while preserving group sizes in {2,3,4}."""
    n = len(groups)
    i, j = rng.sample(range(n), 2)
    gi = groups[i][:]
    gj = groups[j][:]

    op = rng.random()
    if op < 0.55:
        # swap
        a = rng.choice(gi)
        b = rng.choice(gj)
        gi[gi.index(a)] = b
        gj[gj.index(b)] = a
    else:
        # move one lateral if sizes remain within {2,3,4}
        if len(gi) > 2 and len(gj) < 4 and rng.random() < 0.5:
            a = rng.choice(gi)
            gi.remove(a)
            gj.append(a)
        elif len(gj) > 2 and len(gi) < 4:
            b = rng.choice(gj)
            gj.remove(b)
            gi.append(b)
        else:
            a = rng.choice(gi)
            b = rng.choice(gj)
            gi[gi.index(a)] = b
            gj[gj.index(b)] = a

    new_groups = [g[:] for g in groups]
    new_groups[i] = gi
    new_groups[j] = gj
    return new_groups


def anneal(
    init_groups: List[List[str]],
    geval: CachedGroupEvaluator,
    rng: random.Random,
    w_mean: float,
    w_std: float,
    steps: int = 25000,
    T0: float = 1.0,
    Tend: float = 1e-3,
) -> ScheduleEval:
    cur_groups = [g[:] for g in init_groups]
    cur_eval = evaluate_schedule(cur_groups, geval, w_mean, w_std)
    if not cur_eval.ok:
        raise RuntimeError("Initial schedule must be feasible")

    best_eval = cur_eval

    for t in range(steps):
        frac = t / max(steps - 1, 1)
        T = T0 * ((Tend / T0) ** frac)

        new_groups = propose_neighbor(cur_groups, rng)
        new_eval = evaluate_schedule(new_groups, geval, w_mean, w_std)
        if not new_eval.ok:
            continue

        delta = new_eval.objective - cur_eval.objective
        if delta <= 0 or rng.random() < math.exp(-delta / max(T, 1e-12)):
            cur_groups = new_groups
            cur_eval = new_eval
            if cur_eval.objective < best_eval.objective:
                best_eval = cur_eval

    return best_eval


def write_csv(path: Path, header: List[str], rows: List[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def export_solution(
    out_dir: Path,
    H0: float,
    schedule_eval: ScheduleEval,
    lateral_to_node: Dict[str, str],
    Hmin: float,
    w_mean: float,
    w_std: float,
    meta: Dict[str, object],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    group_rows: List[List[object]] = []
    lateral_rows: List[List[object]] = []
    for gi, g in enumerate(schedule_eval.groups, start=1):
        group_rows.append([
            H0, gi, len(g.laterals),
            g.min_pressure, g.min_margin,
            g.mean_surplus, math.sqrt(max(g.var_surplus, 0.0)),
            " ".join(g.laterals),
        ])
        for lat, p, s in zip(g.laterals, g.pressures, g.surplus):
            lateral_rows.append([H0, gi, lat, lateral_to_node[lat], p, s])

    write_csv(
        out_dir / "group_metrics.csv",
        ["H0", "group_id", "group_size", "min_pressure_head", "min_margin",
         "group_mean_surplus", "group_std_surplus", "laterals"],
        group_rows,
    )
    write_csv(
        out_dir / "lateral_metrics.csv",
        ["H0", "group_id", "lateral_id", "node_id", "pressure_head", "surplus_head"],
        lateral_rows,
    )

    payload = {
        "H0": H0,
        "Hmin": Hmin,
        "objective": {
            "definition": "J = w_mean * mu + w_std * sigma, computed on per-group mean surplus",
            "weights": {"w_mean": w_mean, "w_std": w_std},
            "mu_group_mean": schedule_eval.mu_group_mean,
            "std_group_mean": schedule_eval.std_group_mean,
            "J": schedule_eval.objective,
            "group_means": schedule_eval.group_means,
        },
        "groups": [
            {
                "group_id": gi,
                "laterals": list(g.laterals),
                "nodes": [lateral_to_node[lat] for lat in g.laterals],
                "pressures": list(g.pressures),
                "surplus": list(g.surplus),
                "min_pressure": g.min_pressure,
                "min_margin": g.min_margin,
                "group_mean_surplus": g.mean_surplus,
                "group_std_surplus": math.sqrt(max(g.var_surplus, 0.0)),
            }
            for gi, g in enumerate(schedule_eval.groups, start=1)
        ],
        "meta": meta,
    }
    (out_dir / "schedule.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", required=True, help="Path to Nodes.xlsx")
    p.add_argument("--pipes", required=True, help="Path to Pipes.xlsx")
    p.add_argument("--root", default="J0")
    p.add_argument("--Hmin", type=float, default=11.59)
    p.add_argument("--q_lateral", type=float, default=0.012)
    p.add_argument("--H0_list", nargs="+", type=float, default=[15, 16, 17, 18, 19, 20, 21, 22, 23])
    p.add_argument("--out", required=True, help="Output root folder")

    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n_init", type=int, default=1500, help="Number of random feasible initializations")
    p.add_argument("--sa_steps", type=int, default=40000, help="Simulated annealing steps")
    p.add_argument("--max_tries_per_group", type=int, default=600)
    p.add_argument("--w_mean", type=float, default=0.5)
    p.add_argument("--w_std", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    nodes = te.load_nodes_xlsx(args.nodes)
    edges = te.load_pipes_xlsx(args.pipes)

    # Stable ordering
    field_nodes = sorted([nid for nid in nodes.keys() if te.is_field_node_id(nid)])
    lateral_ids, lateral_to_node = te.build_lateral_ids_for_field_nodes(field_nodes)
    lateral_ids = sorted(lateral_ids)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows: List[List[object]] = []
    rng_master = random.Random(args.seed)

    for H0 in args.H0_list:
        t0 = time.time()
        evaluator = te.TreeHydraulicEvaluator(nodes=nodes, edges=edges, root=args.root, H0=H0, Hmin=args.Hmin)
        geval = CachedGroupEvaluator(evaluator, lateral_to_node, args.Hmin, args.q_lateral)

        best_eval: Optional[ScheduleEval] = None
        best_groups: Optional[List[List[str]]] = None

        for _ in range(args.n_init):
            rng = random.Random(rng_master.randrange(1 << 30))
            groups = build_feasible_schedule(
                lateral_ids, geval, rng,
                max_tries_per_group=args.max_tries_per_group,
                size_order=(4, 3, 2),
            )
            if groups is None:
                continue
            se = evaluate_schedule(groups, geval, args.w_mean, args.w_std)
            if se.ok and (best_eval is None or se.objective < best_eval.objective):
                best_eval = se
                best_groups = groups

        if best_eval is None or best_groups is None:
            summary_rows.append([H0, "FAILED", "", "", "", "", "", "", geval.calls, geval.hits, time.time() - t0])
            continue

        rng_sa = random.Random(rng_master.randrange(1 << 30))
        refined = anneal(best_groups, geval, rng_sa, args.w_mean, args.w_std, args.sa_steps)

        size_hist = {2: 0, 3: 0, 4: 0}
        for g in refined.groups:
            size_hist[len(g.laterals)] += 1

        elapsed = time.time() - t0
        meta = {
            "seed": args.seed,
            "n_init": args.n_init,
            "sa_steps": args.sa_steps,
            "max_tries_per_group": args.max_tries_per_group,
            "cache_calls": geval.calls,
            "cache_hits": geval.hits,
            "elapsed_sec": elapsed,
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "n_field_nodes": len(field_nodes),
            "n_laterals": len(lateral_ids),
        }

        out_dir = out_root / f"H0_{H0:.2f}"
        export_solution(out_dir, H0, refined, lateral_to_node, args.Hmin, args.w_mean, args.w_std, meta)

        summary_rows.append([
            H0, "OK",
            len(refined.groups),
            size_hist[4], size_hist[3], size_hist[2],
            refined.mu_group_mean, refined.std_group_mean, refined.objective,
            geval.calls, geval.hits, elapsed
        ])

    write_csv(
        out_root / "summary.csv",
        ["H0", "status", "n_groups", "n_g4", "n_g3", "n_g2",
         "mu_group_mean", "std_group_mean", "objective",
         "eval_calls", "cache_hits", "elapsed_sec"],
        summary_rows,
    )


if __name__ == "__main__":
    main()
