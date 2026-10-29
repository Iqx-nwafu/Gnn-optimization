"""Shared utilities for revision experiments: network, evaluator, GNN-IL inference, SA/GA/greedy baselines."""
from __future__ import annotations

import itertools, json, math, random, sys, time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent  # repository root: dataset_v1/, models_gnn/, Nodes.xlsx, Pipes.xlsx and the shared modules below
sys.path.insert(0, str(REPO))

import tree_evaluator as te
from model_gnn import GroupGenerator, FeasibilityClassifier
from utils_data import load_graph

DATA = REPO / "dataset_v1"
HMIN = 11.59
QLAT = 0.012
W_MEAN = W_STD = 0.5

torch.set_num_threads(1)

NODES = te.load_nodes_xlsx(str(REPO / "Nodes.xlsx"))
EDGES = te.load_pipes_xlsx(str(REPO / "Pipes.xlsx"))
META = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
LAT_IDS: List[str] = META["lateral_ids"]
_field = [n for n in NODES if te.is_field_node_id(n)]
_, _l2n = te.build_lateral_ids_for_field_nodes(_field)
LAT2NODE = {l: _l2n[l] for l in LAT_IDS}
LIDX = {l: i for i, l in enumerate(LAT_IDS)}


class Evaluator:
    """Memoised group evaluator for a given H0 (tree solver, GB/T 50485 power-law friction)."""

    def __init__(self, H0: float, edges=None):
        self.H0 = H0; self.edges = edges or EDGES
        self.tree = te.TreeHydraulicEvaluator(NODES, self.edges, "J0", H0, HMIN)
        self.cache: Dict[Tuple[str, ...], Tuple[bool, float, float, Tuple[float, ...]]] = {}
        self.calls = 0

    def group(self, lids: Sequence[str]):
        key = tuple(sorted(lids))
        r = self.cache.get(key)
        if r is None:
            self.calls += 1
            res = self.tree.evaluate_group(key, LAT2NODE, QLAT)
            sur = tuple(res.pressures[LAT2NODE[l]] - HMIN for l in key)
            r = (bool(res.ok), float(res.min_margin), float(np.mean(sur)), sur)
            self.cache[key] = r
        return r

    def descendants(self, n: str) -> Set[str]:
        out, st = set(), [n]
        while st:
            u = st.pop()
            if u not in out:
                out.add(u)
                st.extend(self.tree.children[u])
        return out


def schedule_metrics(ev: Evaluator, groups: List[List[str]], targets: Set[str]) -> Dict:
    """Objective J = 0.5*mu + 0.5*sigma over per-group mean surplus (population std), as in the paper."""
    if not groups:
        return dict(feasible=False, complete=False, J=np.nan, mu=np.nan, sigma=np.nan, min_margin=np.nan, n_groups=0, feas_rate=np.nan)
    rs = [ev.group(g) for g in groups]
    gm = np.array([r[2] for r in rs])
    covered = [l for g in groups for l in g]
    complete = set(covered) == targets and len(covered) == len(targets)
    feas = all(r[0] for r in rs)
    mu, sd = float(gm.mean()), float(gm.std())
    return dict(feasible=feas, complete=complete, J=W_MEAN * mu + W_STD * sd, mu=mu, sigma=sd,
                min_margin=float(min(r[1] for r in rs)), n_groups=len(groups),
                feas_rate=float(np.mean([r[0] for r in rs])))


# ---------------------------------------------------------------- GNN-IL
class GNNPolicy:
    def __init__(self, gen_ckpt=REPO / "models_gnn/generator.pt", feas_ckpt=REPO / "models_gnn/feasibility_v2.pt", device="cpu"):
        self.dev = torch.device(device)
        self.graph = load_graph(str(DATA), device=device)
        ck = torch.load(gen_ckpt, map_location=self.dev, weights_only=False)
        c = ck["config"]
        enc = c.get("encoder", "fusion")
        self.gen = GroupGenerator(c["node_in_dim"], c["edge_attr_dim"], c["hidden"], c["layers"], c["dropout"], **({"encoder": enc} if enc != "fusion" else {})).to(self.dev).eval()
        self.gen.load_state_dict(ck["state_dict"])
        fk = torch.load(feas_ckpt, map_location=self.dev, weights_only=False)
        f = fk["config"]
        self.feas = FeasibilityClassifier(f["node_in_dim"], f["edge_attr_dim"], f["hidden"], f["layers"], f["dropout"]).to(self.dev).eval()
        self.feas.load_state_dict(fk["state_dict"])

    @torch.no_grad()
    def schedule(self, ev: Evaluator, irrigated: Set[str], blocked: Set[str], use_gate=True, hard_check=True,
                 fixed_k: Optional[int] = None, top_k=14, max_expand=24, feas_thr=0.2, max_random=800, seed=123,
                 H0_after: Optional[float] = None, steps_before_change: int = 0):
        """Post-fault rescheduling from a given state. Returns list of groups and wall time."""
        rng = random.Random(seed)
        g = self.graph
        L = len(LAT_IDS)
        irr = torch.zeros((1, L), dtype=torch.uint8, device=self.dev)
        rea = torch.ones((1, L), dtype=torch.uint8, device=self.dev)
        for l in irrigated:
            irr[0, LIDX[l]] = 1
        for l in blocked:
            rea[0, LIDX[l]] = 0
        H0 = torch.tensor([ev.H0], dtype=torch.float32, device=self.dev)
        groups: List[List[str]] = []
        t0 = time.perf_counter()
        while True:
            avail = (rea.bool() & ~irr.bool()).squeeze(0)
            remain = int(avail.sum())
            if remain < 2:
                break
            tok, szl = self.gen(g.node_x, g.edge_index, g.edge_attr, g.lateral_to_node_idx, g.side_idx, H0, irr, rea)
            tok, szl = tok.squeeze(0), szl.squeeze(0)
            k = fixed_k or int(szl.argmax()) + 2
            k = max(2, min(k, remain))
            if remain - k == 1:  # tail-size rule: never leave a single outlet unscheduled (completeness)
                k = k - 1 if k > 2 else 3
            masked = tok.masked_fill(~avail, -1e9)
            sel = None
            for K in (min(top_k, remain), min(max_expand, remain)):
                top = torch.topk(masked, K).indices.tolist()
                combos = sorted(itertools.combinations(top, k), key=lambda c: -float(masked[list(c)].sum()))[:800]
                trial = combos
                if use_gate:
                    pad = torch.full((len(combos), 4), -1, dtype=torch.int64)
                    for i, c in enumerate(combos):
                        pad[i, :k] = torch.tensor(c)
                    n = len(combos)
                    fp = torch.sigmoid(self.feas(g.node_x, g.edge_index, g.edge_attr, g.lateral_to_node_idx, g.side_idx,
                                                 H0.expand(n), irr.expand(n, -1), rea.expand(n, -1), pad.to(self.dev),
                                                 torch.full((n,), k, dtype=torch.int64, device=self.dev))).tolist()
                    t2 = [c for c, p in zip(combos, fp) if p >= feas_thr]
                    trial = t2 or combos
                if not hard_check:
                    sel = list(trial[0])
                    break
                for c in trial:
                    if ev.group([LAT_IDS[i] for i in c])[0]:
                        sel = list(c)
                        break
                if sel is not None:
                    break
            if sel is None and hard_check:
                idx = torch.nonzero(avail).view(-1).tolist()
                for _ in range(max_random):
                    c = rng.sample(idx, k)
                    if ev.group([LAT_IDS[i] for i in c])[0]:
                        sel = c
                        break
            if sel is None:
                break
            groups.append([LAT_IDS[i] for i in sel])
            for i in sel:
                irr[0, i] = 1
        return groups, time.perf_counter() - t0


# ---------------------------------------------------------------- SA (paper setting)
def _build(lats, ev, rng, sizes=(4, 3, 2), tries=600):
    rem = lats[:]
    rng.shuffle(rem)
    out = []
    while rem:
        ok = False
        for k in sizes:
            if len(rem) < k or len(rem) - k == 1:
                continue
            for _ in range(tries):
                gg = rng.sample(rem, k)
                if ev.group(gg)[0]:
                    for l in gg:
                        rem.remove(l)
                    out.append(gg)
                    ok = True
                    break
            if ok:
                break
        if not ok:
            return None
    return out


def _obj(ev, groups):
    gm = []
    for gg in groups:
        r = ev.group(gg)
        if not r[0]:
            return math.inf
        gm.append(r[2])
    gm = np.asarray(gm)
    return W_MEAN * gm.mean() + W_STD * gm.std()


def _neighbor(groups, rng):
    new = [x[:] for x in groups]
    i, j = rng.sample(range(len(new)), 2)
    gi, gj = new[i], new[j]
    if rng.random() < 0.6:
        a, b = rng.choice(gi), rng.choice(gj)
        gi[gi.index(a)] = b
        gj[gj.index(b)] = a
    else:
        src, dst = (gi, gj) if rng.random() < 0.5 else (gj, gi)
        if len(src) > 2 and len(dst) < 4:
            x = rng.choice(src)
            src.remove(x)
            dst.append(x)
        else:
            a, b = rng.choice(gi), rng.choice(gj)
            gi[gi.index(a)] = b
            gj[gj.index(b)] = a
    return new


def sa_schedule(ev: Evaluator, lats: List[str], seed: int, n_init=800, steps=30000, T0=1.0, Tend=1e-3, tries=600, trace=False):
    """Multi-start construction + geometric-cooling SA, identical to the offline teacher (stop = fixed step budget)."""
    rng = random.Random(seed)
    t0 = time.perf_counter()
    best, bestJ = None, math.inf
    hist = []
    for _ in range(n_init):
        g0 = _build(lats, ev, random.Random(rng.randrange(1 << 30)), tries=tries)
        if g0 is not None:
            J = _obj(ev, g0)
            if J < bestJ:
                best, bestJ = g0, J
                if trace:
                    hist.append((time.perf_counter() - t0, bestJ))
    if best is None:
        return [], time.perf_counter() - t0, []
    cur, curJ = best, bestJ
    r2 = random.Random(rng.randrange(1 << 30))
    for t in range(steps):
        T = T0 * (Tend / T0) ** (t / max(steps - 1, 1))
        if len(cur) < 2:
            break
        cand = _neighbor(cur, r2)
        J = _obj(ev, cand)
        if J == math.inf:
            continue
        if J <= curJ or r2.random() < math.exp(-(J - curJ) / T):
            cur, curJ = cand, J
            if J < bestJ:
                best, bestJ = cand, J
                if trace:
                    hist.append((time.perf_counter() - t0, bestJ))
    return best, time.perf_counter() - t0, hist


# ---------------------------------------------------------------- GA baseline
def _decode(perm, ev):
    """First-fit decoder: grow each group along the permutation while the group stays hydraulically feasible (size<=4)."""
    rem = list(perm)
    groups = []
    while rem:
        gg = [rem.pop(0)]
        i = 0
        while len(gg) < 4 and i < len(rem):
            if len(rem) == 2 and len(gg) >= 2:  # avoid stranding a single outlet
                break
            if ev.group(gg + [rem[i]])[0]:
                gg.append(rem.pop(i))
            else:
                i += 1
        groups.append(gg)
    return groups


def _ga_fit(groups, ev):
    bad = sum(1 for g in groups if len(g) < 2 or not ev.group(g)[0])
    if bad:
        return 1e3 * bad
    return _obj(ev, groups)


def ga_schedule(ev: Evaluator, lats: List[str], seed: int, pop=60, gens=300, pc=0.9, pm=0.3, elite=2, tour=3):
    rng = random.Random(seed)
    t0 = time.perf_counter()
    P = [rng.sample(lats, len(lats)) for _ in range(pop)]
    F = [_ga_fit(_decode(p, ev), ev) for p in P]

    def ox(a, b):
        n = len(a)
        i, j = sorted(rng.sample(range(n), 2))
        mid = a[i:j]
        s = set(mid)
        rest = [x for x in b if x not in s]
        return rest[:i] + mid + rest[i:]

    for _ in range(gens):
        order = sorted(range(pop), key=lambda i: F[i])
        newP = [P[i] for i in order[:elite]]
        while len(newP) < pop:
            pa = min(rng.sample(range(pop), tour), key=lambda i: F[i])
            pb = min(rng.sample(range(pop), tour), key=lambda i: F[i])
            ch = ox(P[pa], P[pb]) if rng.random() < pc else P[pa][:]
            if rng.random() < pm:
                i, j = rng.sample(range(len(ch)), 2)
                ch[i], ch[j] = ch[j], ch[i]
            newP.append(ch)
        P = newP
        F = [_ga_fit(_decode(p, ev), ev) for p in P]
    b = min(range(pop), key=lambda i: F[i])
    return _decode(P[b], ev), time.perf_counter() - t0


# ---------------------------------------------------------------- engineering heuristic
def seq_schedule(ev: Evaluator, lats: List[str]):
    """Sequential first-fit grouping along pipe order (practice-style rule: adjacent outlets opened together)."""
    t0 = time.perf_counter()
    order = sorted(lats, key=lambda l: (int(LAT2NODE[l][1:]), l))
    return _decode(order, ev), time.perf_counter() - t0


# ---------------------------------------------------------------- scenario helpers
def prefault_plan(policy: GNNPolicy, ev: Evaluator):
    groups, _ = policy.schedule(ev, set(), set())
    return groups


def make_state(ev: Evaluator, plan: List[List[str]], fault_nodes: Sequence[str], after: int):
    irrigated = {l for g in plan[:after] for l in g}
    bnodes = set().union(*[ev.descendants(n) for n in fault_nodes]) if fault_nodes else set()
    blocked = {l for l in LAT_IDS if LAT2NODE[l] in bnodes and l not in irrigated}
    targets = set(LAT_IDS) - irrigated - blocked
    return irrigated, blocked, targets
