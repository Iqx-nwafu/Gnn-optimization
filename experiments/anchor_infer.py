"""Inference for the anchor-conditioned generator: anchor = most critical remaining outlet; partners from the network; evaluator hard check."""
import itertools, random, time
import numpy as np, torch
from common import *
from anchor_train import AnchorGenerator, anchor_of, Ln


class AnchorPolicy:
    def __init__(self, ckpt="models_anchor/anchor_generator.pt"):
        ck = torch.load(ckpt, map_location="cpu", weights_only=False); c = ck["config"]
        self.g = load_graph(str(DATA))
        self.m = AnchorGenerator(c["node_in_dim"], c["edge_attr_dim"], encoder=c["encoder"], pair=c["pair"]).eval()
        self.m.load_state_dict(ck["state_dict"])
        fk = torch.load(REPO / "models_gnn/feasibility_v2.pt", map_location="cpu", weights_only=False); f = fk["config"]
        self.feas = FeasibilityClassifier(f["node_in_dim"], f["edge_attr_dim"], f["hidden"], f["layers"], f["dropout"]).eval()
        self.feas.load_state_dict(fk["state_dict"])

    @torch.no_grad()
    def schedule(self, ev, irrigated, blocked, top_m=8, n_eval=20, rank=True, fixed_k=None, target=None, hard_check=True, gate=True, feas_thr=0.2, margin=0.0):
        irr = np.zeros(Ln, bool); rea = np.ones(Ln, bool)
        for l in irrigated: irr[LIDX[l]] = True
        for l in blocked: rea[LIDX[l]] = False
        H0 = torch.tensor([ev.H0]); groups = []; t0 = time.perf_counter()
        # most-unfavourable outlet from the evaluator itself (single-open margin); identical to the training order
        # under normal conditions (property 1), and still correct when pipe parameters change (blockage, ageing)
        single = np.array([ev.group([l])[1] for l in LAT_IDS]); crit = np.empty(Ln, np.int64); crit[np.argsort(single, kind="stable")] = np.arange(Ln)
        while True:
            avail = rea & ~irr; remain = int(avail.sum())
            if remain < 2:
                break
            idx = np.flatnonzero(avail); a = int(idx[np.argmin(crit[idx])])
            tok, sz = self.m(self.g, H0, torch.tensor(irr[None], dtype=torch.uint8), torch.tensor(rea[None], dtype=torch.uint8), torch.tensor([a]))
            tok = tok[0].numpy().copy(); m = avail.copy(); m[a] = False; tok[~m] = -1e9
            k0 = fixed_k or int(sz.argmax()) + 2
            sizes = [k for k in (k0, 3, 2, 4) if 2 <= k <= remain and remain - k != 1]
            sizes = list(dict.fromkeys(sizes))
            cand = np.argsort(-tok)[:min(top_m, int(m.sum()))]
            sel = None
            # safety margin delta: prefer groups whose evaluator min margin >= delta; fall back to delta = 0 (pressure constraint itself)
            for thr in ((margin, 0.0) if margin > 0 else (0.0,)):
                if sel is not None:
                    break
                for k in sizes:
                    combos = sorted(itertools.combinations(cand.tolist(), k - 1), key=lambda c: -tok[list(c)].sum())[:n_eval]
                    if gate and combos:
                        n = len(combos); pad = torch.full((n, 4), -1, dtype=torch.int64)
                        for i, c in enumerate(combos):
                            pad[i, :k] = torch.tensor([a, *c])
                        fp = torch.sigmoid(self.feas(self.g.node_x, self.g.edge_index, self.g.edge_attr, self.g.lateral_to_node_idx, self.g.side_idx,
                                                     H0.expand(n), torch.tensor(irr[None], dtype=torch.uint8).expand(n, -1),
                                                     torch.tensor(rea[None], dtype=torch.uint8).expand(n, -1), pad, torch.full((n,), k))).numpy()
                        kept = [c for c, p in zip(combos, fp) if p >= feas_thr]
                        combos = kept or combos
                    best = None
                    if not hard_check:
                        best = (0, [a, *combos[0]]) if combos else None
                    for c in (combos if hard_check else []):
                        gg = [a, *c]; r = ev.group([LAT_IDS[i] for i in gg])
                        if r[0] and r[1] >= thr:
                            if not rank:
                                best = (0, gg); break
                            sc = r[2] if target is None else abs(r[2] - target)
                            if best is None or sc < best[0]:
                                best = (sc, gg)
                    if best is not None:
                        sel = best[1]; break
            if sel is None:  # anchor cannot be fed with any partner among candidates: random partners, else skip anchor
                idx = [i for i in np.flatnonzero(m)]; rng = random.Random(0)
                for _ in range(400):
                    k = 2 if remain != 3 else 3; c = rng.sample(idx, k - 1)
                    if ev.group([LAT_IDS[i] for i in [a, *c]])[0]:
                        sel = [a, *c]; break
            if sel is None:
                rea[a] = False  # physically infeasible outlet under this head: report as unserved
                continue
            groups.append([LAT_IDS[i] for i in sel]); irr[sel] = True
        return groups, time.perf_counter() - t0


    def schedule2(self, ev, irrigated, blocked, **kw):
        """Two-pass target-level decoding: pass 1 (min surplus) sets the level tau; pass 2 picks groups closest to tau."""
        t0 = time.perf_counter()
        g1, _ = self.schedule(ev, irrigated, blocked, **kw)
        if not g1:
            return g1, time.perf_counter() - t0
        tau = float(np.mean([ev.group(g)[2] for g in g1]))
        g2, _ = self.schedule(ev, irrigated, blocked, target=tau, **kw)
        J = lambda gs: 0.5 * np.mean([ev.group(g)[2] for g in gs]) + 0.5 * np.std([ev.group(g)[2] for g in gs])
        cov = lambda gs: sum(len(g) for g in gs)
        best = min((g1, g2), key=lambda gs: (-cov(gs), J(gs)))  # serve more outlets first, then lower J
        return best, time.perf_counter() - t0
