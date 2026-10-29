"""Anchor-conditioned GNN-IL (no change to teacher data): canonical 'most-critical-first' ordering + pairwise shared-path resistance."""
from __future__ import annotations
import argparse, collections, csv, json, math, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

from common import NODES, EDGES, LAT_IDS, LAT2NODE, DATA, HMIN, QLAT  # also puts the repository root on sys.path
from model_gnn import NodeEdgeFusionEncoder, ENCODERS, MLP
from utils_data import load_graph
import tree_evaluator as te

# ---------------------------------------------------------------- physics priors
_tree = te.TreeHydraulicEvaluator(NODES, EDGES, "J0", 30.0, HMIN)


def _r(e):
    c = te.GBT_COEFS[(e.material, "turbulent")]
    return c.f * 3.6e6 ** c.m * e.L / (e.D * 1000) ** c.b


def path_edges(v):
    out = []
    while _tree.parent[v] is not None:
        out.append(_tree.parent_edge[v]); v = _tree.parent[v]
    return out


_paths = {n: set(e.edge_id for e in path_edges(n)) for n in NODES}
_rmap = {e.edge_id: _r(e) for e in EDGES}
Ln = len(LAT_IDS)
# shared upstream resistance between two laterals (coupling strength, property 3)
SHARED = np.zeros((Ln, Ln), dtype=np.float32)
for i, a in enumerate(LAT_IDS):
    for j, b in enumerate(LAT_IDS):
        SHARED[i, j] = sum(_rmap[e] for e in _paths[LAT2NODE[a]] & _paths[LAT2NODE[b]])
SHARED = np.log1p(SHARED / SHARED.max() * 100.0) / math.log(101.0)
# single-open margin at a reference head (H0-shift invariant ordering, property 1)
_single = np.array([_tree.evaluate_group([l], LAT2NODE, QLAT).min_margin for l in LAT_IDS])
CRIT_ORDER = np.argsort(_single, kind="stable")  # most critical (smallest margin) first
RANK = np.empty(Ln, dtype=np.int64); RANK[CRIT_ORDER] = np.arange(Ln)


def anchor_of(avail: np.ndarray) -> int:
    idx = np.flatnonzero(avail)
    return int(idx[np.argmin(RANK[idx])])


# ---------------------------------------------------------------- model
class AnchorGenerator(nn.Module):
    def __init__(self, node_in_dim, edge_attr_dim, hidden=128, layers=4, dropout=0.1, encoder="fusion", pair=True):
        super().__init__()
        self.encoder = ENCODERS[encoder](node_in_dim, edge_attr_dim, hidden, layers, dropout)
        self.side_emb = nn.Embedding(2, hidden)
        self.h0_mlp = MLP(1, hidden, hidden, dropout)
        self.pair = pair
        self.register_buffer("shared", torch.tensor(SHARED))
        din = 4 * hidden + 2 + (1 if pair else 0)
        self.token_head = nn.Sequential(nn.Linear(din, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.size_head = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 3))

    def forward(self, g, H0, irr, rea, anchor):
        B = H0.size(0)
        nh = self.encoder(g.node_x, g.edge_index, g.edge_attr)
        lat = nh[g.lateral_to_node_idx] + self.side_emb(g.side_idx)  # [L,H]
        latb = lat.unsqueeze(0).expand(B, -1, -1)
        a = lat[anchor]  # [B,H]
        ab = a.unsqueeze(1).expand_as(latb)
        h0 = self.h0_mlp((H0.view(B, 1) - 17.0) / 4.0)
        avail = (rea.bool() & ~irr.bool()).float()
        pooled = (latb * avail.unsqueeze(-1)).sum(1) / avail.sum(1, keepdim=True).clamp_min(1)
        feats = [latb, ab, latb * ab, h0.unsqueeze(1).expand_as(latb), irr.float().unsqueeze(-1), rea.float().unsqueeze(-1)]
        if self.pair:
            feats.append(self.shared[anchor].unsqueeze(-1))
        tok = self.token_head(torch.cat(feats, -1)).squeeze(-1)
        size = self.size_head(torch.cat([pooled, a, h0], -1))
        return tok, size


# ---------------------------------------------------------------- data: canonical re-ordering of teacher schedules
def build_anchor_samples():
    s = torch.load(DATA / "samples.pt", weights_only=False)
    by = collections.defaultdict(list)
    for i, (src, st) in enumerate(zip(s["source"], s["step_idx"].tolist())):
        by[src].append((st, i))
    H0s, IRR, REA, ANC, TGT, SZ, SCEN = [], [], [], [], [], [], []
    for si, (src, v) in enumerate(by.items()):
        v.sort(); i0 = v[0][1]
        groups = [np.flatnonzero(s["target"][i].numpy()) for _, i in v]
        irr = s["irrigated"][i0].numpy().astype(bool).copy(); rea = s["reachable"][i0].numpy().astype(bool)
        H0 = float(s["H0"][i0]); remaining = [set(g.tolist()) for g in groups]
        while remaining:
            a = anchor_of(rea & ~irr)
            gi = next(k for k, gg in enumerate(remaining) if a in gg)
            gg = remaining.pop(gi)
            t = np.zeros(Ln, dtype=np.uint8); t[list(gg - {a})] = 1
            H0s.append(H0); IRR.append(irr.copy()); REA.append(rea.copy()); ANC.append(a); TGT.append(t); SZ.append(len(gg)); SCEN.append(si)
            irr[list(gg)] = True
    return dict(H0=torch.tensor(H0s), irr=torch.tensor(np.array(IRR), dtype=torch.uint8), rea=torch.tensor(np.array(REA), dtype=torch.uint8),
                anchor=torch.tensor(ANC), target=torch.tensor(np.array(TGT)), size=torch.tensor(SZ), scen=torch.tensor(SCEN))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models_anchor")
    ap.add_argument("--encoder", default="fusion")
    ap.add_argument("--no_pair", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    g = load_graph(str(DATA))
    D = build_anchor_samples()
    # scenario-level split 8:1:1 (no leakage of states from the same schedule across splits)
    nS = int(D["scen"].max()) + 1; perm = torch.randperm(nS, generator=torch.Generator().manual_seed(args.seed))
    tr_s, va_s = set(perm[: int(0.8 * nS)].tolist()), set(perm[int(0.8 * nS): int(0.9 * nS)].tolist())
    sc = D["scen"].tolist()
    tr = torch.tensor([i for i, x in enumerate(sc) if x in tr_s]); va = torch.tensor([i for i, x in enumerate(sc) if x in va_s])
    model = AnchorGenerator(g.node_x.shape[1], g.edge_attr.shape[1], encoder=args.encoder, pair=not args.no_pair)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    def batch_loss(idx, train):
        tok, sz = model(g, D["H0"][idx], D["irr"][idx], D["rea"][idx], D["anchor"][idx])
        avail = (D["rea"][idx].bool() & ~D["irr"][idx].bool())
        avail[torch.arange(len(idx)), D["anchor"][idx]] = False
        tok = tok.masked_fill(~avail, -1e4)
        tgt = D["target"][idx].float()
        # listwise loss: softmax over available laterals, multi-positive
        logp = F.log_softmax(tok, -1)
        l_tok = -(logp * tgt).sum(-1).div(tgt.sum(-1).clamp_min(1)).mean()
        l_sz = F.cross_entropy(sz, (D["size"][idx] - 2).clamp(0, 2))
        k = tgt.sum(-1).long()
        with torch.no_grad():
            top = tok.topk(3, -1).indices
            hit = torch.stack([tgt[b, top[b, :k[b]]].sum() / k[b].clamp_min(1) for b in range(len(idx))]).mean()
        return l_tok + 0.2 * l_sz, float(hit)

    log = open(out / "train_log.csv", "w", newline=""); w = csv.writer(log); w.writerow(["epoch", "split", "loss", "partner_acc"])
    best = 1e9; t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); p = tr[torch.randperm(len(tr))]
        for b in range(0, len(p), args.batch):
            loss, _ = batch_loss(p[b:b + args.batch], True)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        sched.step(); model.eval()
        res = {}
        with torch.no_grad():
            for name, ids in (("train", tr[:8000]), ("val", va)):
                L = []; A = []
                for b in range(0, len(ids), 1024):
                    l, a = batch_loss(ids[b:b + 1024], False); n = len(ids[b:b + 1024]); L.append(float(l) * n); A.append(a * n)
                res[name] = (sum(L) / len(ids), sum(A) / len(ids)); w.writerow([ep, name, *res[name]])
        log.flush()
        print(f"[E{ep:03d}] {time.time()-t0:.0f}s train loss={res['train'][0]:.4f} acc={res['train'][1]:.3f} | val loss={res['val'][0]:.4f} acc={res['val'][1]:.3f}", flush=True)
        if res["val"][0] < best:
            best = res["val"][0]
            torch.save({"state_dict": model.state_dict(), "config": dict(node_in_dim=g.node_x.shape[1], edge_attr_dim=g.edge_attr.shape[1], encoder=args.encoder, pair=not args.no_pair),
                        "split": dict(train=sorted(tr_s), val=sorted(va_s))}, out / "anchor_generator.pt")
    json.dump(dict(time_s=time.time() - t0, n_train=len(tr), n_val=len(va)), open(out / "train_meta.json", "w"))


if __name__ == "__main__":
    main()
