"""Held-out teacher scenarios (scenario-level test split): GNN-IL+ vs SA teacher, per H0 bin."""
import json, collections, numpy as np, torch
from multiprocessing import Pool
from common import *
from anchor_infer import AnchorPolicy

CK = 'models_anchor/anchor_generator.pt'

def scenarios():
    s = torch.load(DATA / 'samples.pt', weights_only=False)
    ck = torch.load(CK, weights_only=False); used = set(ck['split']['train']) | set(ck['split']['val'])
    by = collections.defaultdict(list)
    for i, (src, st) in enumerate(zip(s['source'], s['step_idx'].tolist())): by[src].append((st, i))
    out = []
    for si, (src, v) in enumerate(by.items()):
        if si in used: continue
        v.sort(); i0 = v[0][1]
        out.append(dict(sid=si, src=src, H0=float(s['H0'][i0]), fault=int(s['fault_node_idx'][i0]),
                        irr=[LAT_IDS[j] for j in np.flatnonzero(s['irrigated'][i0].numpy())],
                        blk=[LAT_IDS[j] for j in np.flatnonzero(1 - s['reachable'][i0].numpy())],
                        teacher=[[LAT_IDS[j] for j in np.flatnonzero(s['target'][i].numpy())] for _, i in v]))
    return out

def run(sc):
    torch.set_num_threads(1)
    ap = AnchorPolicy(CK); ev = Evaluator(round(sc['H0'], 2))
    tg = set(LAT_IDS) - set(sc['irr']) - set(sc['blk'])
    mt = schedule_metrics(ev, sc['teacher'], tg)
    g, t = ap.schedule2(Evaluator(ev.H0), set(sc['irr']), set(sc['blk']), gate=False)
    mg = dict(schedule_metrics(ev, g, tg), time=t)
    return dict(sid=sc['sid'], H0=sc['H0'], normal=sc['fault'] < 0, n=len(tg), teacher=mt, gnn=mg)

if __name__ == '__main__':
    S = scenarios(); print(len(S))
    with Pool(4) as p: R = p.map(run, S)
    json.dump(R, open('res_testsplit.json', 'w'), default=float)
    gap = np.array([(r['gnn']['J'] - r['teacher']['J']) / r['teacher']['J'] * 100 for r in R])
    print('test gap mean %.1f median %.1f; feasible %.3f complete %.3f; time %.3f' % (gap.mean(), np.median(gap), np.mean([r['gnn']['feasible'] for r in R]), np.mean([r['gnn']['complete'] for r in R]), np.mean([r['gnn']['time'] for r in R])))
