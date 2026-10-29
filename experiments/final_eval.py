"""Final in-distribution comparison (uncontended timing: 4 single-thread workers on 4 cores)."""
import json, sys, numpy as np
from multiprocessing import Pool
from common import *
from anchor_infer import AnchorPolicy
from bench_id import scen_list

VARIANTS = {  # name: (ckpt, method, kwargs)
    'GNN-IL+ (full)': ('models_anchor/anchor_generator.pt', 'schedule2', dict(gate=False)),
    'GNN-IL+ with gate': ('models_anchor/anchor_generator.pt', 'schedule2', {}),
    'GNN-IL+ w/o target-level': ('models_anchor/anchor_generator.pt', 'schedule', dict(gate=False)),
    'GNN-IL+ w/o evaluator ranking': ('models_anchor/anchor_generator.pt', 'schedule', dict(rank=False, gate=False)),
    'GNN-IL+ w/o hard check': ('models_anchor/anchor_generator.pt', 'schedule', dict(hard_check=False, gate=False)),
    'GNN-IL+ w/o size head (k=4)': ('models_anchor/anchor_generator.pt', 'schedule2', dict(fixed_k=4, gate=False)),
    'GNN-IL+ w/o path prior': ('models_anchor_nopair/anchor_generator.pt', 'schedule2', dict(gate=False)),
    'GCN encoder': ('models_anchor_gcn/anchor_generator.pt', 'schedule2', dict(gate=False)),
    'MLP encoder (no graph)': ('models_anchor_mlp/anchor_generator.pt', 'schedule2', dict(gate=False)),
}
SAFAST = {'SA-fast (10, 2000)': (10, 2000), 'SA-fast (10, 10000)': (10, 10000)}

def run(sc):
    torch.set_num_threads(1)
    pol = GNNPolicy(); ev = Evaluator(sc['H0']); plan = prefault_plan(pol, ev)
    af = int(sc['frac'] * len(plan)); irr, blk, tg = make_state(ev, plan, sc['faults'], af); L = sorted(tg)
    out = {}; cache = {}
    for name, (ck, meth, kw) in VARIANTS.items():
        if ck not in cache: cache[ck] = AnchorPolicy(ck)
        ev2 = Evaluator(sc['H0'])  # fresh cache -> fair timing
        g, t = getattr(cache[ck], meth)(ev2, irr, blk, **kw)
        out[name] = dict(schedule_metrics(ev, g, tg), time=t, groups=g, eval_calls=ev2.calls)
    for name, (ni, st) in SAFAST.items():
        for s in range(3):
            ev2 = Evaluator(sc['H0']); g, t, _ = sa_schedule(ev2, L, 300 + s, n_init=ni, steps=st)
            out[f'{name}#{s}'] = dict(schedule_metrics(ev, g, tg), time=t, groups=g)
    return sc['id'], out

if __name__ == '__main__':
    with Pool(4) as p: res = dict(p.map(run, scen_list()))
    json.dump(res, open('res_final.json', 'w'), default=float)
    print('done')
