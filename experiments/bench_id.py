"""In-distribution benchmark: single field-node faults, H0 in training range. One process per scenario."""
import json, random, sys
from multiprocessing import Pool
from common import *

def scen_list(n=30, seed=2026):
    rng = random.Random(seed); fields = sorted(n for n in NODES if te.is_field_node_id(n))
    return [dict(id=i, H0=round(rng.uniform(12, 22), 1), faults=[rng.choice(fields)], frac=rng.random()) for i in range(n)]

def run(sc):
    pol = GNNPolicy(); ev = Evaluator(sc['H0']); plan = prefault_plan(pol, ev)
    af = int(sc['frac'] * len(plan)); irr, blk, tg = make_state(ev, plan, sc['faults'], af); L = sorted(tg)
    rec = dict(sc, after=af, n_remain=len(L), res={})
    if len(L) < 2: return rec
    g, t = pol.schedule(ev, irr, blk); rec['res']['GNN-IL'] = dict(schedule_metrics(ev, g, tg), time=t, groups=g)
    g, t = pol.schedule(ev, irr, blk, use_gate=False); rec['res']['GNN-IL w/o gate'] = dict(schedule_metrics(ev, g, tg), time=t)
    g, t = pol.schedule(ev, irr, blk, use_gate=False, hard_check=False); rec['res']['GNN-IL w/o gate & check'] = dict(schedule_metrics(ev, g, tg), time=t)
    g, t = pol.schedule(ev, irr, blk, fixed_k=4); rec['res']['GNN-IL fixed k=4'] = dict(schedule_metrics(ev, g, tg), time=t)
    g, t = seq_schedule(ev, L); rec['res']['Sequential'] = dict(schedule_metrics(ev, g, tg), time=t)
    for s in range(3):
        g, t, h = sa_schedule(ev, L, 100 + s, trace=True); rec['res'][f'SA#{s}'] = dict(schedule_metrics(ev, g, tg), time=t, trace=h, groups=g)
        g, t = ga_schedule(ev, L, 200 + s); rec['res'][f'GA#{s}'] = dict(schedule_metrics(ev, g, tg), time=t)
    return rec

if __name__ == '__main__':
    out = []
    with Pool(4) as p:
        for r in p.imap_unordered(run, scen_list()):
            out.append(r); print(r['id'], r['H0'], r['faults'], {k: round(v['J'], 3) for k, v in r['res'].items()}, flush=True)
            json.dump(out, open('res_id.json', 'w'), default=float)
