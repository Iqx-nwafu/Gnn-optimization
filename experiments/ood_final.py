"""GNN-IL+ on the OOD / complex-condition scenarios, with EPANET re-check (adds to res_ood.json)."""
import json
from multiprocessing import Pool
from common import *
from anchor_infer import AnchorPolicy
from bench_ood import scen_list, epa

def run(sc):
    torch.set_num_threads(1)
    pol = GNNPolicy(); ap = AnchorPolicy(); ev0 = Evaluator(sc['H0']); plan = prefault_plan(pol, ev0)
    af = int(sc['frac'] * len(plan)); H1 = sc.get('H0_after', sc['H0']); ev = ev0 if H1 == sc['H0'] else Evaluator(H1)
    irr, blk, tg = make_state(ev, plan, sc['faults'], af); L = sorted(tg)
    out = {}
    if len(L) < 2: return sc['id'], out
    g, t = ap.schedule2(Evaluator(H1), irr, blk, gate=False); out['GNN-IL+'] = dict(schedule_metrics(ev, g, tg), time=t, unserved=len(tg) - sum(map(len, g)), **epa(H1, g))
    g, t, _ = sa_schedule(Evaluator(H1), L, 300, n_init=10, steps=10000); out['SA-fast (10, 10000)'] = dict(schedule_metrics(ev, g, tg), time=t, **epa(H1, g))
    return sc['id'], out

if __name__ == '__main__':
    with Pool(4) as p: res = dict(p.map(run, scen_list()))
    json.dump(res, open('res_ood_final.json', 'w'), default=float); print('done')
