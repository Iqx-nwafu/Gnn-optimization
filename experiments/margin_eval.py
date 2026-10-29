"""Hard-check safety margin delta (online setting, no retraining): GNN-IL+ with delta in DELTAS on the in-distribution,
OOD and valve-blockage scenario sets, all re-checked with EPANET 2.2."""
import json, dataclasses
from multiprocessing import Pool
from common import *
from anchor_infer import AnchorPolicy
import bench_id, bench_ood, ood_blockage
from bench_ood import epa

DELTAS = [0.2, 0.5]


def run(job):
    torch.set_num_threads(1)
    kind, sc = job; pol = GNNPolicy(); ap = AnchorPolicy()
    edges = [dataclasses.replace(e, zeta=sc.get('blockage', {}).get(e.edge_id, 0.0)) for e in EDGES]
    ev0 = Evaluator(sc['H0']); plan = prefault_plan(pol, ev0); af = int(sc['frac'] * len(plan))
    H1 = sc.get('H0_after', sc['H0']); ev = Evaluator(H1, edges)
    irr, blk, tg = make_state(ev, plan, sc['faults'], af)
    out = {}
    if len(tg) < 2:
        return kind, sc['id'], out
    for d in DELTAS:
        g, t = ap.schedule2(Evaluator(H1, edges), irr, blk, gate=False, margin=d)
        e = ood_blockage.epa(H1, g, edges) if 'blockage' in sc else epa(H1, g)
        out[f'delta={d}'] = dict(schedule_metrics(ev, g, tg), time=t, unserved=len(tg) - sum(map(len, g)), groups=g, **e)
    return kind, sc['id'], out


if __name__ == '__main__':
    ood = bench_ood.scen_list()
    for i, s in enumerate(ood): s.setdefault('id', i)
    jobs = [('ID', s) for s in bench_id.scen_list()] + [('OOD', s) for s in ood] + [('BLK', s) for s in ood_blockage.scen_list()]
    with Pool(4) as p: R = p.map(run, jobs)
    res = {}
    for kind, sid, out in R: res.setdefault(kind, {})[str(sid)] = out
    json.dump(res, open('res_margin.json', 'w'), default=float); print('done')
