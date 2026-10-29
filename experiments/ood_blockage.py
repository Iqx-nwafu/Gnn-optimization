"""Partial valve blockage at branch inlets (extra local loss, unseen in training) + a field-node fault; EPANET re-check with the same minor losses."""
import json, random, dataclasses
from multiprocessing import Pool
from common import *
from anchor_infer import AnchorPolicy
from epanet_check import build_wn, epanet_group

INLETS = [e for e in EDGES if e.material == 'PE' and not te.is_field_node_id(e.u)]  # trunk node -> first branch node (branch valve)
FIELDS = sorted(n for n in NODES if te.is_field_node_id(n))


def scen_list(n=10, seed=47):
    rng = random.Random(seed); S = []
    for i in range(n):
        blk = rng.sample([e.edge_id for e in INLETS], rng.choice([1, 2]))
        S.append(dict(id=i, set='Valve blockage', H0=round(rng.uniform(13, 22), 1), faults=[rng.choice(FIELDS)], frac=rng.random(),
                      blockage={b: round(rng.uniform(10, 30), 1) for b in blk}))
    return S


def epa(H0, groups, edges):
    if not groups:
        return dict(epa_feas_rate=float('nan'), epa_min_margin=float('nan'))
    wn = build_wn(H0, edges=edges); mm = [min(v - HMIN for v in epanet_group(wn, g).values()) for g in groups]
    return dict(epa_feas_rate=float(np.mean([m >= 0 for m in mm])), epa_min_margin=float(min(mm)))


def run(sc):
    torch.set_num_threads(1)
    edges = [dataclasses.replace(e, zeta=sc['blockage'].get(e.edge_id, 0.0)) for e in EDGES]
    pol = GNNPolicy(); ev0 = Evaluator(sc['H0']); plan = prefault_plan(pol, ev0)
    af = int(sc['frac'] * len(plan)); ev = Evaluator(sc['H0'], edges)
    irr, blk, tg = make_state(ev, plan, sc['faults'], af); L = sorted(tg)
    rec = dict(sc, after=af, n_remain=len(L), res={})
    if len(L) < 2:
        return rec
    ap = AnchorPolicy()
    g, t = ap.schedule2(Evaluator(sc['H0'], edges), irr, blk, gate=False)
    rec['res']['GNN-IL+'] = dict(schedule_metrics(ev, g, tg), time=t, unserved=len(tg) - sum(map(len, g)), **epa(sc['H0'], g, edges))
    g, t, _ = sa_schedule(Evaluator(sc['H0'], edges), L, 100); rec['res']['SA'] = dict(schedule_metrics(ev, g, tg), time=t, **epa(sc['H0'], g, edges))
    g, t, _ = sa_schedule(Evaluator(sc['H0'], edges), L, 300, n_init=10, steps=10000); rec['res']['SA-fast (10, 10000)'] = dict(schedule_metrics(ev, g, tg), time=t, **epa(sc['H0'], g, edges))
    g, t = seq_schedule(Evaluator(sc['H0'], edges), L); rec['res']['Sequential'] = dict(schedule_metrics(ev, g, tg), time=t, **epa(sc['H0'], g, edges))
    return rec


if __name__ == '__main__':
    with Pool(4) as p:
        R = p.map(run, scen_list())
    json.dump(R, open('res_blockage.json', 'w'), default=float)
    for r in R:
        print(r['id'], r['H0'], r['blockage'], r['n_remain'], {k: (round(v['J'], 3), v['feasible'], v['complete'], round(v['epa_min_margin'], 2)) for k, v in r['res'].items()})
