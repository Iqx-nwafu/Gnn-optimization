"""Out-of-distribution / complex-condition benchmark with EPANET re-check (pretrained model, no retraining)."""
import json, random
from multiprocessing import Pool
from common import *
from epanet_check import build_wn, epanet_group

FIELDS = sorted(n for n in NODES if te.is_field_node_id(n))
BR = lambda n: int(n[1:-1])  # branch id of a field node (J53 -> 5)

def scen_list(seed=31):
    rng = random.Random(seed); S = []
    for H in [10.8, 11.0, 11.2, 23.5, 24.5, 25.5, 27.0]:                    # unseen source-head range
        for _ in range(2): S.append(dict(set='OOD-head', H0=H, faults=[rng.choice(FIELDS)], frac=rng.random()))
    for n in ['J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10']:       # unseen fault region: main pipe
        S.append(dict(set='OOD-trunk', H0=round(rng.uniform(12, 22), 1), faults=[n], frac=rng.random()))
    for _ in range(10):                                                        # concurrent multi-node faults
        k = rng.choice([2, 2, 3]); f = rng.sample(FIELDS, k)
        S.append(dict(set='Multi-fault', H0=round(rng.uniform(12, 22), 1), faults=f, frac=rng.random()))
    for _ in range(8):                                                         # fault + source head drop
        H = round(rng.uniform(13, 22), 1)
        S.append(dict(set='Fault+head drop', H0=H, H0_after=round(H - rng.uniform(1.0, 3.0), 1), faults=[rng.choice(FIELDS)], frac=rng.random()))
    for i, s in enumerate(S): s['id'] = i
    return S

def epa(H0, groups):
    if not groups: return dict(epa_feas_rate=float('nan'), epa_min_margin=float('nan'))
    wn = build_wn(H0); mm = []
    for g in groups:
        p = epanet_group(wn, g); mm.append(min(v - HMIN for v in p.values()))
    return dict(epa_feas_rate=float(np.mean([m >= 0 for m in mm])), epa_min_margin=float(min(mm)))

def run(sc):
    pol = GNNPolicy(); ev0 = Evaluator(sc['H0']); plan = prefault_plan(pol, ev0)
    af = int(sc['frac'] * len(plan))
    H1 = sc.get('H0_after', sc['H0']); ev = ev0 if H1 == sc['H0'] else Evaluator(H1)
    irr, blk, tg = make_state(ev, plan, sc['faults'], af); L = sorted(tg)
    rec = dict(sc, after=af, n_remain=len(L), prefault_epa=epa(sc['H0'], plan[:af]), res={})
    if len(L) < 2: return rec
    g, t = pol.schedule(ev, irr, blk); rec['res']['GNN-IL'] = dict(schedule_metrics(ev, g, tg), time=t, **epa(H1, g))
    g, t = seq_schedule(ev, L); rec['res']['Sequential'] = dict(schedule_metrics(ev, g, tg), time=t, **epa(H1, g))
    g, t, _ = sa_schedule(ev, L, 100); rec['res']['SA'] = dict(schedule_metrics(ev, g, tg), time=t, **epa(H1, g))
    return rec

if __name__ == '__main__':
    out = []
    with Pool(4) as p:
        for r in p.imap_unordered(run, scen_list()):
            out.append(r); print(r['id'], r['set'], r['H0'], r['faults'], r['n_remain'], {k: (round(v['J'], 3), v['feasible'], v['complete'], round(v['epa_min_margin'], 2)) for k, v in r['res'].items()}, flush=True)
            json.dump(out, open('res_ood.json', 'w'), default=float)
