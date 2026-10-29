"""Full-network scheduling under normal operation (all 120 laterals, the largest instance) over the training head range:
GNN-IL vs SA / SA-fast / GA / Sequential, single CPU thread per job, EPANET re-check."""
import json, numpy as np
from multiprocessing import Pool
from common import *
from anchor_infer import AnchorPolicy
from bench_ood import epa

HEADS = [float(h) for h in range(12, 24)]


def run(job):
    torch.set_num_threads(1)
    H, meth, s = job; ev = Evaluator(H); L = sorted(LAT_IDS)
    if meth.startswith('GNN-IL'):
        g, t = AnchorPolicy().schedule2(ev, set(), set(), gate=False, margin=s)
    elif meth == 'SA':
        g, t, _ = sa_schedule(ev, L, 100 + s)
    elif meth.startswith('SA-fast'):
        g, t, _ = sa_schedule(ev, L, 300 + s, n_init=10, steps=10000 if '10000' in meth else 2000)
    elif meth == 'GA':
        g, t = ga_schedule(ev, L, 200 + s)
    else:
        g, t = seq_schedule(ev, L)
    return dict(H0=H, method=meth, seed=s, time=t, groups=g, **schedule_metrics(Evaluator(H), g, set(L)), **epa(H, g))


if __name__ == '__main__':
    jobs = [(H, m, s) for H in HEADS for m, ss in (('GNN-IL', [0.0]), ('GNN-IL d0.5', [0.5]), ('SA', range(3)), ('SA-fast (10, 10000)', range(3)),
                                                   ('SA-fast (10, 2000)', range(3)), ('GA', range(3)), ('Sequential', [0])) for s in ss]
    with Pool(4) as p: R = p.map(run, jobs, chunksize=1)
    json.dump(R, open('res_fullnet.json', 'w'), default=float); print('done')
