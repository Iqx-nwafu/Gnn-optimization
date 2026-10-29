"""mu-sigma trade-off of SA under different objective weights (normal operation, full schedule), plus engineering indicators."""
import json, numpy as np
from multiprocessing import Pool
import common as C
from common import *

def run(arg):
    H0, wm, seed = arg
    C.W_MEAN, C.W_STD = wm, 1 - wm
    ev = Evaluator(H0)
    g, t, _ = sa_schedule(ev, sorted(LAT_IDS), seed)
    C.W_MEAN = C.W_STD = 0.5
    m = schedule_metrics(ev, g, set(LAT_IDS))
    return dict(H0=H0, w_mean=wm, seed=seed, time=t, groups=g, **m)

if __name__ == '__main__':
    args = [(H, w, s) for H in (13.0, 17.0, 21.0) for w in (1.0, 0.85, 0.7, 0.5, 0.3, 0.15, 0.0) for s in (1, 2)]
    with Pool(4) as p: R = p.map(run, args)
    json.dump(R, open('res_pareto.json', 'w'), default=float)
    for r in R: print(r['H0'], r['w_mean'], r['seed'], round(r['mu'], 3), round(r['sigma'], 3), r['n_groups'], round(r['min_margin'], 2))
