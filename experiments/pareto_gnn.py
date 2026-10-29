"""GNN-IL+, GA and Sequential full schedules (normal operation) at the Pareto-study heads, for the mu-sigma trade-off plot."""
import json
from common import *
from anchor_infer import AnchorPolicy

if __name__ == '__main__':
    torch.set_num_threads(1); ap = AnchorPolicy(); out = []
    for H in (13.0, 17.0, 21.0):
        ev = Evaluator(H); L = sorted(LAT_IDS)
        g, t = ap.schedule2(Evaluator(H), set(), set(), gate=False); out.append(dict(method='GNN-IL+', H0=H, time=t, groups=g, **schedule_metrics(ev, g, set(L))))
        g, t = seq_schedule(ev, L); out.append(dict(method='Sequential', H0=H, time=t, groups=g, **schedule_metrics(ev, g, set(L))))
        for s in range(2):
            g, t = ga_schedule(ev, L, 200 + s); out.append(dict(method='GA', H0=H, seed=s, time=t, groups=g, **schedule_metrics(ev, g, set(L))))
        print(H, [(o['method'], round(o['mu'], 3), round(o['sigma'], 3), o['n_groups']) for o in out if o['H0'] == H], flush=True)
    json.dump(out, open('res_pareto_gnn.json', 'w'), default=float)
