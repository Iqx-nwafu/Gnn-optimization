"""EPANET (via WNTR) re-check of tree-evaluator group pressures."""
import math, os, tempfile, numpy as np, wntr
from common import NODES, EDGES, LAT2NODE, QLAT, HMIN
import tree_evaluator as te

def build_wn(H0, C=150.0, edges=None):
    wn = wntr.network.WaterNetworkModel()
    wn.options.hydraulic.headloss = 'H-W'
    wn.options.hydraulic.demand_model = 'DD'
    for nid, n in NODES.items():
        if nid == 'J0':
            wn.add_reservoir('R0', base_head=H0)
        else:
            wn.add_junction(nid, base_demand=0.0, elevation=n.z)
    for e in (edges or EDGES):
        u = 'R0' if e.u == 'J0' else e.u; v = 'R0' if e.v == 'J0' else e.v
        wn.add_pipe(e.edge_id, u, v, length=e.L, diameter=e.D, roughness=C, minor_loss=e.zeta)
    return wn

def epanet_group(wn, lids):
    for nid in NODES:
        if nid != 'J0': wn.get_node(nid).demand_timeseries_list[0].base_value = 0.0
    for l in lids:
        j = wn.get_node(LAT2NODE[l]); j.demand_timeseries_list[0].base_value += QLAT
    res = wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=os.path.join(tempfile.gettempdir(), f'epa_{os.getpid()}'))
    p = res.node['pressure'].iloc[0]
    return {LAT2NODE[l]: float(p[LAT2NODE[l]]) for l in lids}

def hw_loss(Q, L, D, C=150.0):
    return 0.0 if Q <= 0 else 10.667 * L * Q ** 1.852 / (C ** 1.852 * D ** 4.871)

if __name__ == '__main__':
    # (a) solver consistency: tree evaluator with H-W vs EPANET H-W on random groups
    import random
    te.friction_loss_gbt_orig = te.friction_loss_gbt
    rng = random.Random(0); lats = list(LAT2NODE)
    wn = build_wn(18.0)
    te.friction_loss_gbt = lambda Q, L, D, m, nu_m2s=None: hw_loss(Q, L, D)
    ev = te.TreeHydraulicEvaluator(NODES, EDGES, 'J0', 18.0, HMIN)
    d = []
    for _ in range(200):
        g = rng.sample(lats, rng.choice([2, 3, 4]))
        r = ev.evaluate_group(g, LAT2NODE, QLAT); pe = epanet_group(wn, g)
        d += [abs(r.pressures[n] - pe[n]) for n in pe]
    print('H-W tree vs EPANET: max|dP|=%.2e mean=%.2e' % (max(d), np.mean(d)))
    te.friction_loss_gbt = te.friction_loss_gbt_orig
    ev = te.TreeHydraulicEvaluator(NODES, EDGES, 'J0', 18.0, HMIN)
    d = []
    for _ in range(200):
        g = rng.sample(lats, rng.choice([2, 3, 4]))
        r = ev.evaluate_group(g, LAT2NODE, QLAT); pe = epanet_group(wn, g)
        d += [r.pressures[n] - pe[n] for n in pe]
    print('GB tree - EPANET HW(C150): max=%.3f min=%.3f mean=%.3f' % (max(d), min(d), np.mean(d)))
