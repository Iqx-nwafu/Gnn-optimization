"""Tables 1-4 (three-line, bilingual headers) and the numbers quoted in the text -> build/tables.json, build/stats.json."""
import json, collections
from pathlib import Path
import numpy as np, pandas as pd

B = Path('tables'); B.mkdir(exist_ok=True)
pm = lambda a, d=2: f'{np.mean(a):.{d}f}±{np.std(a):.{d}f}'.replace('-', '−')
f = lambda x, d=2: f'{x:.{d}f}'
S, T = {}, {}

# ---------------------------------------------------------------- Table 1 (static)
T['t1'] = dict(cap_cn='表1  自压滴灌系统设计参数与管网参数', cap_en='Table 1  Design and pipe-network parameters of the self-pressure drip irrigation system',
               widths=[3300, 1590], header=[['参数\nParameter', '数值\nValue']], rows=[
    ['灌溉水利用系数\nIrrigation water use coefficient', '0.9'],
    ['土壤含水量设计上限\nDesign upper limit of soil water content/%', '90'],
    ['土壤含水量设计下限\nDesign lower limit of soil water content/%', '70'],
    ['计划湿润层深度\nPlanned wetting layer depth/cm', '50'],
    ['田间持水量\nField capacity/%', '28'],
    ['设计土壤湿度比\nDesign soil moisture ratio/%', '75'],
    ['设计用水强度\nDesign water consumption intensity/(mm·d⁻¹)', '6'],
    ['节点数/管段数\nNumber of nodes/pipes', '71/70'],
    ['干管管材与管径\nMain pipe material and diameter/mm', 'UPVC, 250'],
    ['支管管材与管径\nBranch pipe material and diameter/mm', 'PE, 160'],
    ['管段长度\nPipe length/m', '50~100'],
    ['干管/支管坡降\nSlope of main/branch pipes', '0.003 9/0.006'],
    ['节点高程\nNode elevation/m', '−7.005~0'],
    ['斗管数量\nNumber of laterals', '120'],
    ['斗管设计流量\nLateral design flow/(m³·s⁻¹)', '0.012'],
    ['最小工作水头\nMinimum working head/m', '11.59'],
    ['摩阻系数$f$, $m$, $b$（PE）\nFriction coefficients (PE)', '0.505, 1.75, 4.75'],
    ['摩阻系数$f$, $m$, $b$（UPVC）\nFriction coefficients (UPVC)', '0.464, 1.77, 4.77'],
], notes=['注：摩阻系数按GB/T 50485—2020取值，流量单位为L·h⁻¹，管径单位为mm。', 'Note: Friction coefficients follow GB/T 50485—2020 with flow in L·h⁻¹ and diameter in mm.'])

# ---------------------------------------------------------------- in-distribution comparison
F = json.load(open('res_final.json')); R = {str(r['id']): r for r in json.load(open('res_id.json'))}
E = json.load(open('res_final_epa.json'))
sids = sorted(R, key=int)
best = {s: min(R[s]['res'][f'SA#{k}']['J'] for k in range(3)) for s in sids}


def per_scen(getter, seeds=None):
    """-> dict of arrays over scenarios (seed-averaged for stochastic methods) + mean across-seed std of J."""
    rows = collections.defaultdict(list); sstd = []
    for s in sids:
        vs = [getter(s, k) for k in (range(seeds) if seeds else [None])]
        rows['J'].append(np.mean([v['J'] for v in vs])); rows['gap'].append(np.mean([(v['J'] - best[s]) / best[s] * 100 for v in vs]))
        rows['mu'].append(np.mean([v['mu'] for v in vs])); rows['sigma'].append(np.mean([v['sigma'] for v in vs]))
        rows['mm'].append(np.mean([v['min_margin'] for v in vs])); rows['ng'].append(np.mean([v['n_groups'] for v in vs]))
        rows['feas'].append(np.mean([v['feasible'] and v['complete'] for v in vs])); rows['time'].append(np.mean([v['time'] for v in vs]))
        if seeds:
            sstd.append(np.std([v['J'] for v in vs]))
    return {k: np.array(v) for k, v in rows.items()}, (np.mean(sstd) if sstd else None)


METHODS = [
    ('GNN-IL（本文）\nGNN-IL (ours)', lambda s, k: F[s]['GNN-IL+ (full)'], None, lambda s: E[s]['GNN-IL+ (full)']['allok']),
    ('SA（800+30 000）', lambda s, k: R[s]['res'][f'SA#{k}'], 3, lambda s: np.mean([E[s][f'SA#{k}']['allok'] for k in range(3)])),
    ('SA-fast\n（10+10 000）', lambda s, k: F[s][f'SA-fast (10, 10000)#{k}'], 3, lambda s: np.mean([E[s][f'SA-fast (10, 10000)#{k}']['allok'] for k in range(3)])),
    ('SA-fast\n（10+2 000）', lambda s, k: F[s][f'SA-fast (10, 2000)#{k}'], 3, lambda s: np.mean([E[s][f'SA-fast (10, 2000)#{k}']['allok'] for k in range(3)])),
    ('GA', lambda s, k: R[s]['res'][f'GA#{k}'], 3, lambda s: np.mean([E[s][f'GA#{k}']['allok'] for k in range(3)])),
    ('顺序编组\nSequential', lambda s, k: R[s]['res']['Sequential'], None, lambda s: E[s]['Sequential']['allok']),
]
MG0 = json.load(open('res_margin.json'))['ID']
METHODS.insert(1, ('GNN-IL\n（$\\delta$=0.5 m）', lambda s, k: MG0[s]['delta=0.5'], None, lambda s: MG0[s]['delta=0.5']['epa_feas_rate'] == 1))
tfmt = lambda t: f"{t.mean():.2f}" if t.mean() >= 0.01 else f"{t.mean():.3f}"
rows = ['（a）故障重调度（30个场景）  (a) Fault rescheduling (30 scenarios)']; MS = {}
for name, get, seeds, epa in METHODS:
    a, sd = per_scen(get, seeds); ep = np.array([epa(s) for s in sids]) * 100; MS[name.split('\n')[0] + ('2' if '2 000' in name else '')] = (a, sd, ep)
    rows.append([name, pm(a['J']), pm(a['gap'], 1), f"{a['mm'].mean():.2f}", f"{a['feas'].mean() * 100:.0f}/{ep.mean():.0f}", tfmt(a['time'])])
# full-network scheduling under normal operation (120 laterals), 12 source heads
FN = collections.defaultdict(lambda: collections.defaultdict(list))
for r in json.load(open('res_fullnet.json')):
    FN[r['H0']][r['method']].append(r)
HS = sorted(FN); fbest = {H: min(x['J'] for x in FN[H]['SA']) for H in HS}
rows.append('（b）全网编组（正常工况，120条斗管，12个水头）  (b) Full-network scheduling (normal operation, 120 laterals, 12 heads)')
FNS = {}
for name, key in (('GNN-IL（本文）\nGNN-IL (ours)', 'GNN-IL'), ('GNN-IL\n（$\\delta$=0.5 m）', 'GNN-IL d0.5'), ('SA（800+30 000）', 'SA'), ('SA-fast\n（10+10 000）', 'SA-fast (10, 10000)'),
                  ('SA-fast\n（10+2 000）', 'SA-fast (10, 2000)'), ('GA', 'GA'), ('顺序编组\nSequential', 'Sequential')):
    a = {k: np.array([np.mean([fn(x) for x in FN[H][key]]) for H in HS]) for k, fn in (
        ('J', lambda x: x['J']), ('mm', lambda x: x['min_margin']), ('time', lambda x: x['time']), ('ng', lambda x: x['n_groups']),
        ('feas', lambda x: x['feasible'] and x['complete']), ('epa', lambda x: x['epa_feas_rate'] == 1))}
    a['gap'] = np.array([np.mean([(x['J'] - fbest[H]) / fbest[H] * 100 for x in FN[H][key]]) for H in HS]); FNS[key] = a
    rows.append([name, pm(a['J']), pm(a['gap'], 1), f"{a['mm'].mean():.2f}", f"{a['feas'].mean() * 100:.0f}/{a['epa'].mean() * 100:.0f}", tfmt(a['time'])])
fg, fs = FNS['GNN-IL'], FNS['SA']
S.update(FN_J_GNN=f"{fg['J'].mean():.2f}", FN_J_SA=f"{fs['J'].mean():.2f}", FN_N=len(HS), FN_WIN=int(np.sum(fg['J'] < fs['J'])), FN_DJ=f"{(1 - fg['J'].mean() / fs['J'].mean()) * 100:.1f}%", FN_GAP=f"{fg['gap'].mean():.1f}%",
         FN_T_GNN=f"{fg['time'].mean():.2f}", FN_T_SA=f"{fs['time'].mean():.1f}", FN_SPEEDUP=f"{fs['time'].mean() / fg['time'].mean():.0f}",
         FN_EPA_GNN=f"{fg['epa'].mean() * 100:.0f}%", FN_EPA_D=f"{FNS['GNN-IL d0.5']['epa'].mean() * 100:.0f}%", FN_EPA_SA=f"{fs['epa'].mean() * 100:.0f}%",
         FN_DJ_D=f"{(1 - FNS['GNN-IL d0.5']['J'].mean() / fs['J'].mean()) * 100:.1f}%", FN_NG_GNN=f"{fg['ng'].mean():.1f}", FN_NG_SA=f"{fs['ng'].mean():.1f}",
         FN_HWIN=f"{min(H for H in HS if all(fg['J'][i] < fs['J'][i] for i, h in enumerate(HS) if h >= H)):g}",
         FN_GAP_FAST=f"{FNS['SA-fast (10, 10000)']['gap'].mean():.1f}%", FN_GAP_GA=f"{FNS['GA']['gap'].mean():.1f}%", FN_GAP_SEQ=f"{FNS['Sequential']['gap'].mean():.1f}%",
         FN_GAP_SA=f"{fs['gap'].mean():.1f}%")
T['t2'] = dict(cap_cn='表2  训练分布内各方法的调度结果', cap_en='Table 2  Scheduling results of different methods within the training distribution',
               widths=[1350, 800, 800, 600, 700, 640],
               header=[['方法\nMethod', '$J$/m', '相对差距\nGap/%', '$\\Delta p_{\\min}$/m', '可行率\nFeasible/%', 'CPU耗时\nTime/s']], rows=rows,
               notes=['注：（a）30个场景，H₀为12~22 m，故障节点为随机田间节点；（b）H₀为12~23 m（间隔1 m），全部120条斗管参与编组。$J$与相对差距为场景间均值±标准差，其余为均值；SA、SA-fast和GA先对3个随机种子取平均；相对差距以3次SA中的最优解为基准；可行率为“评价器/EPANET”下压力约束全部满足的场景比例；$\\delta$为硬校核安全裕度；耗时为单个CPU线程。',
                      'Note: (a) 30 scenarios (H₀ 12-22 m, random field-node fault); (b) H₀ 12-23 m at 1 m steps, all 120 laterals scheduled. J and Gap are mean±std over scenarios, other columns are means; stochastic methods are first averaged over 3 seeds. Gap is relative to the best of 3 SA runs. Feasible: share of scenarios fully feasible under the evaluator/EPANET. $\\delta$: safety margin of the hard check. Time: single CPU thread.'])
g = MS['GNN-IL（本文）'][0]; sa = MS['SA（800+30 000）']; sf = MS['SA-fast']; sf2 = MS['SA-fast2']; ga = MS['GA']; sq = MS['顺序编组'][0]; old, _ = per_scen(lambda s, k: R[s]['res']['GNN-IL w/o gate'])
S.update(T_GNN=f"{g['time'].mean():.2f}", T_SA=f"{sa[0]['time'].mean():.1f}", SPEEDUP=f"{sa[0]['time'].mean() / g['time'].mean():.0f}",
         GAP_SA=f"{g['gap'].mean():.1f}%", GAP_FAST=f"{sf[0]['gap'].mean():.1f}%", GAP_FAST2=f"{sf2[0]['gap'].mean():.1f}%", GAP_SEQ=f"{sq['gap'].mean():.1f}%", GAP_GA=f"{ga[0]['gap'].mean():.1f}%",
         GAP_OLD=f"{old['gap'].mean():.1f}%", GAP_SA_SELF=f"{sa[0]['gap'].mean():.1f}%",
         T_FAST=f"{sf[0]['time'].mean():.2f}", T_FAST2=f"{sf2[0]['time'].mean():.2f}", T_GA=f"{ga[0]['time'].mean():.1f}", T_OLD=f"{old['time'].mean():.2f}",
         MM_GNN=f"{g['mm'].mean():.2f}", MM_SA=f"{sa[0]['mm'].mean():.2f}", MM_FAST=f"{sf[0]['mm'].mean():.2f}", MM_SEQ=f"{sq['mm'].mean():.2f}",
         NG_GNN=f"{g['ng'].mean():.1f}", NG_SA=f"{sa[0]['ng'].mean():.1f}", SEEDSTD_SA=f"{sa[1]:.3f}", SEEDSTD_GA=f"{ga[1]:.3f}", SEEDSTD_FAST=f"{sf[1]:.3f}",
         EPA_ID_GNN=f"{MS['GNN-IL（本文）'][2].mean():.0f}%", EPA_ID_SA=f"{sa[2].mean():.0f}%", EPA_ID_FAST=f"{sf[2].mean():.0f}%", EPA_ID_SEQ=f"{MS['顺序编组'][2].mean():.0f}%",
         WIN_FAST=f"{np.mean(g['gap'] < sf[0]['gap']) * 100:.0f}%", WIN_FAST2=f"{np.mean(g['gap'] < sf2[0]['gap']) * 100:.0f}%")

# hard-check safety margin delta (online setting, no retraining)
MG0 = json.load(open('res_margin.json'))['ID']
for d, tag in (('delta=0.2', 'D02'), ('delta=0.5', 'D')):
    v = [MG0[s][d] for s in sids]
    S.update({f'GAP_{tag}': f"{np.mean([(x['J'] - best[s]) / best[s] * 100 for x, s in zip(v, sids)]):.1f}%",
              f'EPA_ID_GNN_{tag}': f"{np.mean([x['epa_feas_rate'] == 1 for x in v]) * 100:.0f}%",
              f'DJ_{tag}': f"{np.mean([x['J'] / F[s]['GNN-IL+ (full)']['J'] - 1 for x, s in zip(v, sids)]) * 100:.1f}%",
              f'T_{tag}': f"{np.mean([x['time'] for x in v]):.2f}", f'MM_{tag}': f"{np.mean([x['min_margin'] for x in v]):.2f}"})

# ---------------------------------------------------------------- ablation
AB = [('完整模型 Full', 'GNN-IL+ (full)'), ('启用可行性门控 With gate', 'GNN-IL+ with gate'), ('无目标水平择优 w/o target level', 'GNN-IL+ w/o target-level'),
      ('无评价器择优 w/o ranking', 'GNN-IL+ w/o evaluator ranking'), ('无硬校核 w/o hard check', 'GNN-IL+ w/o hard check'),
      ('无组规模头(k=4) w/o size head', 'GNN-IL+ w/o size head (k=4)'), ('无路径先验 w/o path prior', 'GNN-IL+ w/o path prior'),
      ('GCN编码器 GCN encoder', 'GCN encoder'), ('MLP编码器 MLP encoder', 'MLP encoder (no graph)')]
rows = []; AS = {}
for name, key in AB:
    a, _ = per_scen(lambda s, k: F[s][key]); calls = np.array([F[s][key].get('eval_calls', np.nan) for s in sids]); AS[key] = (a, calls)
    rows.append([name.replace(' ', '\n', 1), pm(a['J']), pm(a['gap'], 1), f"{a['feas'].mean() * 100:.0f}", f"{a['time'].mean():.2f}", f"{np.nanmean(calls):.0f}"])
rows.append(['无锚点（独立打分解码）\nw/o anchor (token decoder)', pm(old['J']), pm(old['gap'], 1), f"{old['feas'].mean() * 100:.0f}", f"{old['time'].mean():.2f}", '—'])
T['t3'] = dict(cap_cn='表3  GNN-IL的消融实验结果', cap_en='Table 3  Ablation results of GNN-IL', widths=[1600, 800, 800, 600, 560, 530],
               header=[['配置\nConfiguration', '$J$/m', '相对差距\nGap/%', '可行率\nFeasible/%', 'CPU耗时\nTime/s', '校核次数\nChecks']], rows=rows,
               notes=['注：场景与表2相同；校核次数为单场景调用水力评价器的平均次数。', 'Note: Same scenarios as Table 2; Checks is the mean number of evaluator calls per scenario.'])
full = AS['GNN-IL+ (full)'][0]
for key, tag in (('GNN-IL+ w/o target-level', 'TL'), ('GNN-IL+ w/o evaluator ranking', 'RANK'), ('GNN-IL+ w/o hard check', 'HC'), ('GNN-IL+ w/o size head (k=4)', 'SIZE'),
                 ('GNN-IL+ w/o path prior', 'PRIOR'), ('GCN encoder', 'GCN'), ('MLP encoder (no graph)', 'MLP'), ('GNN-IL+ with gate', 'GATE')):
    a, c = AS[key]
    S[f'AB_{tag}_GAP'] = f"{a['gap'].mean():.1f}%"; S[f'AB_{tag}_FEAS'] = f"{a['feas'].mean() * 100:.0f}%"; S[f'AB_{tag}_T'] = f"{a['time'].mean():.2f}"; S[f'AB_{tag}_CALLS'] = f"{np.nanmean(c):.0f}"
S['AB_FULL_CALLS'] = f"{np.nanmean(AS['GNN-IL+ (full)'][1]):.0f}"

# ---------------------------------------------------------------- held-out test split
TS = json.load(open('res_testsplit.json'))
gap = np.array([(r['gnn']['J'] - r['teacher']['J']) / r['teacher']['J'] * 100 for r in TS])
S.update(N_TEST=len(TS), GAP_TEST=f"{gap.mean():.1f}%", GAP_TEST_MED=f"{np.median(gap):.1f}%", FEAS_TEST=f"{np.mean([r['gnn']['feasible'] for r in TS]) * 100:.0f}%",
         COMP_TEST=f"{np.mean([r['gnn']['complete'] for r in TS]) * 100:.0f}%", T_TEST=f"{np.mean([r['gnn']['time'] for r in TS]):.2f}",
         MM_TEST_GNN=f"{np.mean([r['gnn']['min_margin'] for r in TS]):.2f}", MM_TEST_SA=f"{np.mean([r['teacher']['min_margin'] for r in TS]):.2f}",
         BETTER_TEST=f"{np.mean(gap < 0) * 100:.0f}%")

# ---------------------------------------------------------------- OOD
O = json.load(open('res_ood.json')); OF = json.load(open('res_ood_final.json'))
SETN = [('OOD-low', '低水头外推\nLow head'), ('OOD-high', '高水头外推\nHigh head'), ('OOD-trunk', '干管故障\nMain-pipe fault'), ('Multi-fault', '多点并发故障\nMulti-node fault'), ('Fault+head drop', '故障+水头骤降\nFault+head drop'), ('Valve blockage', '阀门局部堵塞\nValve blockage')]
agg = collections.defaultdict(lambda: collections.defaultdict(list))
BL = json.load(open('res_blockage.json')); MG = json.load(open('res_margin.json'))
for r in O + BL:
    s = r['set'] if r['set'] != 'OOD-head' else ('OOD-low' if r['H0'] < 11.4 else 'OOD-high')
    f_ = OF.get(str(r['id']), {}) if r['set'] != 'Valve blockage' else r['res']
    n = r['n_remain']
    md = MG['BLK' if r['set'] == 'Valve blockage' else 'OOD'].get(str(r['id'])) or {}
    for m, v in (('GNN-IL+', f_.get('GNN-IL+')), ('GNN-IL+d', md.get('delta=0.5')), ('SA', r['res'].get('SA')), ('SA-fast', f_.get('SA-fast (10, 10000)')), ('Sequential', r['res'].get('Sequential'))):
        if v is None or n < 2:
            continue
        ok = bool(v['feasible'])
        served = (n - v.get('unserved', 0)) / n * 100 if ok and v['n_groups'] > 0 else 0.0
        if not m.startswith('GNN-IL+') and not (v['feasible'] and v['complete']):  # baselines: no partial service
            served = 0.0
        epa_ok = ok and v['n_groups'] > 0 and v['epa_feas_rate'] == 1
        agg[s][m].append(dict(served=served, epa=epa_ok, mm=v['epa_min_margin'] if v['n_groups'] > 0 else np.nan, J=v['J'], time=v['time']))
rows = []
for s, nm in SETN:
    for m, mn in (('GNN-IL+', 'GNN-IL\n$\\delta$=0'), ('GNN-IL+d', 'GNN-IL\n$\\delta$=0.5 m'), ('SA', 'SA'), ('SA-fast', 'SA-fast')):
        v = agg[s][m]
        if not v:
            continue
        J = [x['J'] for x in v if np.isfinite(x['J'])]; mm = [x['mm'] for x in v if np.isfinite(x['mm'])]
        rows.append([nm if m == 'GNN-IL+' else '', mn, f"{len(v)}", f"{np.mean([x['served'] for x in v]):.0f}", f"{np.mean([x['epa'] for x in v]) * 100:.0f}",
                     f"{np.mean(J):.2f}" if J else '—', f"{np.mean([x['time'] for x in v]):.2f}"])
T['t4'] = dict(cap_cn='表4  分布外与复杂工况场景下的调度结果', cap_en='Table 4  Scheduling results under out-of-distribution and complex scenarios',
               widths=[900, 820, 450, 650, 750, 600, 720],
               header=[['场景集\nScenario set', '方法\nMethod', '场景数\nN', '服务率\nServed/%', 'EPANET\n可行率/%\nFeasible/%', '$J$/m', '耗时\nTime/s']], rows=rows,
               notes=['注：模型未重新训练。服务率为获得可行灌水安排的斗管占故障后待灌斗管的比例，SA和SA-fast在无完整可行方案时记为0；EPANET可行率为全部轮灌组在EPANET复核下均满足压力约束的场景比例；$\\delta$为硬校核安全裕度。',
                      'Note: No retraining. Served: share of remaining laterals that receive a feasible group (0 for SA/SA-fast when no complete feasible schedule exists). EPANET feasible: share of scenarios in which every group satisfies the pressure constraint in EPANET; $\\delta$ is the safety margin of the hard check.'])
for s, _ in SETN:
    for m in ('GNN-IL+', 'GNN-IL+d', 'SA', 'SA-fast'):
        v = agg[s][m]
        if v:
            S[f'OOD_{s.replace("+", "_").replace(" ", "_").replace("-", "_")}_{m.replace("-", "_").replace("+", "P")}_EPA'] = f"{np.mean([x['epa'] for x in v]) * 100:.0f}%"
alld = [x for s, _ in SETN[1:] for x in agg[s]['GNN-IL+d']]; S['EPA_GNN_D'] = f"{np.mean([x['epa'] for x in alld]) * 100:.0f}%"
allg = [x for s, _ in SETN[1:] for x in agg[s]['GNN-IL+']]; alls = [x for s, _ in SETN[1:] for x in agg[s]['SA']]; allf = [x for s, _ in SETN[1:] for x in agg[s]['SA-fast']]
S.update(N_OOD=sum(len(agg[s]['GNN-IL+']) for s, _ in SETN), EPA_GNN=f"{np.mean([x['epa'] for x in allg]) * 100:.0f}%", EPA_SA=f"{np.mean([x['epa'] for x in alls]) * 100:.0f}%",
         EPA_FAST=f"{np.mean([x['epa'] for x in allf]) * 100:.0f}%", MMEPA_GNN=f"{np.nanmean([x['mm'] for x in allg]):.2f}", MMEPA_SA=f"{np.nanmean([x['mm'] for x in alls]):.2f}",
         SERVED_LOW_GNN=f"{np.mean([x['served'] for x in agg['OOD-low']['GNN-IL+']]):.0f}%", N_LOW=len(agg['OOD-low']['GNN-IL+']))

# ---------------------------------------------------------------- training / model size
d = pd.read_csv('models_anchor/train_log.csv'); v = d[d.split == 'val']
o = pd.read_csv('../models_gnn/train_log.csv'); o = o[o.split == 'val']
acc_new = v.partner_acc.iloc[int(np.argmin(v.loss.values))]
S.update(ACC_OLD=f"{o['token_acc@k'].iloc[-1] * 100:.1f}%", ACC_NEW=f"{acc_new * 100:.1f}%", ACC_E1=f"{v.partner_acc.iloc[0] * 100:.1f}%",
         ACC_GAIN=f"{(acc_new - o['token_acc@k'].iloc[-1]) * 100:.1f}", EP_BEST=int(v.epoch.iloc[int(np.argmin(v.loss.values))]))
for nm, path in (('GCN', 'models_anchor_gcn'), ('MLP', 'models_anchor_mlp'), ('NOPRIOR', 'models_anchor_nopair')):
    x = pd.read_csv(f'{path}/train_log.csv'); x = x[x.split == 'val']; S[f'ACC_{nm}'] = f"{x.partner_acc.iloc[int(np.argmin(x.loss.values))] * 100:.1f}%"

for t in T.values():  # variables italic in notes
    t['notes'] = [n.replace('H₀', '$H$₀') for n in t['notes']]
json.dump(T, open(B / 'tables.json', 'w'), ensure_ascii=False, indent=1)
json.dump(S, open(B / 'stats.json', 'w'), ensure_ascii=False, indent=1)
for k, v in S.items():
    print(k, v)
