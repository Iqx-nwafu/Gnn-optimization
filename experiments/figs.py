"""Revised figures (journal rules: bilingual axis titles 中文 English/unit and bilingual a./b. captions under every subplot,
no outer frame, inward ticks with ticks at both ends, B/W-distinguishable styles; one chart type per figure, shared Morandi palette)."""
import json, sys, collections
from pathlib import Path
import numpy as np, pandas as pd
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
from figstyle import *

OUT = Path('figs'); OUT.mkdir(exist_ok=True)
REPO = Path('..')
CA = {'GNN-IL+': C['GNN-IL+'], 'GCN': '#C08A9A', 'w/o prior': '#5FA08F', 'MLP': '#A07850', 'GNN-IL': C['GNN-IL']}
LAB = {'GNN-IL+': 'GNN-IL（本文）', 'SA': 'SA', 'SA-fast': 'SA-fast', 'GA': 'GA', 'Sequential': '顺序编组', 'GNN-IL': '无锚点解码'}
CAPS = []  # (fig, ax, cn, en) captions placed under each subplot at save time


def caption(ax, cn, en):
    CAPS.append((ax.figure, ax, cn, en))


def ends(ax, x=True, y=True, nb=6, ylo=None):
    """Put the first and last tick (with labels) exactly at the axis ends."""
    for on, get, setl, sett, axis in ((x, ax.get_xlim, ax.set_xlim, ax.set_xticks, ax.xaxis), (y, ax.get_ylim, ax.set_ylim, ax.set_yticks, ax.yaxis)):
        if not on:
            continue
        lo, hi = get()
        if axis is ax.yaxis and ylo is not None:
            lo = ylo
        if axis.get_scale() == 'log':
            lo, hi = 10 ** np.floor(np.log10(lo)), 10 ** np.ceil(np.log10(hi))
            setl(lo, hi); sett(10.0 ** np.arange(np.log10(lo), np.log10(hi) + 1)); continue
        t = MaxNLocator(nb).tick_values(lo, hi)
        t = t[(t >= t[t <= lo + 1e-12].max()) & (t <= t[t >= hi - 1e-12].min())]
        setl(t[0], t[-1]); sett(t)


def save(fig, name):
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    for f, ax, cn, en in CAPS:
        if f is not fig:
            continue
        bb = ax.get_tightbbox(r).transformed(fig.transFigure.inverted())
        pos = ax.get_position()
        fig.text((pos.x0 + pos.x1) / 2, bb.y0 - 0.006, f'{cn}\n{en}', ha='center', va='top', fontsize=7.5, linespacing=1.3)
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight'); fig.savefig(OUT / f'{name}.tif', bbox_inches='tight', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)


def fig2():
    """Network topology (3D line plot), redrawn with bilingual axis titles."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    nodes = pd.read_excel(REPO / 'Nodes.xlsx').set_index('Nodel ID'); pipes = pd.read_excel(REPO / 'Pipes.xlsx')
    fig = plt.figure(figsize=(8.4 * CM, 7.6 * CM)); ax = fig.add_subplot(111, projection='3d')
    cols = {c.lower(): c for c in pipes.columns}
    u_col = next(c for c in pipes.columns if str(c).lower().startswith(('start', 'from', 'u', '起'))) if 'u' not in cols else cols['u']
    v_col = next(c for c in pipes.columns if str(c).lower().startswith(('end', 'to', 'v', '终'))) if 'v' not in cols else cols['v']
    mat_col = next((c for c in pipes.columns if 'mat' in str(c).lower() or '材' in str(c)), None)
    for _, p in pipes.iterrows():
        a, b = nodes.loc[p[u_col]], nodes.loc[p[v_col]]
        main = (str(p[mat_col]).upper() == 'UPVC') if mat_col else (a.Y == 0 and b.Y == 0)
        ax.plot([a.X, b.X], [a.Y, b.Y], [a.Z, b.Z], '-' if main else '--', color=C['GNN-IL+'] if main else C['Sequential'], lw=1.6 if main else 0.9)
    from common import te
    isf = np.array([te.is_field_node_id(n) for n in nodes.index]); fld = nodes[isf]; trk = nodes[~isf & (nodes.index != 'J0')]
    ax.scatter(fld.X, fld.Y, fld.Z, s=5, marker='o', color=INK, depthshade=False)
    ax.scatter(trk.X, trk.Y, trk.Z, s=9, marker='s', color=C['GNN-IL+'], depthshade=False)
    ax.scatter([0], [0], [0], s=30, marker='^', color=C['SA'], edgecolor=INK, lw=0.4, depthshade=False)
    for n, r in nodes.iterrows():
        if n == 'J0' or (te.is_field_node_id(n) and n.endswith('6')) or not te.is_field_node_id(n):
            ax.text(r.X, r.Y, r.Z + 0.25, n, fontsize=4.5, color=MUTED)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False; pane.set_edgecolor('none')
    ax.grid(False)
    ax.set_xlabel(zh('$x$坐标 $x$-coordinate/m'), fontsize=6.5, labelpad=-2); ax.set_ylabel(zh('$y$坐标 $y$-coordinate/m'), fontsize=6.5, labelpad=-2)
    ax.set_zlabel('高程 Elevation/m', fontsize=6.5, labelpad=-4)
    ax.tick_params(labelsize=5.5, pad=-2); ax.view_init(elev=24, azim=-58)
    h = [plt.Line2D([], [], color=C['GNN-IL+'], lw=1.6, label='干管 Main pipe (UPVC)'), plt.Line2D([], [], color=C['Sequential'], ls='--', lw=0.9, label='支管 Branch pipe (PE)'),
         plt.Line2D([], [], color=C['GNN-IL+'], marker='s', ls='', ms=3, label='干管节点 Main node'), plt.Line2D([], [], color=INK, marker='o', ls='', ms=2.5, label='田间节点 Field node'),
         plt.Line2D([], [], color=C['SA'], marker='^', mec=INK, mew=0.4, ls='', ms=5, label='水源 Source (J0)')]
    ax.legend(handles=h, loc='upper left', fontsize=5.5, bbox_to_anchor=(-0.05, 1.02))
    save(fig, 'fig2_topology')


def fig4():
    """Training curves (line)."""
    fig, ax = plt.subplots(2, 1, figsize=(8.4 * CM, 11.6 * CM)); plt.subplots_adjust(hspace=0.62)
    d = pd.read_csv('models_anchor/train_log.csv')
    for sp, ls, mk, lab in (('train', '-', 'o', '训练集 Train'), ('val', '--', 's', '验证集 Validation')):
        x = d[d.split == sp]
        ax[0].plot(x.epoch, x.loss, ls, color=C['GNN-IL+'], lw=1.2, marker=mk, ms=3, markevery=5, label=lab)
    xmax = int(np.ceil(d.epoch.max() / 10) * 10)
    ax[0].set_xlabel('训练轮次 Epoch'); ax[0].set_ylabel('损失 Loss'); ax[0].legend(loc='upper right')
    ax[0].set_xlim(0, xmax); ends(ax[0], nb=5)
    runs = [('models_anchor', 'GNN-IL+', 'GNN-IL（本文）', 'o', '-'), ('models_anchor_nopair', 'w/o prior', '无路径先验', '^', '--'),
            ('models_anchor_gcn', 'GCN', 'GCN编码器', 'D', '-.'), ('models_anchor_mlp', 'MLP', 'MLP编码器（无图结构）', 'v', ':')]
    for path, key, lab, mk, ls in runs:
        x = pd.read_csv(f'{path}/train_log.csv'); x = x[x.split == 'val']
        ax[1].plot(x.epoch, x.partner_acc * 100, ls, color=CA[key], lw=1.1, marker=mk, ms=3, markevery=(2, 6), label=lab)
    o = pd.read_csv(REPO / 'models_gnn/train_log.csv'); o = o[o.split == 'val']
    y0 = o['token_acc@k'].iloc[-1] * 100
    ax[1].axhline(y0, color=CA['GNN-IL'], ls=(0, (6, 2, 1, 2, 1, 2)), lw=1.0, label='无锚点解码（训练200轮）')
    ax[1].set_xlabel('训练轮次 Epoch'); ax[1].set_ylabel('同组成员命中率 Hit rate/%')
    ax[1].set_xlim(0, xmax); ax[1].set_ylim(0, 60); ends(ax[1], nb=5)
    ax[1].legend(loc='center right', bbox_to_anchor=(1.0, 0.56), ncol=2, fontsize=6.3, columnspacing=0.8, handlelength=2.6)
    caption(ax[0], 'a. 损失', 'a. Loss'); caption(ax[1], 'b. 验证集同组成员命中率', 'b. Validation hit rate of group members')
    save(fig, 'fig4_training')


def fig5():
    """Held-out test split, by source-head bin (box)."""
    R = json.load(open('res_testsplit.json'))
    bins = [(11.4, 13), (13, 15), (15, 17), (17, 19), (19, 21), (21, 23.01)]
    names = ['11.4~13', '13~15', '15~17', '17~19', '19~21', '21~23']
    fig, ax = plt.subplots(2, 1, figsize=(8.4 * CM, 11.6 * CM)); plt.subplots_adjust(hspace=0.62)
    for a, key, ylab in ((ax[0], 'J', '综合目标值 Objective $J$/m'), (ax[1], 'min_margin', '最小压力裕度 Min. margin $\\Delta p_{\\min}$/m')):
        for who, src, off in (('SA', 'teacher', -0.18), ('GNN-IL+', 'gnn', 0.18)):
            data = [[r[src][key] for r in R if lo <= r['H0'] < hi and r[src]['feasible']] for lo, hi in bins]
            bp = a.boxplot(data, positions=np.arange(len(bins)) + off, widths=0.32, patch_artist=True, showfliers=True,
                           medianprops=dict(color=INK, lw=0.9), whiskerprops=dict(lw=0.7, color=INK), capprops=dict(lw=0.7, color=INK),
                           flierprops=dict(marker=MK[who], ms=2.5, mfc=C[who], mec='none', alpha=0.7))
            for b in bp['boxes']:
                b.set(facecolor=C[who], edgecolor=INK, lw=0.7, hatch=HATCH[who], alpha=0.9)
        a.set_xticks(range(len(bins))); a.set_xticklabels(names); a.set_xlim(-0.6, len(bins) - 0.4)
        a.set_xlabel(zh('水源水头 Source head $H_0$/m')); a.set_ylabel(zh(ylab)); ends(a, x=False, ylo=0)
    h = [Patch(facecolor=C['SA'], edgecolor=INK, hatch=HATCH['SA'], label='SA（教师方案）'), Patch(facecolor=C['GNN-IL+'], edgecolor=INK, label='GNN-IL（本文）')]
    ax[0].legend(handles=h, loc='upper left')
    caption(ax[0], 'a. 综合目标值', 'a. Objective value'); caption(ax[1], 'b. 最小压力裕度', 'b. Minimum pressure margin')
    save(fig, 'fig5_testsplit')


def _gap_id():
    MG = json.load(open('res_margin.json')); F = json.load(open('res_final.json')); R = {str(r['id']): r for r in json.load(open('res_id.json'))}
    rows = collections.defaultdict(list)
    for sid, r in R.items():
        best = min(r['res'][f'SA#{s}']['J'] for s in range(3))
        add = lambda m, v: rows[m].append(((v['J'] - best) / best * 100, v['time']))
        add('GNN-IL+', F[sid]['GNN-IL+ (full)']); add('Sequential', r['res']['Sequential']); add('GNN-IL+d', MG['ID'][sid]['delta=0.5'])
        for s in range(3):
            add('SA', r['res'][f'SA#{s}']); add('GA', r['res'][f'GA#{s}'])
            add('SA-fast (10, 2000)', F[sid][f'SA-fast (10, 2000)#{s}']); add('SA-fast (10, 10000)', F[sid][f'SA-fast (10, 10000)#{s}'])
    return rows


def _gap_ood():
    MG = json.load(open('res_margin.json')); R = json.load(open('res_ood.json')); F = json.load(open('res_ood_final.json'))
    rows = collections.defaultdict(list)
    for r in R:
        if r['set'] == 'OOD-head' and r['H0'] < 11.4:
            continue  # full service physically impossible; reported in Table 4
        f = F[str(r['id'])]; sa = r['res'].get('SA')
        if not f or sa is None or not sa['feasible']:
            continue
        for m, v in (('GNN-IL+', f['GNN-IL+']), ('GNN-IL+d', MG['OOD'][str(r['id'])]['delta=0.5']), ('SA-fast (10, 10000)', f['SA-fast (10, 10000)']), ('Sequential', r['res']['Sequential']), ('SA', sa)):
            if v['feasible'] and v['complete']:
                rows[m].append(((v['J'] - sa['J']) / sa['J'] * 100, v['time']))
    for r in json.load(open('res_blockage.json')):
        sa = r['res'].get('SA')
        if sa is None or not sa['feasible'] or not sa['complete']:
            continue
        for m, k in (('GNN-IL+', 'GNN-IL+'), ('GNN-IL+d', None), ('SA-fast (10, 10000)', 'SA-fast (10, 10000)'), ('Sequential', 'Sequential'), ('SA', 'SA')):
            v = r['res'][k] if k else MG['BLK'][str(r['id'])]['delta=0.5']
            if v['feasible'] and v['complete']:
                rows[m].append(((v['J'] - sa['J']) / sa['J'] * 100, v['time']))
    return rows


def _gap_fn():
    R = json.load(open('res_fullnet.json')); best = collections.defaultdict(lambda: np.inf)
    for r in R:
        if r['method'] == 'SA':
            best[r['H0']] = min(best[r['H0']], r['J'])
    key = {'GNN-IL': 'GNN-IL+', 'GNN-IL d0.5': 'GNN-IL+d'}
    rows = collections.defaultdict(list)
    for r in R:
        rows[key.get(r['method'], r['method'])].append(((r['J'] - best[r['H0']]) / best[r['H0']] * 100, r['time']))
    return rows


def fig6():
    """Quality-time trade-off (scatter, mean +- std over scenarios and seeds)."""
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(3, 1, figsize=(8.4 * CM, 18.5 * CM)); plt.subplots_adjust(hspace=0.72)
    style = {'GNN-IL+': ('GNN-IL+', zh(r'GNN-IL（本文，$\delta$=0）'), 'o', None), 'GNN-IL+d': ('GNN-IL+d', zh(r'GNN-IL（本文，$\delta$=0.5 m）'), 'h', None),
             'SA': ('SA', 'SA（800+30 000步）', 's', None), 'SA-fast (10, 10000)': ('SA-fast', 'SA-fast（10+10 000步）', '^', None),
             'SA-fast (10, 2000)': ('SA-fast', 'SA-fast（10+2 000步）', '^', 'white'), 'GA': ('GA', 'GA', 'D', None), 'Sequential': ('Sequential', '顺序编组', 'v', None)}
    for a, rows in ((ax[0], _gap_id()), (ax[1], _gap_fn()), (ax[2], _gap_ood())):
        for m, (key, lab, mk, face) in style.items():
            if m not in rows:
                continue
            g, t = np.array(rows[m]).T
            a.errorbar(t.mean(), g.mean(), yerr=g.std(), fmt=mk, ms=6, color=C[key], mfc=face or C[key], mec=C[key] if face else INK, mew=0.6,
                       ecolor=C[key], elinewidth=0.8, capsize=2, zorder=3)
        a.set_xscale('log'); a.set_xlim(1e-3, 1e2); a.axhline(0, color=MUTED, lw=0.6, ls=':', zorder=0)
        a.set_xlabel('CPU耗时 CPU time/s'); a.set_ylabel('与SA最优解的相对差距 Gap/%'); ends(a, nb=6)
    h = [Line2D([], [], ls='none', marker=mk, ms=6, color=C[key], mfc=face or C[key], mec=C[key] if face else INK, mew=0.6, label=lab) for key, lab, mk, face in style.values()]
    ax[0].legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, 1.04), ncol=2, fontsize=6.3, columnspacing=0.8, handletextpad=0.3)
    caption(ax[0], 'a. 故障重调度', 'a. Fault rescheduling'); caption(ax[1], 'b. 全网编组', 'b. Full-network scheduling')
    caption(ax[2], 'c. 分布外与复杂工况', 'c. Out-of-distribution and complex scenarios')
    save(fig, 'fig6_quality_time')


SETS = [('ID', '分布内'), ('OOD-head-high', '高水头\n外推'), ('OOD-trunk', '干管\n故障'), ('Multi-fault', '多点\n并发故障'), ('Fault+head drop', '故障+\n水头骤降'), ('Valve blockage', '阀门\n局部堵塞')]


def robustness_table():
    """Per scenario set: EPANET all-feasible rate and mean min margin, for GNN-IL+, SA, SA-fast, Sequential."""
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    E = json.load(open('res_final_epa.json'))
    for sid, e in E.items():
        for m, k in (('GNN-IL+', ['GNN-IL+ (full)']), ('SA', ['SA#0', 'SA#1', 'SA#2']), ('SA-fast', [f'SA-fast (10, 10000)#{s}' for s in range(3)]), ('Sequential', ['Sequential'])):
            out['ID'][m].append((np.mean([e[x]['allok'] for x in k]), np.mean([e[x]['mm'] for x in k])))
    R = json.load(open('res_ood.json')); F = json.load(open('res_ood_final.json'))
    for r in json.load(open('res_blockage.json')):
        for m, k in (('GNN-IL+', 'GNN-IL+'), ('SA', 'SA'), ('SA-fast', 'SA-fast (10, 10000)'), ('Sequential', 'Sequential')):
            v = r['res'].get(k)
            if v is not None:
                out['Valve blockage'][m].append((bool(v['feasible'] and v['complete'] and v['epa_feas_rate'] == 1), v['epa_min_margin']))
    M = json.load(open('res_margin.json'))  # GNN-IL with hard-check safety margin delta = 0.5 m
    for sid, o in M['ID'].items():
        out['ID']['GNN-IL+d'].append((o['delta=0.5']['epa_feas_rate'] == 1, o['delta=0.5']['epa_min_margin']))
    for sid, o in M['BLK'].items():
        if o:
            v = o['delta=0.5']; out['Valve blockage']['GNN-IL+d'].append((bool(v['feasible'] and v['complete'] and v['epa_feas_rate'] == 1), v['epa_min_margin']))
    for r in R:
        o = M['OOD'].get(str(r['id']))
        if o and not (r['set'] == 'OOD-head' and r['H0'] < 11.4):
            v = o['delta=0.5']; s = 'OOD-head-high' if r['set'] == 'OOD-head' else r['set']
            out[s]['GNN-IL+d'].append((bool(v['feasible'] and v['complete'] and v['epa_feas_rate'] == 1), v['epa_min_margin']))
    for r in R:
        s = r['set']
        if s == 'OOD-head':
            if r['H0'] < 11.4:
                continue
            s = 'OOD-head-high'
        f = F[str(r['id'])]
        for m, v in (('GNN-IL+', f.get('GNN-IL+')), ('SA', r['res'].get('SA')), ('SA-fast', f.get('SA-fast (10, 10000)')), ('Sequential', r['res'].get('Sequential'))):
            if v is None:
                continue
            out[s][m].append((bool(v['feasible'] and v['complete'] and v['epa_feas_rate'] == 1), v['epa_min_margin']))
    return out


def fig7():
    """EPANET robustness by scenario set (grouped bars)."""
    T = robustness_table(); ms = ['GNN-IL+', 'GNN-IL+d', 'SA', 'SA-fast', 'Sequential']
    fig, ax = plt.subplots(2, 1, figsize=(8.4 * CM, 11.6 * CM)); plt.subplots_adjust(hspace=0.68)
    w = 0.16; x = np.arange(len(SETS))
    for j, m in enumerate(ms):
        r = [np.mean([o for o, _ in T[s][m]]) * 100 if T[s][m] else np.nan for s, _ in SETS]
        mm = [np.nanmean([v for _, v in T[s][m]]) if T[s][m] else np.nan for s, _ in SETS]
        lab = {'SA-fast': 'SA-fast（10+10 000步）', 'Sequential': '顺序编组', 'GNN-IL+': zh(r'GNN-IL（本文，$\delta$=0）'), 'GNN-IL+d': zh(r'GNN-IL（本文，$\delta$=0.5 m）'), 'SA': 'SA'}[m]
        for a, y in ((ax[0], r), (ax[1], mm)):
            a.bar(x + (j - 2) * w, y, w * 0.92, color=C[m], hatch=HATCH[m], edgecolor=INK, lw=0.3, label=lab)
    for a in ax:
        a.set_xticks(x); a.set_xticklabels([n for _, n in SETS], fontsize=6.3); a.set_xlim(-0.55, len(SETS) - 0.45)
        a.set_xlabel('场景集 Scenario set')
    ax[0].set_ylabel('EPANET全可行率 Feasible rate/%'); ax[0].set_ylim(0, 100); ax[0].set_yticks(np.arange(0, 101, 20))
    ax[1].set_ylabel('EPANET最小压力裕度 Min. margin/m'); ax[1].axhline(0, color=INK, lw=0.5); ends(ax[1], x=False)
    ax[0].legend(loc='lower center', ncol=2, fontsize=6.3, bbox_to_anchor=(0.5, 1.04), columnspacing=1.0)
    caption(ax[0], 'a. 全可行率', 'a. Fully feasible rate'); caption(ax[1], 'b. 最小压力裕度', 'b. Minimum pressure margin')
    save(fig, 'fig7_robustness')


def fig8():
    """mu-sigma trade-off: SA weight-sweep front (shaded dominated area) with method points."""
    P = json.load(open('res_pareto.json')); G = json.load(open('res_pareto_gnn.json'))
    fig, ax = plt.subplots(3, 1, figsize=(8.4 * CM, 17.5 * CM)); plt.subplots_adjust(hspace=0.72)
    for a, H, lab in zip(ax, (13.0, 17.0, 21.0), 'abc'):
        pts = np.array([(p['mu'], p['sigma']) for p in P if p['H0'] == H])
        nd = np.array([p for p in pts if not any((q[0] <= p[0]) and (q[1] <= p[1]) and (q != p).any() for q in pts)])
        nd = nd[np.argsort(nd[:, 0])]
        allx = np.r_[pts[:, 0], [o['mu'] for o in G if o['H0'] == H]]; ally = np.r_[pts[:, 1], [o['sigma'] for o in G if o['H0'] == H]]
        a.set_xlim(allx.min() - 0.05 * np.ptp(allx), allx.max() + 0.05 * np.ptp(allx)); a.set_ylim(0, ally.max() * 1.1); ends(a, nb=5)
        xmax, ymax = a.get_xlim()[1], a.get_ylim()[1]
        sx = np.r_[np.repeat(nd[:, 0], 2)[1:], xmax]; sy = np.r_[np.repeat(nd[:, 1], 2)[:-1], nd[-1, 1]]
        a.fill_between(sx, sy, ymax, color=C['SA'], alpha=0.16, lw=0, hatch='////', edgecolor=C['SA'], label='SA前沿支配区')
        a.plot(sx, sy, '-', color=C['SA'], lw=1.2, label='SA帕累托前沿（权重扫描）')
        a.scatter(pts[:, 0], pts[:, 1], s=12, color=C['SA'], marker=MK['SA'], lw=0, zorder=3)
        for m in ('GNN-IL+', 'GA', 'Sequential'):
            q = np.array([(o['mu'], o['sigma']) for o in G if o['H0'] == H and o['method'] == m])
            a.scatter(q[:, 0], q[:, 1], s=36 if m == 'GNN-IL+' else 24, color=C[m], marker=MK[m], edgecolor='white' if m != 'GNN-IL+' else INK, lw=0.6, zorder=4, label=LAB[m])
        a.set_xlabel(zh('富余水头均值 Mean surplus $F_1$/m')); a.set_ylabel(zh('富余水头标准差 Std $F_2$/m'))
        caption(a, zh(f'{lab}. 水源水头 $H_0$={H:g} m'), f'{lab}. Source head $H_0$={H:g} m')
    ax[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.04), ncol=2, fontsize=6.3, columnspacing=1.0)
    save(fig, 'fig8_pareto')


def export_csv():
    """Plot data for each figure, so the figures can be restyled elsewhere (e.g. Origin)."""
    D = OUT / 'data'; D.mkdir(exist_ok=True)
    nodes = pd.read_excel(REPO / 'Nodes.xlsx'); nodes.to_csv(D / 'fig2_nodes.csv', index=False, encoding='utf-8-sig')
    pd.read_excel(REPO / 'Pipes.xlsx').to_csv(D / 'fig2_pipes.csv', index=False, encoding='utf-8-sig')
    logs = []
    for path, name in (('models_anchor', 'GNN-IL'), ('models_anchor_nopair', 'w/o path prior'), ('models_anchor_gcn', 'GCN encoder'), ('models_anchor_mlp', 'MLP encoder')):
        x = pd.read_csv(f'{path}/train_log.csv'); x.insert(0, 'model', name); logs.append(x)
    pd.concat(logs).to_csv(D / 'fig4_training_logs.csv', index=False, encoding='utf-8-sig')
    o = pd.read_csv(REPO / 'models_gnn/train_log.csv'); o.to_csv(D / 'fig4_token_decoder_log.csv', index=False, encoding='utf-8-sig')
    R = json.load(open('res_testsplit.json'))
    pd.DataFrame([dict(H0=r['H0'], J_SA=r['teacher']['J'], J_GNN=r['gnn']['J'], margin_SA=r['teacher']['min_margin'], margin_GNN=r['gnn']['min_margin']) for r in R]).to_csv(D / 'fig5_testsplit.csv', index=False, encoding='utf-8-sig')
    rows = []
    for panel, d in (('a', _gap_id()), ('b', _gap_fn()), ('c', _gap_ood())):
        for m, v in d.items():
            g, t = np.array(v).T
            rows.append(dict(panel=panel, method=m, n=len(g), time_mean_s=t.mean(), gap_mean_pct=g.mean(), gap_std_pct=g.std()))
    pd.DataFrame(rows).to_csv(D / 'fig6_quality_time.csv', index=False, encoding='utf-8-sig')
    T = robustness_table()
    pd.DataFrame([dict(set=s, method=m, n=len(T[s][m]), feasible_rate_pct=np.mean([o for o, _ in T[s][m]]) * 100, min_margin_m=np.nanmean([v for _, v in T[s][m]]))
                  for s, _ in SETS for m in ('GNN-IL+', 'GNN-IL+d', 'SA', 'SA-fast', 'Sequential') if T[s][m]]).to_csv(D / 'fig7_robustness.csv', index=False, encoding='utf-8-sig')
    P = json.load(open('res_pareto.json')); G = json.load(open('res_pareto_gnn.json'))
    pd.DataFrame([dict(method='SA', H0=p['H0'], w_mean=p['w_mean'], seed=p['seed'], F1=p['mu'], F2=p['sigma']) for p in P] +
                 [dict(method=o['method'], H0=o['H0'], F1=o['mu'], F2=o['sigma']) for o in G]).to_csv(D / 'fig8_pareto.csv', index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    for f in sys.argv[1:] or ['fig2', 'fig4', 'fig5', 'fig6', 'fig7', 'fig8', 'export_csv']:
        globals()[f]()
