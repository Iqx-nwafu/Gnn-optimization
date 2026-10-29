# experiments：锚点条件解码器 GNN-IL 与全部对比实验

本目录包含论文中 GNN-IL（锚点条件双头解码器）的训练、推理脚本，全部对比实验与消融，原始结果（`res_*.json`）、训练好的模型、论文表 2–4 和图 2、4–8 的生成脚本及数据。

所有命令都在本目录下运行（`cd experiments`）。脚本通过 `common.py` 引用仓库根目录的 `tree_evaluator.py`、`model_gnn.py`、`utils_data.py`、`dataset_v1/`、`models_gnn/`、`Nodes.xlsx`、`Pipes.xlsx`。

## 1 只复现表格与图件（约 1 min）
`res_*.json` 是论文所用的全部原始结果，可直接生成表格与图件：
```bash
python3 make_tables.py   # 表 2–4 与正文统计量 → tables/tables.json、tables/stats.json
python3 figs.py          # 图 2、4–8（PNG 600 dpi + TIF）→ figs/；各图数据 → figs/data/*.csv
```

## 2 用已训练模型推理
```python
from common import *                      # Evaluator, GNNPolicy, prefault_plan, make_state, schedule_metrics ...
from anchor_infer import AnchorPolicy
import bench_id

sc = bench_id.scen_list()[0]              # 一个分布内故障场景：H0、故障斗管、故障发生时刻
ev = Evaluator(sc['H0'])
plan = prefault_plan(GNNPolicy(), ev)     # 故障前的原轮灌计划
irr, blk, targets = make_state(ev, plan, sc['faults'], int(sc['frac'] * len(plan)))
groups, t = AnchorPolicy().schedule2(Evaluator(sc['H0']), irr, blk, gate=False, margin=0.5)
print(groups, t, schedule_metrics(ev, groups, targets))
```
`margin` 为硬校核安全裕度 δ（m），在线设定，无需重训；`gate=True` 启用可选的可行性门控。

## 3 完整复现
```bash
python3 anchor_train.py --threads 4 --out models_anchor              # GNN-IL（节点–管段融合编码器）
python3 anchor_train.py --encoder gcn  --out models_anchor_gcn       # 消融：GCN 编码器
python3 anchor_train.py --encoder mlp  --out models_anchor_mlp       # 消融：MLP 编码器
python3 anchor_train.py --no_pair      --out models_anchor_nopair    # 消融：去掉共享路径阻力先验
./run_all.sh                                                         # 全部评价 → res_*.json → 表格与图件
```
计时请在空闲机器上运行。所有方法均为单 CPU 线程（`torch.set_num_threads(1)`），场景分配到 4 个进程并行，各方法使用独立的评价缓存。

## 4 脚本
| 脚本 | 作用 | 输出 |
|---|---|---|
| `common.py` | 评价器封装、目标函数 J = 0.5F1 + 0.5F2、SA/GA/顺序编组基线、故障状态构造 | — |
| `anchor_train.py` | 训练锚点条件生成器；`--encoder gcn/mlp`、`--no_pair` 为消融 | `models_anchor*/` |
| `anchor_infer.py` | 推理：锚点（评价器在线计算的最不利出水口）→ 伙伴选择头 + 组规模头 → 水力硬校核（p ≥ Hmin + δ）→ 目标水平择优 | — |
| `bench_id.py` | 30 个分布内故障场景：无锚点解码、顺序编组、SA×3、GA×3 | `res_id.json` |
| `final_eval.py` | 同一批场景：GNN-IL 及消融、SA-fast×3 | `res_final.json` |
| `test_split_eval.py` | 按轮灌序列划分的测试集（268 个状态） | `res_testsplit.json` |
| `bench_ood.py` / `ood_final.py` | 分布外与复杂工况：水头外推、干管故障、多点故障、水头骤降 | `res_ood.json`、`res_ood_final.json` |
| `ood_blockage.py` | 阀门局部堵塞 | `res_blockage.json` |
| `epanet_check.py` / `epa_final.py` | EPANET 2.2（WNTR）独立复核 | `res_final_epa.json` |
| `pareto.py` / `pareto_gnn.py` | F1–F2 权重扫描与帕累托分析 | `res_pareto.json`、`res_pareto_gnn.json` |
| `margin_eval.py` | 安全裕度 δ = 0.2、0.5 m 在分布内、分布外和阀门堵塞场景上的效果（含 EPANET 复核） | `res_margin.json` |
| `full_net.py` | 正常工况 120 条斗管全网编组（H0 = 12~23 m）：GNN-IL（δ = 0、0.5 m）、SA×3、SA-fast×3、GA×3、顺序编组（含 EPANET 复核） | `res_fullnet.json` |
| `make_tables.py` / `figs.py` / `figstyle.py` | 表 2–4、图 2/4–8 及其数据 | `tables/`、`figs/` |

## 5 主要结果（单 CPU 线程）
| 任务 | 结果 |
|---|---|
| 故障重调度（30 个场景） | GNN-IL 0.23 s，为 SA（5.8 s）的 1/25；同为亚秒级预算时，与 SA 最优解的差距 14.2%，SA-fast 为 21.5%，顺序编组为 26.3% |
| 全网编组（120 条斗管，12 个水头） | GNN-IL 的 J 比 SA 低 16.3%（2.44 vs 2.92），12 个水头中 9 个优于 SA；耗时 0.53 s vs 11.4 s |
| EPANET 2.2 复核 | δ = 0.5 m 时分布内两类任务与 6 类分布外场景全部 100% 可行，故障重调度 J 仅变化 0.4% |
| 分布外（49 个场景，未重训） | 低水头工况 SA 无完整可行解，GNN-IL 为 91% 的斗管给出可行方案并上报其余斗管；其余场景与 SA 的 J 相差 0.26~0.39 m |
