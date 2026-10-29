# GNN-IL：基于物理感知图神经网络的轮灌组实时调度

本仓库是论文《基于物理感知图神经网络的轮灌组实时调度方法》的代码、数据、模型和实验结果。研究对象为一个树状自压滴灌管网（71 个节点、70 段管道、120 条斗管）。当斗管发生故障时，方法在亚秒级时间内为剩余斗管重新生成满足水力约束的轮灌组。

方法由三部分组成：
- **节点–管段融合图编码器**：同时编码节点（高程、状态）和管段（长度、管径、摩阻）特征；
- **锚点条件双头解码器**：以评价器在线计算的最不利出水口为锚点，由伙伴选择头（带共享路径阻力先验）和组规模头生成轮灌组；
- **水力硬校核与目标水平择优**：用树状管网评价器（GB/T 50485—2020 摩阻公式）校核 p ≥ Hmin + δ，安全裕度 δ 可在线设定，无需重训。

单 CPU 线程下，GNN-IL 的故障重调度耗时 0.23 s，是模拟退火（SA）的 1/25；120 条斗管全网编组的目标值比 SA 低 16.3%，耗时为其 1/21；δ = 0.5 m 时全部方案通过 EPANET 2.2 独立复核。

## 目录
| 路径 | 内容 |
|---|---|
| `experiments/` | **论文方法与全部实验**：锚点条件解码器的训练与推理、SA/GA/顺序编组基线、消融、分布外与阀门堵塞工况、EPANET 复核、帕累托分析、全网编组；原始结果 `res_*.json`、模型 `models_anchor*/`、表格与图件。见 [experiments/README.md](experiments/README.md) |
| `tree_evaluator.py` | 树状管网水力评价器（GB/T 50485—2020 摩阻与局部损失，节点压力与压力裕度） |
| `model_gnn.py` | 图编码器（融合编码器、GCN、MLP）与生成器、可行性分类器 |
| `utils_data.py` | 图张量与数据集读取 |
| `Nodes.xlsx`、`Pipes.xlsx` | 管网节点与管段参数 |
| `dataset_v1/` | 模仿学习数据集（正常与故障工况下的教师轮灌方案、可行性样本对） |
| `models_gnn/` | 逐组令牌解码器（原始 GNN-IL 生成器与可行性分类器），实验中用于生成故障前的轮灌计划，并作为“无锚点解码”消融 |
| `gnn_il_stage2_train_infer/` | 逐组令牌解码器的训练与推理脚本 |
| `build_*.py`、`rotation_optimize_groupstats.py`、`simulate_fault_reopt*.py` 等 | 教师方案生成（SA 优化与故障重优化）与数据集构建 |

## 快速开始
```bash
pip install -r requirements.txt
cd experiments
python3 make_tables.py && python3 figs.py   # 由 res_*.json 复现论文表 2–4、图 2/4–8
```
推理示例和完整复现步骤见 [experiments/README.md](experiments/README.md)。

## 数据集构建（Stage 1）
以下步骤从教师方案生成 `dataset_v1/`。仓库中已附带构建好的数据集，复现实验时无需重跑。

### 1) 主数据集
```bash
python build_gnn_il_dataset.py \
  --nodes Nodes.xlsx --pipes Pipes.xlsx \
  --root J0 --Hmin 11.59 \
  --opt_root runs_opt_groupstats \
  --fault_root runs_fault_reopt_50perH0 \
  --out_dir dataset_v1
```
- `--opt_root`：`rotation_optimize_groupstats.py` 输出的正常工况 SA 方案目录；
- `--fault_root`：`simulate_fault_reopt*.py` 输出的故障重优化方案目录（可选，缺省时只用正常工况方案）；
- 无法解析的 `schedule.json` 会被跳过。

### 2) 可行性分类器的样本对（可选）
调用 `tree_evaluator.evaluate_group` 为随机候选组标注可行/不可行：
```bash
python build_feasibility_pairs.py \
  --dataset_dir dataset_v1 --nodes Nodes.xlsx --pipes Pipes.xlsx \
  --root J0 --Hmin 11.59 --q_lateral 0.012 \
  --neg_per_pos 8 --max_states 30000
```
- `neg_per_pos`：每个状态采样的负样本数（教师组为 1 个正样本）；
- `max_states`：状态数上限（0 表示全部）。

### 3) 校验
```bash
python verify_dataset.py --dataset_dir dataset_v1
```
数据格式见 [dataset_schema.md](dataset_schema.md)。
