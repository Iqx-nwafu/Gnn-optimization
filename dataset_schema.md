# Dataset v1 (GNN-IL) schema

This dataset is for **sequential group generation** (2–4 laterals per group) under:
- varying source head `H0`
- optional random faults (node + time), modeled as downstream subtree cut
- **hard feasibility constraint** (pressure head >= Hmin) checked by `tree_evaluator.py`

Two model improvements planned (you confirmed):
1) **Node-edge fusion** GNN encoder (node + pipe features)
2) **Feasibility classifier** as a constraint/gating module (filters infeasible candidate groups)

This package provides **Stage 1** only: dataset construction (and optional feasibility-pair labels).

---

## Files written by `build_gnn_il_dataset.py`

Output folder layout (example: `dataset_v1/`):

- `meta.json`
  - mappings: `node_ids`, `edge_ids`, `lateral_ids`
  - `lateral_to_node_idx`
  - `side_idx` (0 for L, 1 for R)
  - feature descriptions

- `graph_tensors.pt`
  - `node_x`: FloatTensor [N, Fn]
  - `edge_index`: LongTensor [2, 2E] (bidirectional)
  - `edge_attr`: FloatTensor [2E, Fe]
  - `lateral_to_node_idx`: LongTensor [L]
  - `side_idx`: LongTensor [L]

- `samples.pt`
  - `H0`: FloatTensor [S]
  - `irrigated`: UInt8Tensor [S, L]  (1=already irrigated)
  - `reachable`: UInt8Tensor [S, L]  (1=still reachable; 0=blocked by fault)
  - `target`: UInt8Tensor [S, L]     (multi-hot of next group laterals)
  - `target_size`: LongTensor [S]    (2/3/4)
  - `scenario_type`: UInt8Tensor [S] (0=normal, 1=fault-post)
  - `fault_node_idx`: LongTensor [S] (-1 if normal)
  - `step_idx`: LongTensor [S]       (step within the sequence being generated)
  - `source`: list[str] (python list, same length S) provenance path to schedule.json

> For fault scenarios, samples represent the **post-fault rescheduling phase** only.
> Initial masks are set from the schedule.json:
> - irrigated = `fault_effect.irrigated_before_fault`
> - reachable = all ones except `fault_effect.blocked_laterals` set to 0

---

## Optional feasibility-pair dataset (`build_feasibility_pairs.py`)

Produces: `feas_pairs.pt`

- `state_idx`: LongTensor [P]   (index into `samples.pt`)
- `cand_pad`: LongTensor [P, 4] (candidate laterals padded with -1)
- `cand_size`: LongTensor [P]   (2/3/4)
- `label_ok`: UInt8Tensor [P]   (1 if feasible, else 0)
- `min_margin`: FloatTensor [P] (min(pressure - Hmin) from evaluator)

This supports training a **binary feasibility classifier** (the gating module).

---

## What counts as "teacher" schedule.json

### Normal schedules
Your offline optimizer `rotation_optimize_groupstats.py` writes:

- top-level key: `groups` (list)
- each group has: `laterals` (list of 2–4)

### Fault schedules
Your `simulate_fault_reopt.py` writes:

- `fault_effect.irrigated_before_fault`
- `fault_effect.blocked_laterals`
- `postfault_reopt.groups` (list of groups, laterals 2–4)

This builder supports both formats.

