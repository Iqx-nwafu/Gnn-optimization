# GNN-IL Stage 1: build dataset (normal + fault) for irrigation rotation scheduling

You confirmed two model improvements:
- node-edge fusion GNN encoder (node + pipe features)
- feasibility classifier gating module

This package is **Stage 1** only: build datasets needed to train those modules.

## Install
```bash
pip install -r requirements.txt
```

## 1) Build the main IL dataset

### Example (your Windows paths)
```bash
python build_gnn_il_dataset.py ^
  --nodes "E:\test\pythonProject\rotation_il\Nodes.xlsx" ^
  --pipes "E:\test\pythonProject\rotation_il\Pipes.xlsx" ^
  --root J0 --Hmin 11.59 ^
  --opt_root "E:\test\pythonProject\rotation_il\runs_opt_groupstats" ^
  --fault_root "E:\test\pythonProject\rotation_il\runs_fault_reopt_50perH0" ^
  --out_dir "E:\test\pythonProject\rotation_il\dataset_v1"
```

Notes:
- `--fault_root` is optional. If omitted, only normal schedules are used.
- The script will **skip** any schedule.json it cannot parse.

## 2) (Optional) Build feasibility-pair labels for the gating classifier
This will call `tree_evaluator.evaluate_group` to label random candidate groups as feasible/infeasible.

```bash
python build_feasibility_pairs.py ^
  --dataset_dir "E:\test\pythonProject\rotation_il\dataset_v1" ^
  --nodes "E:\test\pythonProject\rotation_il\Nodes.xlsx" ^
  --pipes "E:\test\pythonProject\rotation_il\Pipes.xlsx" ^
  --root J0 --Hmin 11.59 --q_lateral 0.012 ^
  --neg_per_pos 8 ^
  --max_states 30000
```

- `neg_per_pos`: # negative (random) groups sampled per state (teacher group is used as 1 positive)
- `max_states`: cap to control runtime (set to 0 to use all)

## 3) Quick verification
```bash
python verify_dataset.py --dataset_dir "E:\test\pythonProject\rotation_il\dataset_v1"
```

