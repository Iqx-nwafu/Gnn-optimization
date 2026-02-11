# GNN-IL Stage 2: train generator (node-edge fusion) + train feasibility gating + inference with hard hydraulic check

You accepted:
- Next-group target is **multi-hot** over 120 laterals (BCE loss)
- Group size (2/3/4) is a **classification** auxiliary head (CE loss)
- Two method improvements:
  1) **Node-edge fusion** GNN encoder (nodes + pipes)
  2) **Feasibility classifier** gating module (binary feasible/infeasible)

This package assumes you already built `dataset_v1/` using Stage 1 scripts:
- `dataset_v1/meta.json`
- `dataset_v1/graph_tensors.pt`
- `dataset_v1/samples.pt`
- (optional) `dataset_v1/feas_pairs.pt` for feasibility classifier training

`tree_evaluator.py` must be importable (same folder or PYTHONPATH) for inference hard-check.

---

## 0) Install
```bash
pip install -r requirements.txt
```

---

## 1) Train the group generator (GNN + multi-task heads)
```bash
python train_gnn_generator.py ^
  --dataset_dir "E:\test\pythonProject\rotation_il\dataset_v1" ^
  --out_dir "E:\test\pythonProject\rotation_il\models_gnn" ^
  --epochs 60 --batch 256 --lr 3e-4 --val_ratio 0.15 ^
  --lambda_size 0.2
```

Outputs:
- `models_gnn/generator.pt`  (checkpoint + meta)
- `models_gnn/train_log.csv`

---

## 2) (Optional) Train the feasibility gating classifier
Requires `dataset_v1/feas_pairs.pt` from Stage 1.

```bash
python train_feasibility_classifier.py ^
  --dataset_dir "E:\test\pythonProject\rotation_il\dataset_v1" ^
  --out_dir "E:\test\pythonProject\rotation_il\models_gnn" ^
  --epochs 12 --batch 512 --lr 3e-4 --val_ratio 0.15
```

Outputs:
- `models_gnn/feasibility.pt`
- `models_gnn/feas_train_log.csv`

---

## 3) Inference (generate schedule for a given H0), with feasibility gating + evaluator hard-check + repair
```bash
python infer_gnn_generator.py ^
  --dataset_dir "E:\test\pythonProject\rotation_il\dataset_v1" ^
  --generator_ckpt "E:\test\pythonProject\rotation_il\models_gnn\generator.pt" ^
  --nodes "E:\test\pythonProject\rotation_il\Nodes.xlsx" ^
  --pipes "E:\test\pythonProject\rotation_il\Pipes.xlsx" ^
  --root J0 --Hmin 11.59 --q_lateral 0.012 ^
  --H0 17.8 ^
  --out "E:\test\pythonProject\rotation_il\infer_gnn_H0_17.8" ^
  --feas_ckpt "E:\test\pythonProject\rotation_il\models_gnn\feasibility.pt"
```

If you omit `--feas_ckpt`, it will still work (hard-check only), just slower.

---

## Notes on decoding
We do NOT trust the network blindly:
1) Use generator logits to get top-K candidates (default K=14).
2) Enumerate combinations of size 2/3/4 among top-K.
3) Optionally filter/sort by feasibility classifier probability.
4) Final decision uses `tree_evaluator.evaluate_group()` as **hard constraint**.
5) If none feasible, expand K and/or random sample until a feasible group is found; otherwise stop.

