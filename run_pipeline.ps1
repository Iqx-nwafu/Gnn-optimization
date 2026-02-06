# run_pipeline.ps1
# One-click pipeline (Windows PowerShell)
# Adjust paths to your local files.

$nodes = "Nodes.xlsx"
$pipes = "Pipes.xlsx"
$root  = "J0"
$Hmin  = 11.59
$qLat  = 0.012

$outOpt = "runs_opt_groupstats"
$outDS  = "dataset.pt"
$outPol = "policy.pt"

# 1) Offline near-optimal solutions
python rotation_optimize_groupstats.py `
  --nodes "$nodes" --pipes "$pipes" --root $root `
  --Hmin $Hmin --q_lateral $qLat `
  --H0_list 15 16 17 18 19 20 21 22 23 `
  --out "$outOpt" `
  --seed 123 --n_init 1500 --sa_steps 40000 `
  --w_mean 0.5 --w_std 0.5

# 2) Build IL dataset
python build_imitation_dataset.py `
  --nodes "$nodes" --pipes "$pipes" `
  --opt_root "$outOpt" `
  --out "$outDS"

# 3) Train policy
python train_group_policy.py `
  --dataset "$outDS" --out "$outPol" `
  --epochs 60 --batch 256 --lr 3e-4 --val_ratio 0.15

# 4) Inference example
python infer_group_policy.py `
  --policy "$outPol" `
  --nodes "$nodes" --pipes "$pipes" --root $root `
  --H0 17.8 --Hmin $Hmin --q_lateral $qLat `
  --out "infer_H0_17.8"
