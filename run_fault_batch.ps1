# run_fault_batch.ps1
# Batch: for H0 = 11.0..23.0 step 0.2, sample 50 random (but reproducible) fault scenarios per H0,
# using the pre-fault schedule from runs_opt_groupstats\H0_XX.XX\schedule.json

$PROJ = "E:\test\pythonProject\rotation_il"
$NODES = Join-Path $PROJ "Nodes.xlsx"
$PIPES = Join-Path $PROJ "Pipes.xlsx"

$OPT_ROOT = Join-Path $PROJ "runs_opt_groupstats"
$OUT_ROOT = Join-Path $PROJ "runs_fault_reopt_50perH0"

# Make sure the script exists in your project folder
$SCRIPT = Join-Path $PROJ "simulate_fault_reopt_from_opt.py"

# Global params
$ROOT_NODE = "J0"
$HMIN = 11.59
$Q_LATERAL = 0.012

# Post-fault optimizer params (tune if needed)
$N_INIT = 800
$SA_STEPS = 30000
$MAX_TRIES_PER_GROUP = 600
$W_MEAN = 0.5
$W_STD  = 0.5

# Turn non-zero exit codes into a visible failure but continue looping
function Run-One($argsArray) {
  & python @argsArray
  return $LASTEXITCODE
}

# H0 list: 11.0..23.0 step 0.2  (k/5)
$kStart = 55
$kEnd = 115

for ($k = $kStart; $k -le $kEnd; $k++) {
  $H0 = [double]$k / 5.0
  $H0Tag = "{0:F2}" -f $H0

  $prefaultDir = Join-Path $OPT_ROOT ("H0_{0}" -f $H0Tag)
  $prefaultJson = Join-Path $prefaultDir "schedule.json"

  if (!(Test-Path $prefaultJson)) {
    Write-Host "[SKIP] Missing pre-fault schedule.json for H0=$H0Tag at $prefaultJson"
    continue
  }

  for ($s = 1; $s -le 50; $s++) {
    $scenarioTag = "S{0:D3}" -f $s
    $outDir = Join-Path (Join-Path $OUT_ROOT ("H0_{0}" -f $H0Tag)) $scenarioTag

    $doneFlag = Join-Path $outDir "scenario_summary.csv"
    if (Test-Path $doneFlag) {
      Write-Host "[SKIP] Done H0=$H0Tag $scenarioTag"
      continue
    }

    # Reproducible seed: deterministic from (H0 index, scenario index)
    $hIdx = $k - $kStart   # 0..60
    $seed = 1000000 + $hIdx * 1000 + $s

    $args = @(
      $SCRIPT,
      "--nodes", $NODES,
      "--pipes", $PIPES,
      "--root", $ROOT_NODE,
      "--H0", $H0,
      "--Hmin", $HMIN,
      "--q_lateral", $Q_LATERAL,
      "--prefault_schedule_json", $prefaultJson,
      "--out", $outDir,
      "--seed", $seed,
      "--n_init", $N_INIT,
      "--sa_steps", $SA_STEPS,
      "--max_tries_per_group", $MAX_TRIES_PER_GROUP,
      "--w_mean", $W_MEAN,
      "--w_std", $W_STD
    )

    Write-Host "[RUN ] H0=$H0Tag $scenarioTag seed=$seed"
    $code = Run-One $args
    if ($code -ne 0) {
      Write-Host "[FAIL] H0=$H0Tag $scenarioTag exit=$code"
      continue
    }
  }
}

Write-Host "Batch finished."
