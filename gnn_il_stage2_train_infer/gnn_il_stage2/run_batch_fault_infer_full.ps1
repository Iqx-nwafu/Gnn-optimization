<#
.SYNOPSIS
    批量故障推理自动化脚本 (Windows PowerShell 版)
    
.DESCRIPTION
    1. 解析 Nodes.xlsx 获取合规故障节点 (J11+)。
    2. 遍历 H0 (11.0 -> 23.0)。
    3. 生成确定性随机种子。
    4. 调用 Python 脚本执行单场景推理。
    5. 汇总结果。

.NOTES
    Author: Top-tier Programmer
    Date: 2026-02-10
#>

$ErrorActionPreference = "Stop"

# ================= 配置区 =================
$ProjectRoot   = "E:\test\pythonProject\rotation_il"
$DatasetDir    = Join-Path $ProjectRoot "dataset_v1"
$NodesXlsx     = Join-Path $ProjectRoot "Nodes.xlsx"
$PipesXlsx     = Join-Path $ProjectRoot "Pipes.xlsx"

# 推理脚本路径
$InferScript   = Join-Path $ProjectRoot "infer_gnn_generator_fault.py"
$CollectScript = Join-Path $ProjectRoot "collect_fault_summary_all.py"

# 模型路径
$GenCkpt       = Join-Path $ProjectRoot "models_gnn\generator.pt"
$FeasCkpt      = Join-Path $ProjectRoot "models_gnn\feasibility_v2.pt"
$FeasThreshold = 0.2

# 输出路径
$OutRoot       = Join-Path $ProjectRoot "runs_fault_infer_full"

# 实验参数
$BaseSeed      = 123
$ScenesPerH0   = 50
$PrefaultForceK = 4    # 故障前固定 4 斗/组

# 环境变量设置 (防止 Numpy/PyTorch 过度占用 CPU)
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
$env:NODES_XLSX = $NodesXlsx

# ================= 初始化 =================
if (-not (Test-Path $ProjectRoot)) { Write-Error "Project root not found: $ProjectRoot"; exit 1 }
Set-Location $ProjectRoot
if (-not (Test-Path $OutRoot)) { New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null }

Write-Host "[INFO] Starting Batch Inference Process..." -ForegroundColor Cyan

# ================= 解析 Nodes.xlsx =================
Write-Host "[INFO] Parsing Nodes.xlsx for eligible fault nodes..."
# PowerShell 必须使用临时文件或 -c 方式来执行多行 Python 代码
$PyCode = @"
import os, json, sys
try:
    import tree_evaluator as te
except ImportError:
    # 假设 tree_evaluator 在当前目录或 pythonpath 中
    sys.path.append('.')
    import tree_evaluator as te

try:
    nodes = te.load_nodes_xlsx(os.environ["NODES_XLSX"])
    ids = []
    for nid in nodes.keys():
        if te.is_field_node_id(nid):
            try:
                n = int(nid[1:])
                if n >= 11:  # 排除 J0-J10
                    ids.append(nid)
            except:
                pass
    ids = sorted(set(ids))
    print(json.dumps(ids, ensure_ascii=False))
except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)
"@

$TmpPyFile = Join-Path $ProjectRoot "_tmp_get_nodes.py"
$PyCode | Set-Content -Path $TmpPyFile -Encoding UTF8

try {
    $eligibleJson = python $TmpPyFile
    if ($LASTEXITCODE -ne 0) { throw "Python script failed execution." }
}
catch {
    Remove-Item $TmpPyFile -ErrorAction SilentlyContinue
    Write-Error "Failed to execute Python node parser. Error: $_"
}
finally {
    Remove-Item $TmpPyFile -ErrorAction SilentlyContinue
}

# 解析 JSON 结果
try {
    $Eligible = $eligibleJson | ConvertFrom-Json
    if ($null -ne $Eligible.error) { throw $Eligible.error }
    if ($Eligible.Count -lt 1) { throw "Eligible fault node list is empty." }
    Write-Host "[INFO] Loaded $($Eligible.Count) eligible nodes." -ForegroundColor Green
}
catch {
    Write-Error "Failed to parse JSON output from Python: $_"
}

# ================= 主循环：H0 Sweep =================
# 11.0 ~ 23.0, step 0.2 => 0..60 共 61 个点
for ($hi = 0; $hi -le 60; $hi++) {
    
    # 计算 H0 (保留一位小数)
    $H0_Val = 11.0 + 0.2 * $hi
    $H0_Tag = "H0_{0:F1}" -f $H0_Val
    $H0_Dir = Join-Path $OutRoot $H0_Tag

    if (-not (Test-Path $H0_Dir)) { New-Item -ItemType Directory -Force -Path $H0_Dir | Out-Null }

    Write-Host "`n>>> Processing Group: $H0_Tag ($($hi+1)/61)" -ForegroundColor Yellow

    for ($s = 0; $s -lt $ScenesPerH0; $s++) {

        # ------ 确定性 Seed 生成 (LCG) ------
        # 使用 [uint64] 防止溢出，模拟 Python 的大整数运算
        $term1 = [uint64]$BaseSeed * 1000003
        $term2 = [uint64]$hi * 1009
        $term3 = [uint64]$s * 9176
        $seed64 = ($term1 + $term2 + $term3) % 4294967296
        
        # 为了 Python 的 random.seed (通常接受 int)，取 32 位正整数
        $seed32 = [int]($seed64 % 2147483647)
        if ($seed32 -lt 0) { $seed32 = -$seed32 }

        # ------ 随机参数选择 ------
        $rng = [System.Random]::new($seed32)
        $faultNode  = $Eligible[$rng.Next(0, $Eligible.Count)]
        # 故障发生在第 1~29 个轮灌组之后
        $faultAfter = $rng.Next(1, 30) 

        # ------ 路径构建 ------
        $sceneName   = "scene_{0:000}__node_{1}__t{2:00}__seed_{3}" -f $s, $faultNode, $faultAfter, $seed64
        $sceneDir    = Join-Path $H0_Dir $sceneName
        $payloadPath = Join-Path $sceneDir "payload.json"

        # 断点续传：跳过已存在 payload.json 的目录
        if (Test-Path $payloadPath) { 
            Write-Host "    [Skip] $sceneName (Finished)" -ForegroundColor DarkGray
            continue 
        }

        # 创建场景目录
        New-Item -ItemType Directory -Force -Path $sceneDir | Out-Null

        # ------ 调用 Python 推理 ------
        Write-Host "    [Run] $sceneName (Node=$faultNode, After=$faultAfter)"
        
        $t0 = [System.Diagnostics.Stopwatch]::StartNew()

        # 注意：PowerShell 中传递参数最好显式用引号包裹字符串
        python $InferScript `
            --dataset_dir "$DatasetDir" `
            --generator_ckpt "$GenCkpt" `
            --feas_ckpt "$FeasCkpt" `
            --feas_threshold $FeasThreshold `
            --nodes "$NodesXlsx" `
            --pipes "$PipesXlsx" `
            --root "J0" `
            --Hmin 11.59 `
            --q_lateral 0.012 `
            --H0 $H0_Val `
            --fault_node "$faultNode" `
            --fault_after_groups $faultAfter `
            --prefault_force_k $PrefaultForceK `
            --seed $seed64 `
            --out "$sceneDir"

        $t0.Stop()
        
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "    [Error] Python script exited with code $LASTEXITCODE in $sceneDir"
        } else {
            # 记录运行时间
            $rt = "{0:F6}" -f $t0.Elapsed.TotalSeconds
            Set-Content -Path (Join-Path $sceneDir "runtime_sec.txt") -Value $rt
        }
    }
}

# ================= 汇总结果 =================
Write-Host "`n[INFO] All iterations completed. Generating summary CSV..." -ForegroundColor Cyan

if (Test-Path $CollectScript) {
    python $CollectScript `
        --runs_root "$OutRoot" `
        --out_csv (Join-Path $OutRoot "summary_all.csv")
    Write-Host "[INFO] Summary generated at: $(Join-Path $OutRoot "summary_all.csv")" -ForegroundColor Green
} else {
    Write-Warning "Collection script not found at $CollectScript. Skipping summary generation."
}

Write-Host "[DONE] Script execution finished."