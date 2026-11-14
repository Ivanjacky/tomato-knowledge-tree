param(
    [string]$EnvPath = "",
    [int[]]$Seeds = @(43, 44, 45, 46),
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $EnvPath) {
    $EnvPath = Join-Path $root ".conda-env"
}
$pythonExe = Join-Path $EnvPath "python.exe"
$configPath = Join-Path $root "configs\tomatomap.yaml"
$referenceDir = Join-Path $root "outputs\tomatomap_qwen3vl"
$aggregateDir = Join-Path $root "outputs\tomatomap_qwen3vl_multiseed"

if (-not (Test-Path $pythonExe)) {
    throw "Python environment was not found at $pythonExe"
}
foreach ($name in @(
    "metrics.json",
    "predicted_edges.csv",
    "constructed_tree_edges.csv",
    "supplementary_metrics.json",
    "per_class_metrics.csv"
)) {
    if (-not (Test-Path (Join-Path $referenceDir $name))) {
        throw "Seed 42 reference output is incomplete: $name is missing"
    }
}

Push-Location $root
try {
    $env:PYTHONPATH = Join-Path $root "src"
    $runSpecs = @("42=$referenceDir")
    foreach ($seed in $Seeds) {
        if ($seed -eq 42) {
            continue
        }
        $runDir = Join-Path $root "outputs\tomatomap_qwen3vl_seed$seed"
        $required = @(
            "metrics.json",
            "predicted_edges.csv",
            "constructed_tree_edges.csv",
            "supplementary_metrics.json",
            "per_class_metrics.csv"
        )
        $isComplete = ($required | Where-Object {
            -not (Test-Path (Join-Path $runDir $_))
        }).Count -eq 0
        if ($isComplete -and -not $Force) {
            Write-Host "Seed $seed is complete; reusing $runDir"
        } else {
            Write-Host "Training complete model with seed $seed..."
            & $pythonExe -m tomato_kt.train `
                --config $configPath `
                --device cuda `
                --seed $seed `
                --output-dir $runDir
            if ($LASTEXITCODE -ne 0) {
                throw "Training failed for seed $seed with exit code $LASTEXITCODE"
            }
            & $pythonExe -m tomato_kt.evaluate `
                --data-dir "data\processed\tomatomap_kt5k" `
                --output-dir $runDir
            if ($LASTEXITCODE -ne 0) {
                throw "Supplementary evaluation failed for seed $seed"
            }
        }
        $runSpecs += "$seed=$runDir"
    }

    $aggregateArgs = @(
        "-m", "tomato_kt.aggregate_runs",
        "--output-dir", $aggregateDir
    )
    foreach ($spec in $runSpecs) {
        $aggregateArgs += @("--run", $spec)
    }
    & $pythonExe @aggregateArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Multi-seed aggregation failed with exit code $LASTEXITCODE"
    }
    $analysisArgs = @(
        "-m", "tomato_kt.analyze_runs",
        "--data-dir", "data\processed\tomatomap_kt5k",
        "--output-dir", $aggregateDir
    )
    foreach ($spec in $runSpecs) {
        $analysisArgs += @("--run", $spec)
    }
    & $pythonExe @analysisArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Multi-seed error analysis failed with exit code $LASTEXITCODE"
    }
    Write-Host "Multi-seed results: $(Join-Path $aggregateDir 'multiseed_metrics.json')"
} finally {
    Pop-Location
}
