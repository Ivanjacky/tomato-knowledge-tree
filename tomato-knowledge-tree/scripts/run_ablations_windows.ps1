param(
    [string]$EnvPath = "",
    [int[]]$Seeds = @(42, 43, 44, 45, 46),
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $EnvPath) {
    $EnvPath = Join-Path $root ".conda-env"
}
$pythonExe = Join-Path $EnvPath "python.exe"
$dataDir = Join-Path $root "data\processed\tomatomap_kt5k"
$ablationRoot = Join-Path $root "outputs\ablations"
$fullAggregateDir = Join-Path $root "outputs\tomatomap_qwen3vl_multiseed"

$experiments = @(
    @{ Name = "euclidean"; Config = "configs\ablation_euclidean.yaml"; AllowInvalid = $false },
    @{ Name = "mean_fusion"; Config = "configs\ablation_mean_fusion.yaml"; AllowInvalid = $false },
    @{ Name = "no_depth"; Config = "configs\ablation_no_depth.yaml"; AllowInvalid = $false },
    @{ Name = "no_align"; Config = "configs\ablation_no_align.yaml"; AllowInvalid = $false },
    @{ Name = "symmetric_score"; Config = "configs\ablation_symmetric_score.yaml"; AllowInvalid = $false },
    @{ Name = "no_item_text"; Config = "configs\ablation_no_item_text.yaml"; AllowInvalid = $false },
    @{ Name = "relaxed_candidates"; Config = "configs\ablation_relaxed_candidates.yaml"; AllowInvalid = $true }
)

function Invoke-LoggedPython {
    param(
        [string[]]$CommandArgs,
        [string]$LogPath,
        [string]$FailureMessage
    )
    & $pythonExe @CommandArgs 2>&1 | Tee-Object -FilePath $LogPath -Append
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$FailureMessage (exit code $exitCode); complete output: $LogPath"
    }
}

if (-not (Test-Path $pythonExe)) {
    throw "Python environment was not found at $pythonExe"
}
foreach ($path in @(
    (Join-Path $dataDir "nodes.csv"),
    (Join-Path $dataDir "edges.csv"),
    (Join-Path $dataDir "manifest.csv"),
    (Join-Path $dataDir "embeddings.npz"),
    (Join-Path $fullAggregateDir "multiseed_metrics.json")
)) {
    if (-not (Test-Path $path)) {
        throw "Required formal input is missing: $path"
    }
}

Push-Location $root
try {
    $env:PYTHONPATH = Join-Path $root "src"
    & $pythonExe -c "import torch, sys; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable'); sys.exit(0 if torch.cuda.is_available() else 2)"
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA GPU is required for the formal ablation runs"
    }

    $comparisonSpecs = @("full=$fullAggregateDir")
    foreach ($experiment in $experiments) {
        $name = $experiment.Name
        $configPath = Join-Path $root $experiment.Config
        if (-not (Test-Path $configPath)) {
            throw "Ablation config is missing: $configPath"
        }
        $experimentDir = Join-Path $ablationRoot $name
        $aggregateDir = Join-Path $experimentDir "multiseed"
        New-Item -ItemType Directory -Force -Path $experimentDir | Out-Null
        $runSpecs = @()
        foreach ($seed in $Seeds) {
            $runDir = Join-Path $experimentDir "seed$seed"
            New-Item -ItemType Directory -Force -Path $runDir | Out-Null
            $logPath = Join-Path $runDir "run.log"
            $trainingRequired = @(
                "preflight.json",
                "best_model.pt",
                "history.json",
                "metrics.json",
                "predicted_edges.csv",
                "constructed_tree_edges.csv"
            )
            $trainingComplete = ($trainingRequired | Where-Object {
                -not (Test-Path (Join-Path $runDir $_))
            }).Count -eq 0
            if ($trainingComplete -and -not $Force) {
                Write-Host "[$name seed $seed] training outputs complete; reusing $runDir"
            } else {
                Write-Host "[$name seed $seed] starting formal training..."
                Invoke-LoggedPython `
                    -CommandArgs @(
                        "-m", "tomato_kt.train",
                        "--config", $configPath,
                        "--device", "cuda",
                        "--seed", "$seed",
                        "--output-dir", $runDir
                    ) `
                    -LogPath $logPath `
                    -FailureMessage "Training failed for $name seed $seed"
            }

            $evaluationRequired = @(
                "supplementary_metrics.json",
                "per_class_metrics.csv",
                "structure_validation.json"
            )
            $evaluationComplete = ($evaluationRequired | Where-Object {
                -not (Test-Path (Join-Path $runDir $_))
            }).Count -eq 0
            if ($evaluationComplete -and -not $Force) {
                Write-Host "[$name seed $seed] supplementary outputs complete; reusing them"
            } else {
                Write-Host "[$name seed $seed] computing supplementary metrics..."
                Invoke-LoggedPython `
                    -CommandArgs @(
                        "-m", "tomato_kt.evaluate",
                        "--data-dir", $dataDir,
                        "--output-dir", $runDir
                    ) `
                    -LogPath $logPath `
                    -FailureMessage "Evaluation failed for $name seed $seed"
            }
            $runSpecs += "$seed=$runDir"
        }

        $aggregateLog = Join-Path $experimentDir "aggregate.log"
        $aggregateArgs = @(
            "-m", "tomato_kt.aggregate_runs",
            "--output-dir", $aggregateDir
        )
        if ($experiment.AllowInvalid) {
            $aggregateArgs += "--allow-invalid-tree"
        }
        foreach ($spec in $runSpecs) {
            $aggregateArgs += @("--run", $spec)
        }
        Invoke-LoggedPython `
            -CommandArgs $aggregateArgs `
            -LogPath $aggregateLog `
            -FailureMessage "Aggregation failed for $name"
        $comparisonSpecs += "$name=$aggregateDir"
    }

    $summaryDir = Join-Path $ablationRoot "summary"
    New-Item -ItemType Directory -Force -Path $summaryDir | Out-Null
    $comparisonArgs = @(
        "-m", "tomato_kt.compare_experiments",
        "--reference", "full",
        "--data-dir", $dataDir,
        "--output-dir", $summaryDir
    )
    foreach ($spec in $comparisonSpecs) {
        $comparisonArgs += @("--experiment", $spec)
    }
    Invoke-LoggedPython `
        -CommandArgs $comparisonArgs `
        -LogPath (Join-Path $summaryDir "comparison.log") `
        -FailureMessage "Ablation comparison failed"
    Write-Host "Ablation results: $(Join-Path $summaryDir 'ablation_comparison.json')"
} finally {
    Pop-Location
}
