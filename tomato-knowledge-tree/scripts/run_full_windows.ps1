param(
    [string]$EnvPath = "",
    [string]$FiftyOneDataset = "tomato-map-local",
    [ValidateSet("modelscope", "huggingface", "existing")]
    [string]$ModelSource = "modelscope",
    [string]$ModelPath = "models\Qwen3-VL-Embedding-2B",
    [int[]]$CameraIds = @(2),
    [int[]]$PoseIds = @(1, 4, 7, 10),
    [int]$EmbeddingBatchSize = 0,
    [int]$MaxPixels = 0,
    [switch]$SkipInstall,
    [switch]$SkipPrepare,
    [switch]$ForceEmbeddings,
    [switch]$AllowCpu
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $EnvPath) {
    $EnvPath = Join-Path $root ".conda-env"
}

function Resolve-ProjectPath {
    param([string]$Path)
    if ([IO.Path]::IsPathRooted($Path)) {
        return [IO.Path]::GetFullPath($Path)
    }
    return [IO.Path]::GetFullPath((Join-Path $root $Path))
}

function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$FilePath failed with exit code $LASTEXITCODE"
    }
}

$EnvPath = Resolve-ProjectPath $EnvPath
$ModelPath = Resolve-ProjectPath $ModelPath
$processedDir = Join-Path $root "data\processed\tomatomap_kt5k"
$embeddingPath = Join-Path $processedDir "embeddings.npz"
$embeddingCache = Join-Path $processedDir "embeddings_cache"
$configPath = Join-Path $root "configs\tomatomap.yaml"
$pythonExe = Join-Path $EnvPath "python.exe"

Push-Location $root
try {
    if (-not $SkipInstall) {
        Write-Host "[1/5] Installing the CUDA, Qwen, and FiftyOne environment..."
        & (Join-Path $PSScriptRoot "bootstrap_windows.ps1") -Device cu128 -EnvPath $EnvPath -WithQwen -WithFiftyOne
    } else {
        Write-Host "[1/5] Skipping dependency installation as requested."
    }
    if (-not (Test-Path $pythonExe)) {
        throw "Python environment was not found at $pythonExe"
    }
    $env:PYTHONPATH = Join-Path $root "src"

    $gpuOutput = & $pythonExe -c @"
import json
import torch
result = {'available': torch.cuda.is_available(), 'torch': torch.__version__}
if result['available']:
    p = torch.cuda.get_device_properties(0)
    result.update({'name': p.name, 'memory_gb': round(p.total_memory / 1024**3, 2)})
print(json.dumps(result, ensure_ascii=False))
"@
    if ($LASTEXITCODE -ne 0) {
        throw "PyTorch GPU inspection failed"
    }
    $gpu = ($gpuOutput | Select-Object -Last 1) | ConvertFrom-Json
    Write-Host "PyTorch device: $($gpu | ConvertTo-Json -Compress)"
    if (-not $gpu.available -and -not $AllowCpu) {
        throw "CUDA is unavailable. Do not extract 5,000+ Qwen image vectors on CPU. Check the NVIDIA driver, then rerun."
    }

    Write-Host "[2/5] Preparing the leakage-safe TomatoMAP knowledge-tree tables..."
    if (-not $SkipPrepare) {
        $prepareArgs = @(
            "-m", "tomato_kt.prepare_tomatomap",
            "--fiftyone-dataset", $FiftyOneDataset,
            "--output", $processedDir,
            "--camera-ids"
        )
        $prepareArgs += @($CameraIds | ForEach-Object { $_.ToString() })
        $prepareArgs += "--pose-ids"
        $prepareArgs += @($PoseIds | ForEach-Object { $_.ToString() })
        Invoke-Checked -FilePath $pythonExe -Arguments $prepareArgs
    } else {
        foreach ($name in @("manifest.csv", "nodes.csv", "edges.csv", "media.csv")) {
            if (-not (Test-Path (Join-Path $processedDir $name))) {
                throw "-SkipPrepare was used but $name is missing below $processedDir"
            }
        }
        Write-Host "Using existing tables in $processedDir"
    }

    Write-Host "[3/5] Ensuring Qwen3-VL-Embedding-2B weights are available..."
    if (-not (Test-Path (Join-Path $ModelPath "config.json"))) {
        if ($ModelSource -eq "existing") {
            throw "config.json is missing below $ModelPath"
        }
        if ($ModelSource -eq "modelscope") {
            Invoke-Checked -FilePath $pythonExe -Arguments @(
                "-m", "pip", "install", "modelscope>=1.31,<2",
                "--index-url", "https://mirrors.aliyun.com/pypi/simple/",
                "--trusted-host", "mirrors.aliyun.com"
            )
            $modelId = "Qwen/Qwen3-VL-Embedding-2B"
        } else {
            Invoke-Checked -FilePath $pythonExe -Arguments @(
                "-m", "pip", "install", "huggingface-hub>=0.34"
            )
            $modelId = "Qwen/Qwen3-VL-Embedding-2B"
        }
        Invoke-Checked -FilePath $pythonExe -Arguments @(
            "-m", "tomato_kt.download_model",
            "--source", $ModelSource,
            "--model-id", $modelId,
            "--output", $ModelPath
        )
    } else {
        Write-Host "Using existing model at $ModelPath"
    }

    if ($EmbeddingBatchSize -le 0) {
        if ($gpu.available -and [double]$gpu.memory_gb -ge 11) {
            $EmbeddingBatchSize = 2
        } else {
            $EmbeddingBatchSize = 1
        }
    }
    if ($MaxPixels -le 0) {
        if (-not $gpu.available) {
            $MaxPixels = 65536
        } elseif ([double]$gpu.memory_gb -lt 8) {
            $MaxPixels = 131072
        } elseif ([double]$gpu.memory_gb -lt 11) {
            $MaxPixels = 196608
        } else {
            $MaxPixels = 262144
        }
    }

    Write-Host "[4/5] Extracting resumable Qwen text/image embeddings..."
    Write-Host "Embedding batch size=$EmbeddingBatchSize; max pixels=$MaxPixels"
    if ($ForceEmbeddings -or -not (Test-Path $embeddingPath)) {
        Invoke-Checked -FilePath $pythonExe -Arguments @(
            "-m", "tomato_kt.embeddings",
            "--backend", "qwen",
            "--data-dir", $processedDir,
            "--output", $embeddingPath,
            "--model-path", $ModelPath,
            "--vendor-root", (Join-Path $root "vendor\qwen_embed"),
            "--batch-size", $EmbeddingBatchSize.ToString(),
            "--max-pixels", $MaxPixels.ToString(),
            "--cache-dir", $embeddingCache
        )
    } else {
        Write-Host "Using existing embeddings at $embeddingPath"
    }

    Write-Host "[5/5] Starting the complete gated + hyperbolic model (100 epochs with early stopping)..."
    $trainArgs = @("-m", "tomato_kt.train", "--config", $configPath)
    if ($gpu.available) {
        $trainArgs += @("--device", "cuda")
    } else {
        $trainArgs += @("--device", "cpu")
    }
    Invoke-Checked -FilePath $pythonExe -Arguments $trainArgs

    Write-Host ""
    Write-Host "Complete run finished."
    Write-Host "Metrics: $(Join-Path $root 'outputs\tomatomap_qwen3vl\metrics.json')"
    Write-Host "Tree:    $(Join-Path $root 'outputs\tomatomap_qwen3vl\constructed_tree_edges.csv')"
} finally {
    Pop-Location
}
