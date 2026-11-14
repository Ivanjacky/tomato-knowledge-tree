param(
    [string]$DataDir = "data\processed\tomatomap_kt5k",
    [string]$OutputRoot = "outputs\comparisons",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$python = Join-Path $projectRoot ".conda-env\python.exe"
$hf = Join-Path $projectRoot ".conda-env\Scripts\hf.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python was not found: $python"
}
if (-not (Test-Path -LiteralPath $hf)) {
    throw "Hugging Face CLI was not found: $hf"
}
$env:PYTHONPATH = Join-Path $projectRoot "src"

$clipDir = Join-Path $projectRoot "models\CLIP-ViT-B-32"
$siglipDir = Join-Path $projectRoot "models\SigLIP2-base-patch16-224"
if (-not ((Test-Path -LiteralPath (Join-Path $clipDir "config.json")) -and
          (Test-Path -LiteralPath (Join-Path $clipDir "pytorch_model.bin")))) {
    $clipFiles = @(
        "config.json", "preprocessor_config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json", "vocab.json",
        "merges.txt", "pytorch_model.bin"
    )
    & $hf download openai/clip-vit-base-patch32 @clipFiles --local-dir $clipDir
    if ($LASTEXITCODE -ne 0) { throw "CLIP download failed with exit code $LASTEXITCODE" }
}
if (-not ((Test-Path -LiteralPath (Join-Path $siglipDir "config.json")) -and
          (Test-Path -LiteralPath (Join-Path $siglipDir "model.safetensors")))) {
    $siglipFiles = @(
        "config.json", "model.safetensors", "preprocessor_config.json",
        "special_tokens_map.json", "tokenizer.json", "tokenizer.model",
        "tokenizer_config.json"
    )
    & $hf download google/siglip2-base-patch16-224 @siglipFiles --local-dir $siglipDir
    if ($LASTEXITCODE -ne 0) { throw "SigLIP2 download failed with exit code $LASTEXITCODE" }
}

function Test-BaselineComplete([string]$Directory) {
    $required = @(
        "metrics.json", "predicted_edges.csv", "constructed_tree_edges.csv",
        "supplementary_metrics.json", "per_class_metrics.csv",
        "structure_validation.json", "run_info.json"
    )
    foreach ($name in $required) {
        if (-not (Test-Path -LiteralPath (Join-Path $Directory $name))) { return $false }
    }
    return $true
}

$qwenOutput = Join-Path $OutputRoot "qwen3vl_cosine"
$clipOutput = Join-Path $OutputRoot "clip_vit_b32"
$siglipOutput = Join-Path $OutputRoot "siglip2_base_224"
if ($Force -or -not (Test-BaselineComplete $qwenOutput)) {
    & $python -m tomato_kt.cosine_baseline --backend qwen --data-dir $DataDir `
        --descriptions "configs\bbch_descriptions_en.json" --output-dir $qwenOutput `
        --device cuda
    if ($LASTEXITCODE -ne 0) { throw "Qwen cosine baseline failed" }
}
if ($Force -or -not (Test-BaselineComplete $clipOutput)) {
    & $python -m tomato_kt.cosine_baseline --backend hf --data-dir $DataDir `
        --descriptions "configs\bbch_descriptions_en.json" --model-path $clipDir `
        --model-id "openai/clip-vit-base-patch32" --output-dir $clipOutput `
        --cache-dir (Join-Path $DataDir "baseline_cache\clip_vit_b32") `
        --device cuda --image-batch-size 64
    if ($LASTEXITCODE -ne 0) { throw "CLIP baseline failed" }
}
if ($Force -or -not (Test-BaselineComplete $siglipOutput)) {
    & $python -m tomato_kt.cosine_baseline --backend hf --data-dir $DataDir `
        --descriptions "configs\bbch_descriptions_en.json" --model-path $siglipDir `
        --model-id "google/siglip2-base-patch16-224" --output-dir $siglipOutput `
        --cache-dir (Join-Path $DataDir "baseline_cache\siglip2_base_224") `
        --device cuda --image-batch-size 64
    if ($LASTEXITCODE -ne 0) { throw "SigLIP2 baseline failed" }
}

& $python -m tomato_kt.compare_baselines `
    --single "qwen_cosine=$qwenOutput" `
    --single "clip=$clipOutput" `
    --single "siglip2=$siglipOutput" `
    --multi "qwen_euclidean=outputs\ablations\euclidean\multiseed" `
    --multi "qwen_mean=outputs\ablations\mean_fusion\multiseed" `
    --multi "qwen_gate_hyperbolic=outputs\ablations\no_depth\multiseed" `
    --multi "full=outputs\tomatomap_qwen3vl_multiseed" `
    --reference full --data-dir $DataDir --output-dir (Join-Path $OutputRoot "summary")
if ($LASTEXITCODE -ne 0) { throw "Comparison aggregation failed" }
