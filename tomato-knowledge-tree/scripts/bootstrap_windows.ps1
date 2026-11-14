param(
    [ValidateSet("cpu", "cu128")]
    [string]$Device = "cpu",
    [string]$EnvPath = "",
    [switch]$WithQwen,
    [switch]$WithFiftyOne
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $EnvPath) {
    $EnvPath = Join-Path $root ".conda-env"
}
$localPackageCache = Join-Path $root ".conda-pkgs"
$localTemp = Join-Path $root ".tmp"
$localAppData = Join-Path $root ".localappdata"
New-Item -ItemType Directory -Force -Path $localPackageCache, $localTemp, $localAppData | Out-Null
$env:CONDA_NO_PLUGINS = "true"
$env:CONDA_PKGS_DIRS = $localPackageCache
$env:CONDA_NUMBER_CHANNEL_NOTICES = "0"
$env:TEMP = $localTemp
$env:TMP = $localTemp
$env:LOCALAPPDATA = $localAppData
$env:PIP_CACHE_DIR = Join-Path $root ".pip-cache"

function Invoke-CondaCommand {
    param([string[]]$CondaArgs)
    & conda --no-plugins @CondaArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Conda command failed with exit code $LASTEXITCODE"
    }
}

function Invoke-EnvironmentPython {
    param([string[]]$PythonArgs)
    $pythonExe = Join-Path $EnvPath "python.exe"
    & $pythonExe @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Install Miniconda/Anaconda first."
}

if (-not (Test-Path (Join-Path $EnvPath "python.exe"))) {
    Invoke-CondaCommand @(
        "create", "--solver", "classic", "-p", $EnvPath, "python=3.11", "pip", "-y"
    )
}

if ($Device -eq "cpu") {
    Invoke-EnvironmentPython @(
        "-m", "pip", "install",
        "torch==2.8.0", "torchvision==0.23.0",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    )
} else {
    Invoke-EnvironmentPython @(
        "-m", "pip", "install",
        "torch==2.8.0", "torchvision==0.23.0",
        "--index-url", "https://download.pytorch.org/whl/cu128"
    )
}

Invoke-EnvironmentPython @(
    "-m", "pip", "install", "-r", (Join-Path $root "requirements-core.txt"),
    "--index-url", "https://mirrors.aliyun.com/pypi/simple/",
    "--trusted-host", "mirrors.aliyun.com"
)
if ($WithQwen) {
    Invoke-EnvironmentPython @(
        "-m", "pip", "install", "-r", (Join-Path $root "requirements-qwen.txt"),
        "--index-url", "https://mirrors.aliyun.com/pypi/simple/",
        "--trusted-host", "mirrors.aliyun.com"
    )
}
if ($WithFiftyOne) {
    Invoke-EnvironmentPython @(
        "-m", "pip", "install", "-r", (Join-Path $root "requirements-data.txt"),
        "--index-url", "https://mirrors.aliyun.com/pypi/simple/",
        "--trusted-host", "mirrors.aliyun.com"
    )
}

Write-Host "Environment ready. Activate it with: conda activate `"$EnvPath`""
Write-Host "See README.md for data preparation, training, and evaluation commands."
