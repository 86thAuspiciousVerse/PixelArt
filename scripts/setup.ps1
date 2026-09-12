#requires -Version 5.1
<#
.SYNOPSIS
    pixelart dev environment bootstrap (Windows / PowerShell).

.DESCRIPTION
    Creates .venv inside the repo, installs dependencies, installs the project
    in editable mode, then runs an import self-check.

    Model weights are NOT handled here -- see scripts\fetch_models.ps1.

    NOTE: this file is intentionally PURE ASCII.
    Windows PowerShell 5.1 reads .ps1 files using the system ANSI codepage
    (GBK on a Chinese Windows) unless the file has a UTF-8 BOM. Any non-ASCII
    character here would be mis-decoded and break parsing.
    Keep it ASCII only. Do not "translate" the messages.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Dev

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Dev -Mirror https://mirrors.aliyun.com/pypi/simple/

.EXAMPLE
    # rebuild the virtualenv from scratch
    powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -Fresh -Dev
#>
[CmdletBinding()]
param(
    [string]$Python = "",
    [string]$Mirror = "https://pypi.tuna.tsinghua.edu.cn/simple",
    [switch]$Fresh,
    [switch]$Dev
)

$ErrorActionPreference = "Stop"

# ---- locate repo root --------------------------------------------------- #
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Root "pyproject.toml"))) { $Root = $PSScriptRoot }
if (-not (Test-Path (Join-Path $Root "pyproject.toml"))) {
    throw "pyproject.toml not found. Run this script from inside the pixelart repo."
}
Set-Location $Root
Write-Host "repo root : $Root" -ForegroundColor Cyan

if ($Fresh -and (Test-Path ".venv")) {
    Write-Host "  -Fresh: removing existing .venv ..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".venv"
}

# ---- pick a Python interpreter ------------------------------------------ #
if (-not $Python) {
    $cands = @(
        "C:\Miniconda3\python.exe",
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
        (Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"),
        "C:\Python313\python.exe"
    )
    foreach ($c in $cands) {
        if (Test-Path $c) { $Python = $c; break }
    }
    if (-not $Python) {
        $g = Get-Command python -ErrorAction SilentlyContinue
        if ($g) { $Python = $g.Source }
    }
}
if (-not $Python) {
    throw "Python not found. Pass one explicitly, e.g. -Python 'C:\Miniconda3\python.exe'"
}

$ver = (& $Python --version 2>&1) -join " "
Write-Host "python    : $Python" -ForegroundColor Cyan
Write-Host "version   : $ver" -ForegroundColor Cyan
if ("$ver" -notmatch "3\.(1[1-9]|[2-9][0-9])") {
    Write-Host "  WARNING: Python >= 3.11 required, got $ver" -ForegroundColor Yellow
}

# ---- virtualenv --------------------------------------------------------- #
$VPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VPy)) {
    Write-Host "[1/5] creating virtualenv .venv ..." -ForegroundColor Green
    & $Python -m venv ".venv"
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
} else {
    Write-Host "[1/5] reusing existing .venv ..." -ForegroundColor Green
}

# ---- pip ---------------------------------------------------------------- #
Write-Host "[2/5] upgrading pip ..." -ForegroundColor Green
& $VPy -m pip install --upgrade pip -i $Mirror --quiet
if ($LASTEXITCODE -ne 0) {
    throw "pip upgrade failed. Try another -Mirror, or drop the -i option to use default PyPI."
}

Write-Host "[3/5] installing runtime dependencies ..." -ForegroundColor Green
$pkgs = @("numpy>=1.26", "pillow>=10.0", "opencv-python-headless>=4.9", "onnxruntime>=1.17")
& $VPy -m pip install @pkgs -i $Mirror
if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }

if ($Dev) {
    Write-Host "      (dev) installing pytest ..." -ForegroundColor Green
    & $VPy -m pip install "pytest>=8.0" -i $Mirror
    if ($LASTEXITCODE -ne 0) { throw "pytest install failed" }
}

Write-Host "[4/5] installing project in editable mode ..." -ForegroundColor Green
& $VPy -m pip install -e . -i $Mirror
if ($LASTEXITCODE -ne 0) { throw "editable install failed" }

# ---- self-check --------------------------------------------------------- #
# NOTE: do NOT inline a multi-line python script via `python -c "..."` here.
# PowerShell 5.1 does not escape embedded double quotes when passing args to
# native executables, so the script gets truncated and python reports
#   SyntaxError: '(' was never closed
# Keep it as a real file: tools\selfcheck.py
Write-Host "[5/5] running tools\selfcheck.py ..." -ForegroundColor Green
& $VPy (Join-Path $Root "tools\selfcheck.py")
if ($LASTEXITCODE -ne 0) { throw "self-check failed. See the output above." }

Write-Host ""
Write-Host "Done. Next steps:" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\python.exe tools\selfcheck.py       # rerun diagnostics anytime"
Write-Host "  .\.venv\Scripts\python.exe tools\fetch_models.py    # only if models are missing"
Write-Host "  .\.venv\Scripts\python.exe tools\depth_probe.py     # spike-1"
Write-Host "  .\.venv\Scripts\python.exe -m pytest                # unit tests (needs -Dev)"
Write-Host ""
