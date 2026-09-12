#requires -Version 5.1
<#
.SYNOPSIS
    Download model weights into models\ (not tracked by git).

.DESCRIPTION
    Defaults to hf-mirror.com, which is a full mirror of huggingface.co.
    huggingface.co is unreachable from some networks -- keep the default there.
    If you can reach huggingface.co directly, pass -Endpoint https://huggingface.co

    NOTE: this file is intentionally PURE ASCII.
    Windows PowerShell 5.1 reads .ps1 files using the system ANSI codepage
    (GBK on a Chinese Windows) unless the file has a UTF-8 BOM. Any non-ASCII
    character here would be mis-decoded and break parsing.
    Keep it ASCII only. Do not "translate" the messages.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\fetch_models.ps1

.EXAMPLE
    # force re-download
    powershell -ExecutionPolicy Bypass -File scripts\fetch_models.ps1 -Force
#>
[CmdletBinding()]
param(
    [string]$Endpoint = "https://hf-mirror.com",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# ---- locate repo root --------------------------------------------------- #
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Root "pyproject.toml"))) { $Root = $PSScriptRoot }
if (-not (Test-Path (Join-Path $Root "pyproject.toml"))) {
    throw "pyproject.toml not found. Run this script from inside the pixelart repo."
}
Set-Location $Root

$Repo    = "onnx-community/depth-anything-v2-small"
$DestDir = Join-Path "models" "depth-anything-v2-small"
$Base    = "$Endpoint/$Repo/resolve/main"
$Files   = @("config.json", "preprocessor_config.json", "onnx/model.onnx")

New-Item -ItemType Directory -Force -Path (Join-Path $DestDir "onnx") | Out-Null

Write-Host "endpoint : $Endpoint" -ForegroundColor Cyan
Write-Host "dest     : $DestDir" -ForegroundColor Cyan
Write-Host ""

foreach ($rel in $Files) {
    $dest = Join-Path $DestDir ($rel -replace "/", "\")
    if ((Test-Path $dest) -and -not $Force -and ((Get-Item $dest).Length -gt 0)) {
        $mb = [math]::Round((Get-Item $dest).Length / 1MB, 1)
        Write-Host ("  [skip] {0}  ({1} MB)" -f $rel, $mb) -ForegroundColor DarkGray
        continue
    }
    $url = "$Base/$rel"
    Write-Host ("  [get ] {0}" -f $rel) -ForegroundColor Green
    & curl.exe -L --fail --retry 3 --retry-delay 2 --connect-timeout 20 -o $dest $url
    if ($LASTEXITCODE -ne 0) {
        throw "download failed: $url"
    }
    $mb = [math]::Round((Get-Item $dest).Length / 1MB, 1)
    Write-Host ("  [ok  ] {0}  ({1} MB)" -f $rel, $mb) -ForegroundColor Green
}

# ---- sanity check: ONNX protobuf header should be printable ASCII ------- #
$onnx  = Join-Path $DestDir "onnx\model.onnx"
$len   = (Get-Item $onnx).Length
$bytes = [System.IO.File]::ReadAllBytes($onnx)
$head  = ($bytes[0..15] | ForEach-Object { $_.ToString("x2") }) -join " "
$ascii = -join ($bytes[0..47] | ForEach-Object {
    if ($_ -ge 32 -and $_ -lt 127) { [char]$_ } else { "." }
})

Write-Host ""
Write-Host "model.onnx"
Write-Host ("  size   : {0} MB" -f [math]::Round($len / 1MB, 1))
Write-Host ("  header : {0}" -f $head)
Write-Host ("  ascii  : {0}" -f $ascii)

if ($len -lt 50MB) {
    Write-Host ""
    Write-Host "WARNING: file looks too small (expected about 99 MB). Re-run with -Force." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Done. Next:" -ForegroundColor Cyan
Write-Host "  .\.venv\Scripts\python.exe tools\depth_probe.py"
Write-Host ""
