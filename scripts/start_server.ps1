#requires -Version 5.1
<#
.SYNOPSIS
    Start the pixelart tuning server (tools\m3_server.py) on localhost.

.DESCRIPTION
    Uses the project venv (.venv) created by scripts\setup.ps1.
    Refuses to start if the port is already in use (the server self-checks
    too, but failing here gives a clearer message).

    NOTE: this file is intentionally PURE ASCII.
    Windows PowerShell 5.1 reads .ps1 files using the system ANSI codepage
    (GBK on a Chinese Windows) unless the file has a UTF-8 BOM. Any non-ASCII
    character here would be mis-decoded and break parsing.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1

.EXAMPLE
    # different port
    powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1 -Port 8801

.EXAMPLE
    # also open the tuning UI in the default browser
    powershell -ExecutionPolicy Bypass -File scripts\start_server.ps1 -Open
#>
[CmdletBinding()]
param(
    [int]$Port = 8770,
    [switch]$Open
)

$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "venv not found at $venvPython - run scripts\setup.ps1 first"
    exit 1
}

# fail fast if something is already listening on the port
$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    Write-Output "Port $Port is already in use by PID $($busy.OwningProcess[0])."
    Write-Output "Nothing to do (or kill that process first:  taskkill /PID <pid> /F)"
    exit 1
}

Write-Output "Starting pixelart server on http://127.0.0.1:$Port ..."
Write-Output "(first visit per image runs depth + scene prepare, ~10s; then cached)"

if ($Open) { Start-Process "http://127.0.0.1:$Port/" }

& $venvPython -u (Join-Path $repo "tools\m3_server.py") --port $Port
