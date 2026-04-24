# ============================================================
# Comni Desktop — Windows quick start (dev mode)
#
# Runs the PySide6 GUI directly with the system / conda Python.
# No packaging involved — fastest way to try the app.
#
# Usage (from repo root):
#   pwsh apps/start.ps1
#   pwsh apps/start.ps1 -Cli -ModelDir 'D:\models\MiniCPM-o-4_5-gguf'
#   pwsh apps/start.ps1 -Python 'C:\Users\me\miniconda3\envs\tc\python.exe'
# ============================================================

[CmdletBinding()]
param(
    [switch]$Cli,
    [string]$ModelDir = $null,
    [int]$Port = 8006,
    [string]$Python = $null,
    [switch]$SkipDeps
)

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot  = (Resolve-Path (Join-Path $ScriptDir '..')).Path

Set-Location $RepoRoot

Write-Host "=============================================================="
Write-Host "  llama.cpp-omni Desktop App (Windows)"
Write-Host "=============================================================="

# ── Resolve Python ──
if (-not $Python) {
    $cand = @(
        "$env:USERPROFILE\miniconda3\python.exe",
        "$env:USERPROFILE\anaconda3\python.exe",
        "$env:USERPROFILE\Anaconda3\python.exe"
    )
    foreach ($c in $cand) {
        if (Test-Path $c) { $Python = $c; break }
    }
}
if (-not $Python) {
    $pyCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pyCmd) { $Python = $pyCmd.Source }
}
if (-not $Python) { throw "Python not found. Install miniconda or pass -Python." }

Write-Host "  Python : $Python"
$pyVer = & $Python -c "import sys; print(sys.version.split()[0])"
Write-Host "  Version: $pyVer"

# ── Install deps ──
if (-not $SkipDeps) {
    Write-Host ""
    Write-Host "Checking Python dependencies..."
    $needsInstall = $false
    $checkMods = if ($Cli) { @('fastapi', 'uvicorn') } else { @('PySide6', 'fastapi', 'uvicorn') }
    foreach ($m in $checkMods) {
        & $Python -c "import $m" 2>$null
        if ($LASTEXITCODE -ne 0) { $needsInstall = $true; break }
    }
    if ($needsInstall) {
        Write-Host "Installing missing deps from apps/requirements.txt..."
        & $Python -m pip install -r (Join-Path $ScriptDir 'requirements.txt') -q
        if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
    } else {
        Write-Host "  All required packages present."
    }
}

# ── Check llama-server.exe ──
$LlamaCandidates = @(
    (Join-Path $RepoRoot 'build\bin\Release\llama-server.exe'),
    (Join-Path $RepoRoot 'build\bin\llama-server.exe')
)
$LlamaExe = $null
foreach ($c in $LlamaCandidates) {
    if (Test-Path $c) { $LlamaExe = $c; break }
}
if ($LlamaExe) {
    Write-Host "  llama-server.exe: $LlamaExe"
} else {
    Write-Host "  llama-server.exe: (not built yet)" -ForegroundColor Yellow
    Write-Host "    To build:  cmake -B build -DCMAKE_BUILD_TYPE=Release"
    Write-Host "               cmake --build build --config Release --target llama-server -j"
}

# ── Launch ──
Write-Host ""
Write-Host "=============================================================="

if ($Cli) {
    $launcher = Join-Path $ScriptDir 'desktop\launcher.py'
    $launcherArgs = @('--http', '--port', $Port.ToString())
    if ($ModelDir) { $launcherArgs += @('--model-dir', $ModelDir) }
    Write-Host "Launching CLI mode: python $launcher $launcherArgs"
    & $Python $launcher @launcherArgs
} else {
    $gui = Join-Path $ScriptDir 'desktop\windows_app.py'
    Write-Host "Launching GUI: python $gui"
    & $Python $gui
}
