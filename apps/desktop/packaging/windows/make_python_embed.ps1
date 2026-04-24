# ============================================================
# Build the embedded Python distribution that gets shipped
# inside Comni.exe, so the app is fully self-contained (no
# user-side Python / miniconda required).
#
# Outputs:  apps/desktop/packaging/windows/python-embed/
#
# Usage:
#   pwsh apps/desktop/packaging/windows/make_python_embed.ps1
#   pwsh apps/desktop/packaging/windows/make_python_embed.ps1 -PyVer 3.12.10
# ============================================================

[CmdletBinding()]
param(
    [string]$PyVer = "3.12.10",
    [string]$CacheDir = "D:\tc_mb"
)

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot  = (Resolve-Path (Join-Path $ScriptDir '..\..\..\..')).Path
Set-Location $RepoRoot

$EmbedDir  = Join-Path $ScriptDir 'python-embed'
$ZipPath   = Join-Path $CacheDir "python-$PyVer-embed-amd64.zip"
$ReqFile   = Join-Path $ScriptDir 'runtime_requirements.txt'

Write-Host "=============================================================="
Write-Host "  Build embedded Python $PyVer for Comni"
Write-Host "=============================================================="
Write-Host "  Output dir : $EmbedDir"
Write-Host "  Zip cache  : $ZipPath"
Write-Host "  Req file   : $ReqFile"
Write-Host ""

# ── 1. Download embeddable Python if needed ──
if (-not (Test-Path $ZipPath)) {
    $url = "https://www.python.org/ftp/python/$PyVer/python-$PyVer-embed-amd64.zip"
    Write-Host "[1/4] Downloading $url ..."
    $ProgressPreference = 'SilentlyContinue'
    if (-not (Test-Path $CacheDir)) { New-Item -ItemType Directory -Path $CacheDir | Out-Null }
    Invoke-WebRequest -Uri $url -OutFile $ZipPath -UseBasicParsing
}
Write-Host ("[1/4] Zip: {0:N1} MB" -f ((Get-Item $ZipPath).Length / 1MB))

# ── 2. Extract ──
if (Test-Path $EmbedDir) {
    Write-Host "[2/4] Cleaning old $EmbedDir"
    Remove-Item -Recurse -Force $EmbedDir
}
New-Item -ItemType Directory -Force -Path $EmbedDir | Out-Null
Expand-Archive -Path $ZipPath -DestinationPath $EmbedDir -Force
Write-Host "[2/4] Extracted to $EmbedDir"

# ── 3. Enable site module so Lib\site-packages works ──
$pthPath = Join-Path $EmbedDir 'python312._pth'
if (-not (Test-Path $pthPath)) {
    throw "Missing $pthPath — embeddable distribution layout changed?"
}
Set-Content -Path $pthPath -Value @"
python312.zip
.
Lib\site-packages
..\apps\server
import site
"@
Write-Host "[3/4] Patched python312._pth (enabled site + apps\server in sys.path)"

# ── 4. Bootstrap pip + install runtime deps ──
$PyExe = Join-Path $EmbedDir 'python.exe'
$GetPip = Join-Path $EmbedDir 'get-pip.py'

Write-Host "[4/4] Bootstrapping pip ..."
New-Item -ItemType Directory -Force -Path (Join-Path $EmbedDir 'Lib\site-packages') | Out-Null
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $GetPip -UseBasicParsing
& $PyExe $GetPip --no-warn-script-location --quiet
if ($LASTEXITCODE -ne 0) { throw "get-pip.py failed" }

Write-Host "[4/4] Installing deps from $ReqFile ..."
& $PyExe -m pip install -r $ReqFile --no-warn-script-location --quiet
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

# ── Strip obvious junk to save bundle size ──
Remove-Item (Join-Path $EmbedDir 'get-pip.py') -ErrorAction SilentlyContinue
Get-ChildItem $EmbedDir -Recurse -Directory `
    | Where-Object { $_.Name -in @('tests', 'test', '__pycache__') } `
    | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# ── Verify ──
Write-Host ""
Write-Host "=============================================================="
Write-Host "  Verify"
Write-Host "=============================================================="
& $PyExe -c "import fastapi, uvicorn, httpx, websockets, pydantic, numpy, soundfile, librosa, PIL, onnxruntime, tqdm, yaml, huggingface_hub, requests; print('All runtime imports OK')"
if ($LASTEXITCODE -ne 0) { throw "Runtime import verification failed" }

$size = (Get-ChildItem $EmbedDir -Recurse | Measure-Object -Property Length -Sum).Sum
Write-Host ("  python-embed total: {0:N1} MB" -f ($size / 1MB))
Write-Host ""
Write-Host "Done. You can now run:"
Write-Host "  pwsh apps/desktop/packaging/windows/build.ps1"
