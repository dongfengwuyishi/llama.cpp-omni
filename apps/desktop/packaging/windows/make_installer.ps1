<#
.SYNOPSIS
    One-click packager for the Windows installer.

.DESCRIPTION
    Verifies that the PyInstaller bundle (dist\Comni\) exists, locates Inno
    Setup 6 (ISCC.exe), and compiles Comni.iss into a single-file installer:
        release\Comni-Setup-<version>-win64.exe

    If you haven't built the bundle yet, run build.ps1 first.

.PARAMETER Version
    Semantic version embedded into the installer (defaults to 1.0.0).

.PARAMETER BundleDir
    Path to the PyInstaller onedir output (defaults to repo\dist\Comni).

.PARAMETER OutputDir
    Where the final installer is written (defaults to repo\release).

.EXAMPLE
    PS> powershell -ExecutionPolicy Bypass -File .\make_installer.ps1
    PS> powershell -ExecutionPolicy Bypass -File .\make_installer.ps1 -Version 1.2.3
#>
param(
    [string]$Version = "1.0.0",
    [string]$BundleDir,
    [string]$OutputDir
)

$ErrorActionPreference = "Stop"

# ---------- Resolve paths ----------
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Resolve-Path (Join-Path $ScriptDir "..\..\..\..") | Select-Object -ExpandProperty Path
if (-not $BundleDir) { $BundleDir = Join-Path $RepoRoot "dist\Comni" }
if (-not $OutputDir) { $OutputDir = Join-Path $RepoRoot "release" }
$IssFile = Join-Path $ScriptDir "Comni.iss"

Write-Host "========================================"
Write-Host " Comni installer builder"
Write-Host "========================================"
Write-Host "  Repo root : $RepoRoot"
Write-Host "  Bundle    : $BundleDir"
Write-Host "  Version   : $Version"
Write-Host "  Output    : $OutputDir"
Write-Host ""

# ---------- Sanity checks ----------
if (-not (Test-Path $BundleDir)) {
    throw "Bundle not found: $BundleDir. Build it first with build.ps1."
}
$exePath = Join-Path $BundleDir "Comni.exe"
if (-not (Test-Path $exePath)) {
    throw "Comni.exe not found in bundle: $exePath"
}
if (-not (Test-Path $IssFile)) {
    throw "Inno Setup script not found: $IssFile"
}

# ---------- Locate ISCC.exe ----------
$candidates = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles}\Inno Setup 6\ISCC.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
)
$ISCC = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $ISCC) {
    Write-Host "Inno Setup 6 not found. Installing via winget..." -ForegroundColor Yellow
    & winget install --id JRSoftware.InnoSetup -e --accept-source-agreements --accept-package-agreements --silent
    if ($LASTEXITCODE -ne 0) {
        throw "winget failed to install Inno Setup. Download manually from https://jrsoftware.org/isdl.php"
    }
    $ISCC = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $ISCC) { throw "Inno Setup installed but ISCC.exe still not found." }
}
Write-Host "ISCC: $ISCC" -ForegroundColor Cyan

# ---------- Compute bundle size (informational) ----------
$sizeBytes = (Get-ChildItem $BundleDir -Recurse -File -ErrorAction SilentlyContinue |
    Measure-Object -Property Length -Sum).Sum
$sizeGB = [math]::Round($sizeBytes / 1GB, 2)
Write-Host "Bundle size: $sizeGB GB ($([math]::Round($sizeBytes/1MB,0)) MB)"
Write-Host ""

# ---------- Prepare output dir ----------
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

# ---------- Compile ----------
Write-Host "Compiling installer (this can take several minutes due to LZMA2/max)..." -ForegroundColor Yellow
$started = Get-Date

$absBundle = (Resolve-Path $BundleDir).Path
$absOutput = (Resolve-Path $OutputDir).Path

$isccArgs = @(
    "/DMyAppVersion=$Version",
    "/DMySourceDir=$absBundle",
    "/DMyOutputDir=$absOutput",
    "/Qp",
    $IssFile
)

& $ISCC @isccArgs
$code = $LASTEXITCODE
$elapsed = (Get-Date) - $started
Write-Host ""
if ($code -ne 0) {
    throw "ISCC failed with exit code $code after $([math]::Round($elapsed.TotalSeconds,1))s"
}

# ---------- Report ----------
$installer = Get-ChildItem $OutputDir -Filter "Comni-Setup-$Version-win64.exe" -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $installer) {
    throw "Installer not produced in $OutputDir"
}

$sha = Get-FileHash $installer.FullName -Algorithm SHA256
$sizeMB = [math]::Round($installer.Length / 1MB, 1)

Write-Host "========================================" -ForegroundColor Green
Write-Host " Installer built successfully"           -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "  File   : $($installer.FullName)"
Write-Host "  Size   : $sizeMB MB"
Write-Host "  SHA256 : $($sha.Hash)"
Write-Host "  Time   : $([math]::Round($elapsed.TotalSeconds,1)) s"
Write-Host ""
Write-Host "Ship this single file to your users. They double-click to install." -ForegroundColor Cyan
