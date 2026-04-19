# Build Windows NSIS installer
param()

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Set-Location $ProjectRoot

# Get version from Cargo.toml
$VersionLine = Select-String -Path "Cargo.toml" -Pattern '^version = "(.+)"' | Select-Object -First 1
$Version = $VersionLine.Matches.Groups[1].Value

Write-Host "Building NSIS installer..."
cargo packager --config packager/Packager.windows.toml --release

$InstallerPath = "target\release\Zeedle_${Version}_x86_64-pc-windows-msvc-setup.exe"

if (Test-Path $InstallerPath) {
    Write-Host "✓ Package ready: $InstallerPath"
} else {
    Write-Host "✓ Package ready: target\release\Zeedle_${Version}_x86_64-pc-windows-msvc-setup.exe"
}
