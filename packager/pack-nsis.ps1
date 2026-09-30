# Build the Windows NSIS installer.
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ReleaseDir = Join-Path $ProjectRoot "target\release"
$ManifestPath = Join-Path $ProjectRoot "Cargo.toml"
$ConfigPath = Join-Path $PSScriptRoot "Packager.windows.toml"

$InPackageSection = $false
$Version = $null
foreach ($Line in Get-Content -LiteralPath $ManifestPath) {
    if ($Line -match '^\[package\]\s*$') {
        $InPackageSection = $true
        continue
    }
    if ($InPackageSection -and $Line -match '^\s*\[') {
        break
    }
    if ($InPackageSection -and $Line -match '^\s*version\s*=\s*"([^"]+)"') {
        $Version = $Matches[1]
        break
    }
}
if (-not $Version) {
    throw "Could not read the package version from $ManifestPath"
}

$Installer = Join-Path $ReleaseDir "Zeedle_${Version}_x64-setup.exe"
$StagingDir = Join-Path $ReleaseDir ".nsis-staging-$([guid]::NewGuid().ToString('N'))"
$StagedInstaller = Join-Path $StagingDir "zeedle_${Version}_x64-setup.exe"
$VersionedConfigPath = Join-Path $PSScriptRoot ".Packager.windows.$([guid]::NewGuid().ToString('N')).toml"
$BuildCompleted = $false

Push-Location $ProjectRoot
try {
    $ConfigContent = [IO.File]::ReadAllText($ConfigPath)
    $ConfigWithVersion = "version = `"$Version`"`r`n$ConfigContent"
    [IO.File]::WriteAllText($VersionedConfigPath, $ConfigWithVersion, [Text.UTF8Encoding]::new($false))
    New-Item -ItemType Directory -Path $StagingDir -Force | Out-Null

    Write-Host "Building Zeedle $Version NSIS installer..."
    & cargo packager --config $VersionedConfigPath --out-dir $StagingDir
    if ($LASTEXITCODE -ne 0) {
        throw "cargo packager failed with exit code $LASTEXITCODE"
    }

    if (-not (Test-Path -LiteralPath $StagedInstaller -PathType Leaf)) {
        throw "Expected installer was not created: $StagedInstaller"
    }
    $BuildCompleted = $true

    Move-Item -LiteralPath $StagedInstaller -Destination $Installer -Force

    if (-not (Test-Path -LiteralPath $Installer -PathType Leaf)) {
        throw "Installer rename failed: $Installer"
    }
    Write-Host "Package ready: $Installer"
}
finally {
    try {
        if (Test-Path -LiteralPath $VersionedConfigPath) {
            Remove-Item -LiteralPath $VersionedConfigPath -Force
        }
        if (Test-Path -LiteralPath $StagingDir -PathType Container) {
            if ($BuildCompleted -and (Test-Path -LiteralPath $StagedInstaller -PathType Leaf)) {
                Write-Warning "Installer could not be moved; staged artifact remains at $StagedInstaller"
            }
            else {
                Remove-Item -LiteralPath $StagingDir -Recurse -Force
            }
        }
    }
    finally {
        Pop-Location
    }
}
