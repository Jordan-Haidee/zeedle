# Build the Windows MSI installer.
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ReleaseDir = Join-Path $ProjectRoot "target\release"
$ManifestPath = Join-Path $ProjectRoot "Cargo.toml"
$ConfigPath = Join-Path $PSScriptRoot "Packager.windows.toml"
$PackageArch = $env:ZEEDLE_PACKAGE_ARCH
if (-not $PackageArch) {
    $PackageArch = if ([Runtime.InteropServices.RuntimeInformation]::OSArchitecture -eq [Runtime.InteropServices.Architecture]::Arm64) { "arm64" } else { "x64" }
}
if ($PackageArch -notin @("x64", "arm64")) {
    throw "Unsupported Windows package architecture: $PackageArch"
}
if ($PackageArch -eq "arm64") {
    & (Join-Path $PSScriptRoot "prepare-windows-arm64-runtime.ps1")
    & (Join-Path $PSScriptRoot "prepare-wix-arm64.ps1")
}
$ResourceGlob = if ($PackageArch -eq "arm64") { "../target/windows-arm64-runtime/*.dll" } else { "DLLs/*.dll" }

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

$MsiVersion = $Version
# WiX accepts numeric prerelease identifiers; assign alpha, beta, and rc distinct numeric ranges.
if ($Version -match '^(?<base>\d+\.\d+\.\d+)-(?<channel>alpha|beta|rc)\.(?<sequence>\d+)(?:\+[0-9A-Za-z.-]+)?$') {
    [long]$Sequence = $Matches.sequence
    $Channel = $Matches.channel
    $Offset = switch ($Channel) {
        "alpha" { 0 }
        "beta" { 20000 }
        "rc" { 40000 }
    }
    $SequenceLimit = switch ($Channel) {
        "alpha" { 19999 }
        "beta" { 19999 }
        "rc" { 25535 }
    }
    if ($Sequence -gt $SequenceLimit) {
        throw "The $Channel prerelease sequence must not exceed $SequenceLimit for MSI packaging."
    }
    $MsiVersion = "$($Matches.base)-$($Offset + $Sequence)"
}
elseif ($Version -notmatch '^\d+\.\d+\.\d+(?:-\d+)?(?:\+\d+)?$') {
    throw "MSI packaging supports numeric versions and alpha.N, beta.N, or rc.N prerelease versions; got '$Version'."
}

$Installer = Join-Path $ReleaseDir "Zeedle_${Version}_${PackageArch}.msi"
$StagingDir = Join-Path $ReleaseDir ".msi-staging-$([guid]::NewGuid().ToString('N'))"
$StagedInstaller = Join-Path $StagingDir "Zeedle_${Version}_${PackageArch}.msi"
$VersionedConfigPath = Join-Path $PSScriptRoot ".Packager.windows.$([guid]::NewGuid().ToString('N')).toml"
$BuildCompleted = $false

Push-Location $ProjectRoot
try {
    $ConfigContent = [IO.File]::ReadAllText($ConfigPath)
    $ConfigContent = [regex]::Replace($ConfigContent, '(?m)^resources\s*=\s*\[[^\r\n]*\]', "resources = [`"$ResourceGlob`"]")
    $ConfigPrefix = "version = `"$MsiVersion`"`r`n"
    if ($PackageArch -eq "arm64") {
        $ConfigContent += "`r`n[wix]`r`ntemplate = `"Packager.windows.arm64.wxs`"`r`n"
    }
    $ConfigWithVersion = "$ConfigPrefix$ConfigContent"
    [IO.File]::WriteAllText($VersionedConfigPath, $ConfigWithVersion, [Text.UTF8Encoding]::new($false))
    New-Item -ItemType Directory -Path $StagingDir -Force | Out-Null

    Write-Host "Building Zeedle $Version MSI installer..."
    & cargo packager --config $VersionedConfigPath --out-dir $StagingDir --formats wix
    if ($LASTEXITCODE -ne 0) {
        throw "cargo packager failed with exit code $LASTEXITCODE"
    }

    $GeneratedInstallers = @(Get-ChildItem -LiteralPath $StagingDir -Filter "*.msi" -File)
    if ($GeneratedInstallers.Count -ne 1) {
        throw "Expected exactly one MSI in $StagingDir, found $($GeneratedInstallers.Count)"
    }
    if ($GeneratedInstallers[0].Length -le 0) {
        throw "Generated MSI is empty: $($GeneratedInstallers[0].FullName)"
    }
    if ($GeneratedInstallers[0].FullName -ne $StagedInstaller) {
        Move-Item -LiteralPath $GeneratedInstallers[0].FullName -Destination $StagedInstaller -Force
    }
    $BuildCompleted = $true

    Move-Item -LiteralPath $StagedInstaller -Destination $Installer -Force

    if (-not (Test-Path -LiteralPath $Installer -PathType Leaf)) {
        throw "Installer move failed: $Installer"
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
                Write-Warning "MSI could not be moved; staged artifact remains at $StagedInstaller"
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
