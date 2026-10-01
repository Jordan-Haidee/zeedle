$ErrorActionPreference = "Stop"

$ProjectRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$WixUrl = "https://github.com/wixtoolset/wix3/releases/download/wix3141rtm/wix314-binaries.zip"
$WixSha256 = "6ac824e1642d6f7277d0ed7ea09411a508f6116ba6fae0aa5f2c7daa2ff43d31"
$RequiredFiles = @(
    "candle.exe",
    "candle.exe.config",
    "darice.cub",
    "light.exe",
    "light.exe.config",
    "wconsole.dll",
    "winterop.dll",
    "wix.dll",
    "WixUIExtension.dll",
    "WixUtilExtension.dll"
)

if (-not $env:LOCALAPPDATA) {
    throw "LOCALAPPDATA is required to prepare WiX for cargo-packager."
}

$LocalAppData = [IO.Path]::GetFullPath($env:LOCALAPPDATA)
$CacheRoot = Join-Path $LocalAppData ".cargo-packager"
$WixToolsDir = Join-Path $CacheRoot "WixTools"
$TargetRoot = [IO.Path]::GetFullPath((Join-Path $ProjectRoot "target"))
$TempDir = Join-Path $TargetRoot ".wix314-$([guid]::NewGuid().ToString('N'))"
$ArchivePath = Join-Path $TempDir "wix314-binaries.zip"
$ExtractDir = Join-Path $TempDir "extracted"

$LocalAppDataPrefix = $LocalAppData.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
$ResolvedCacheRoot = [IO.Path]::GetFullPath($CacheRoot)
if (-not $ResolvedCacheRoot.StartsWith($LocalAppDataPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "WiX cache path must remain under LOCALAPPDATA: $ResolvedCacheRoot"
}
$TargetRootPrefix = $TargetRoot.TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
if (-not $TempDir.StartsWith($TargetRootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw "WiX temporary directory must remain under the project target directory: $TempDir"
}

New-Item -ItemType Directory -Path $TempDir -Force | Out-Null
try {
    Write-Host "Downloading WiX Toolset 3.14.1 for ARM64 MSI generation..."
    Invoke-WebRequest -Uri $WixUrl -OutFile $ArchivePath
    $Sha256 = [Security.Cryptography.SHA256]::Create()
    $ArchiveStream = [IO.File]::OpenRead($ArchivePath)
    try {
        $ActualSha256 = [BitConverter]::ToString($Sha256.ComputeHash($ArchiveStream)).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $ArchiveStream.Dispose()
        $Sha256.Dispose()
    }
    if ($ActualSha256 -ne $WixSha256) {
        throw "WiX archive SHA-256 mismatch: expected $WixSha256, got $ActualSha256"
    }

    Expand-Archive -LiteralPath $ArchivePath -DestinationPath $ExtractDir
    $CandleExe = Get-ChildItem -LiteralPath $ExtractDir -Filter "candle.exe" -File -Recurse | Select-Object -First 1
    if (-not $CandleExe) {
        throw "WiX archive does not contain candle.exe."
    }
    $WixSourceDir = $CandleExe.DirectoryName
    $MissingFiles = @($RequiredFiles | Where-Object { -not (Test-Path -LiteralPath (Join-Path $WixSourceDir $_) -PathType Leaf) })
    if ($MissingFiles.Count -gt 0) {
        throw "WiX archive is missing required cargo-packager files: $($MissingFiles -join ', ')"
    }

    New-Item -ItemType Directory -Path $CacheRoot -Force | Out-Null
    if (Test-Path -LiteralPath $WixToolsDir -PathType Container) {
        Remove-Item -LiteralPath $WixToolsDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $WixToolsDir -Force | Out-Null
    Copy-Item -Path (Join-Path $WixSourceDir "*") -Destination $WixToolsDir -Recurse -Force

    Write-Host "WiX Toolset 3.14.1 staged in cargo-packager cache for ARM64."
}
finally {
    if (Test-Path -LiteralPath $TempDir -PathType Container) {
        Remove-Item -LiteralPath $TempDir -Recurse -Force
    }
}
