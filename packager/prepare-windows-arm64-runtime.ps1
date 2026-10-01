$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RuntimeDir = Join-Path $ProjectRoot "target\windows-arm64-runtime"
$RequiredDlls = @("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")
$VisualStudioRoot = $null

if ($env:VCToolsRedistDir -and (Test-Path -LiteralPath $env:VCToolsRedistDir -PathType Container)) {
    $VisualStudioRoot = $env:VCToolsRedistDir
}
else {
    $VsWherePath = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $VsWherePath -PathType Leaf) {
        $VisualStudioInstall = & $VsWherePath -latest -products '*' -property installationPath
        if ($LASTEXITCODE -eq 0 -and $VisualStudioInstall) {
            $VisualStudioRoot = Join-Path $VisualStudioInstall "VC\Redist\MSVC"
        }
    }
}

if (-not $VisualStudioRoot) {
    throw "Could not locate the Visual C++ redistributable directory for ARM64."
}

$RuntimeCandidates = @()
$DirectCandidate = Join-Path $VisualStudioRoot "arm64\Microsoft.VC143.CRT"
if (Test-Path -LiteralPath $DirectCandidate -PathType Container) {
    $RuntimeCandidates += Get-Item -LiteralPath $DirectCandidate
}
$RuntimeCandidates += @(Get-ChildItem -LiteralPath $VisualStudioRoot -Directory -ErrorAction SilentlyContinue |
    ForEach-Object {
        $Candidate = Join-Path $_.FullName "arm64\Microsoft.VC143.CRT"
        if (Test-Path -LiteralPath $Candidate -PathType Container) {
            Get-Item -LiteralPath $Candidate
        }
    })
$RuntimeCandidates = @($RuntimeCandidates | Sort-Object { [version]($_.Parent.Parent.Name) } -Descending)

$RuntimeSource = $RuntimeCandidates | Select-Object -First 1
if (-not $RuntimeSource) {
    throw "Could not find the ARM64 Microsoft.VC143.CRT runtime under $VisualStudioRoot."
}

$MissingDlls = @($RequiredDlls | Where-Object { -not (Test-Path -LiteralPath (Join-Path $RuntimeSource.FullName $_) -PathType Leaf) })
if ($MissingDlls.Count -gt 0) {
    throw "The ARM64 Visual C++ runtime is missing required DLLs: $($MissingDlls -join ', ')"
}

if (Test-Path -LiteralPath $RuntimeDir -PathType Container) {
    Remove-Item -LiteralPath $RuntimeDir -Recurse -Force
}
New-Item -ItemType Directory -Path $RuntimeDir -Force | Out-Null
Copy-Item -Path (Join-Path $RuntimeSource.FullName "*.dll") -Destination $RuntimeDir

Write-Host "Staged ARM64 Visual C++ runtime from $($RuntimeSource.FullName)"
