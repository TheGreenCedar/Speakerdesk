param(
    [Parameter(Mandatory=$true)][string]$Artifacts,
    [Parameter(Mandatory=$true)][string]$OrtRoot,
    [Parameter(Mandatory=$true)][string]$DirectMLRoot,
    [int]$Adapter = 0
)
$ErrorActionPreference = 'Stop'
$Artifacts = (Resolve-Path $Artifacts).Path
$OrtRoot = (Resolve-Path $OrtRoot).Path
$DirectMLRoot = (Resolve-Path $DirectMLRoot).Path
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw 'Run from an x64 Visual Studio Native Tools PowerShell, or load VsDevCmd x64 environment first.'
}
$provenance = Get-Content (Join-Path $Artifacts 'provenance.json') -Raw | ConvertFrom-Json
foreach ($property in $provenance.outputs.PSObject.Properties) {
    $actual = (Get-FileHash (Join-Path $Artifacts $property.Name) -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $property.Value) { throw "Artifact hash mismatch: $($property.Name)" }
}
$runtimeFiles = @(
    @{ Source = (Join-Path $OrtRoot 'runtimes/win-x64/native/onnxruntime.dll'); Hash = 'e7eedec6a6f26dc39dc948276a75ef6d2bee3fff944d874ceed0bbd3b97bff40' },
    @{ Source = (Join-Path $OrtRoot 'runtimes/win-x64/native/onnxruntime_providers_shared.dll'); Hash = '265c8daf29637cb259cac8be9f08f2cd45f3883f0f0e4949cbfddd5b4cbec3b6' },
    @{ Source = (Join-Path $DirectMLRoot 'bin/x64-win/DirectML.dll'); Hash = '9c9e6d822561c6c41b90e6994b3e8857cf1d66dbfb1e0c4c799c7c89b4e92da1' }
)
$runtimeEvidence = @()
foreach ($item in $runtimeFiles) {
    $actual = (Get-FileHash $item.Source -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $item.Hash) { throw "Runtime hash mismatch: $($item.Source)" }
    $target = Join-Path $Artifacts (Split-Path $item.Source -Leaf)
    Copy-Item $item.Source $target -Force
    $runtimeEvidence += @{ path = $target; sha256 = $actual }
}
$runtimeEvidence | ConvertTo-Json | Set-Content (Join-Path $Artifacts 'runtime-provenance.json')
$binary = Join-Path $Artifacts 'silero_ort_probe.exe'
& cl.exe /nologo /std:c++17 /EHsc /O2 /DNOMINMAX /DWIN32_LEAN_AND_MEAN `
    "/I$OrtRoot/build/native/include" "/I$DirectMLRoot/include" `
    (Join-Path $PSScriptRoot 'silero_ort_probe.cpp') "/Fe:$binary" `
    "/Fo:$Artifacts/silero_ort_probe.obj" /link `
    (Join-Path $OrtRoot 'runtimes/win-x64/native/onnxruntime.lib') dxgi.lib
if ($LASTEXITCODE -ne 0) { throw "Probe compilation failed: $LASTEXITCODE" }
$savedErrorPreference = $ErrorActionPreference
try {
    # Windows PowerShell 5.1 turns redirected native stderr into ErrorRecords.
    $ErrorActionPreference = 'Continue'
    & $binary $Artifacts $Adapter 2>&1 | ForEach-Object { "$_" } |
        Tee-Object (Join-Path $Artifacts 'run.log') -ErrorAction Stop
    $probeExitCode = $LASTEXITCODE
} finally {
    $ErrorActionPreference = $savedErrorPreference
}
if ($probeExitCode -ne 0) { throw "Probe execution failed: $probeExitCode; retain run.log and profiles" }
$profiles = @{}
foreach ($provider in @('cpu', 'dml')) {
    $profile = Get-ChildItem $Artifacts -Filter "$provider-profile*.json" |
        Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
    if (-not $profile) { throw "Missing $provider execution profile" }
    $events = Get-Content $profile.FullName -Raw | ConvertFrom-Json
    $executed = @($events | Where-Object { $_.cat -eq 'Node' -and $_.args.provider })
    $groups = @($executed | Group-Object { $_.args.provider } | ForEach-Object {
        @{
            provider = $_.Name
            executed_events = $_.Count
            summed_profile_duration_us = ($_.Group | Measure-Object dur -Sum).Sum
            operations = @($_.Group.args.op_name | Sort-Object -Unique)
        }
    })
    $profiles[$provider] = @{ profile = $profile.FullName; sha256 = (Get-FileHash $profile.FullName).Hash; executed_nodes = $groups }
}
$profiles | ConvertTo-Json -Depth 8 | Tee-Object (Join-Path $Artifacts 'placement.json')
$dmlEvents = @($profiles.dml.executed_nodes | Where-Object { $_.provider -eq 'DmlExecutionProvider' })
if (-not $dmlEvents -or $dmlEvents[0].executed_events -le 0) {
    throw 'No executed DirectML kernels in profile; accelerated feasibility remains unproven.'
}
Write-Output 'Silero numerical and executed DirectML placement checks passed. See run.log and placement.json.'
