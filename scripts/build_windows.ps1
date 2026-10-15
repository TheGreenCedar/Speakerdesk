<#
Build an unsigned x64 NSIS candidate using installed Rust, Visual Studio, Node and
build-only Python. Signing, updater signatures and publication follow review of
this exact candidate; model weights and Python never enter the installer.
#>
[CmdletBinding()]
param(
    [string]$Python = $env:SPEAKERDESK_BUILD_PYTHON,
    [string]$TargetDirectory = $env:SPEAKERDESK_NATIVE_TARGET_DIR,
    [string]$RuntimeArchive,
    [string]$OutputDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT -or
    $env:PROCESSOR_ARCHITECTURE -ne 'AMD64') {
    throw 'Build Speakerdesk Windows candidates from x64 Windows with the installed MSVC toolchain.'
}
$taskRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = 'python' }
foreach ($tool in @('cargo', 'rustc', 'node', 'npm.cmd', 'cmake', 'git', $Python)) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "Required build tool is unavailable: $tool. Use the installed Visual Studio x64 build environment."
    }
}
function Invoke-Native {
    param([string]$File, [string[]]$CommandArguments)
    & $File @CommandArguments
    if ($LASTEXITCODE -ne 0) { throw "$File failed with exit code $LASTEXITCODE" }
}
function Assert-RegularFile {
    param([string]$Path)
    $item = Get-Item -LiteralPath $Path -Force
    if ($item.PSIsContainer -or $item.Length -eq 0 -or
        ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Required build input must be a nonempty regular file: $Path"
    }
}
function Write-Json {
    param([string]$Path, $Value)
    [IO.File]::WriteAllText($Path, (($Value | ConvertTo-Json -Depth 20) + "`n"),
        [Text.UTF8Encoding]::new($false))
}
function File-Digest {
    param([string]$Path)
    Assert-RegularFile $Path
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

$previousTarget = $env:CARGO_TARGET_DIR
$previousJobs = $env:CARGO_BUILD_JOBS
Push-Location $taskRoot
try {
    if (-not $TargetDirectory) {
        $TargetDirectory = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { 'target/windows' }
    }
    $TargetDirectory = [IO.Path]::GetFullPath($TargetDirectory)
    $env:CARGO_TARGET_DIR = $TargetDirectory
    if (-not $env:CARGO_BUILD_JOBS) { $env:CARGO_BUILD_JOBS = '2' }
    $target = 'x86_64-pc-windows-msvc'
    $stage = Join-Path $taskRoot '.cache/windows-bundle'
    $runtimeStage = Join-Path $stage 'runtime'
    New-Item -ItemType Directory -Path $stage, $runtimeStage -Force | Out-Null
    foreach ($directory in @($stage, $runtimeStage)) {
        if ((Get-Item -LiteralPath $directory -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Windows packaging staging must not be a link: $directory"
        }
    }

    # These scripts verify pinned sources and reuse their project-local caches.
    Invoke-Native $Python @('scripts/setup_echo.py')
    $fetchArguments = @('scripts/fetch_windows_runtime.py')
    if ($RuntimeArchive) { $fetchArguments += @('--archive', [IO.Path]::GetFullPath($RuntimeArchive)) }
    $runtimeOutput = @(Invoke-Native $Python $fetchArguments)
    $runtimeDirectory = [IO.Path]::GetFullPath([string]$runtimeOutput[-1])
    $runtimePin = Get-Content 'crates/models/src/windows-runtime.json' -Raw | ConvertFrom-Json
    $runtimeNames = @($runtimePin.files.PSObject.Properties.Name)
    foreach ($entry in @(Get-ChildItem -LiteralPath $runtimeStage -Force)) {
        if ($entry.Name -notin $runtimeNames) {
            throw "Unexpected file in Windows runtime staging: $($entry.FullName)"
        }
    }
    foreach ($name in $runtimeNames) {
        $source = Join-Path $runtimeDirectory $name
        Assert-RegularFile $source
        $destination = Join-Path $runtimeStage $name
        Copy-Item -LiteralPath $source -Destination $destination -Force
        if ((File-Digest $destination) -ne $runtimePin.files.$name.sha256) {
            throw "Staged Windows runtime differs from its pin: $name"
        }
    }

    Invoke-Native cargo @('build', '--locked', '--release', '--target', $target,
        '--manifest-path', 'desktop/capture/windows/Cargo.toml')
    $helper = Join-Path $stage 'speakerdesk-capture.exe'
    Copy-Item -LiteralPath (Join-Path $TargetDirectory "$target/release/speakerdesk-capture.exe") -Destination $helper -Force
    $description = (@(Invoke-Native $helper @('--describe')) -join "`n") | ConvertFrom-Json
    if ($description.type -ne 'helper_check' -or $description.capture_started -ne $false -or
        $description.capability_version -ne 'windows_wasapi_shared_v1' -or
        $description.export_contract -ne 'raw_dual_mono_stereo16k_v1' -or
        $description.progress_contract -ne 'audio_queue_fence_v1') {
        throw 'The built Windows capture helper does not implement the reviewed raw capture/progress contract.'
    }
    Write-Json "$helper.capability.json" ([ordered]@{
        helper_sha256 = File-Digest $helper
        export_contract = $description.export_contract
        capability_version = $description.capability_version
        progress_contract = $description.progress_contract
    })
    $echo = Join-Path $taskRoot 'desktop/capture/echo/speakerdesk_echo.dll'
    Assert-RegularFile $echo
    Copy-Item -LiteralPath $echo -Destination (Join-Path $stage 'speakerdesk_echo.dll') -Force
    Invoke-Native $Python @('scripts/native_licenses.py', '--platform', 'windows',
        '--capture-manifest', 'desktop/capture/windows/Cargo.toml',
        '--runtime-directory', $runtimeDirectory, '--destination', (Join-Path $stage 'THIRD_PARTY_NOTICES.txt'))

    Push-Location (Join-Path $taskRoot 'desktop')
    try {
        Invoke-Native 'npm.cmd' @('ci')
        # Use the locked Tauri image converter; all generated icons stay in Windows staging.
        Invoke-Native node @('node_modules/@tauri-apps/cli/tauri.js', 'icon',
            'src-tauri/icons/icon.png', '--output', (Join-Path $stage 'icons'))
        Invoke-Native 'npm.cmd' @('run', 'build', '--', '--target', $target,
            '--bundles', 'nsis', '--', '--locked')
    } finally { Pop-Location }

    $version = (Get-Content 'desktop/src-tauri/tauri.conf.json' -Raw | ConvertFrom-Json).version
    $installer = Join-Path $TargetDirectory "$target/release/bundle/nsis/Speakerdesk_${version}_x64-setup.exe"
    Assert-RegularFile $installer
    $authenticodeStatus = [string](Get-AuthenticodeSignature -LiteralPath $installer).Status
    if ($authenticodeStatus -ne 'NotSigned') {
        throw 'Expected NSIS payload derivation is limited to unsigned candidates. Signed payloads require their own observed inventory.'
    }
    $executable = Join-Path $TargetDirectory "$target/release/speakerdesk.exe"
    Assert-RegularFile $executable
    $executableReceipt = (@(Invoke-Native $Python @('scripts/windows_payload.py',
        '--executable', $executable)) -join "`n") | ConvertFrom-Json
    if (-not $OutputDirectory) {
        $candidate = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
        $OutputDirectory = Join-Path $taskRoot "release/windows/$version/$candidate"
    }
    $OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
    if (Test-Path -LiteralPath $OutputDirectory) { throw "Candidate destination already exists: $OutputDirectory" }
    New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
    $packagedInstaller = Join-Path $OutputDirectory (Split-Path -Leaf $installer)
    Copy-Item -LiteralPath $installer -Destination $packagedInstaller
    $inputs = [ordered]@{}
    foreach ($name in @('speakerdesk-capture.exe', 'speakerdesk-capture.exe.capability.json',
        'speakerdesk_echo.dll', 'THIRD_PARTY_NOTICES.txt') + @($runtimeNames | ForEach-Object { "runtime/$_" })) {
        $inputs[$name] = File-Digest (Join-Path $stage $name)
    }
    $inputs['speakerdesk.exe'] = File-Digest $executable
    if ($inputs['speakerdesk.exe'] -ne $executableReceipt.input_sha256) {
        throw 'The executable changed after deriving its expected NSIS payload hash.'
    }
    # Tauri restores the loose UNK executable after bundling its NSS-patched copy.
    # These are expected payload digests; installation must verify them separately.
    $installedPayloads = [ordered]@{}
    foreach ($name in $inputs.Keys) { $installedPayloads[$name] = $inputs[$name] }
    $installedPayloads['speakerdesk.exe'] = $executableReceipt.expected_payload_sha256
    $sourceCommit = @(Invoke-Native git @('rev-parse', 'HEAD')) -join ''
    $sourceChanges = @(Invoke-Native git @('status', '--porcelain', '--untracked-files=normal'))
    Write-Json (Join-Path $OutputDirectory 'artifact-manifest.json') ([ordered]@{
        platform = 'windows-x86_64'
        version = $version
        source_commit = $sourceCommit
        source_dirty = ($sourceChanges.Count -ne 0)
        public_ready = $false
        authenticode_status = $authenticodeStatus
        installer = [ordered]@{ name = Split-Path -Leaf $packagedInstaller; sha256 = File-Digest $packagedInstaller; bytes = (Get-Item -LiteralPath $packagedInstaller).Length }
        inputs = $inputs
        installed_payloads = $installedPayloads
        installed_payloads_observed = $false
        executable_payload_derivation = $executableReceipt
        updater_signature = $null
    })
    Write-Output "Unsigned Windows candidate: $OutputDirectory"
} finally {
    $env:CARGO_TARGET_DIR = $previousTarget
    $env:CARGO_BUILD_JOBS = $previousJobs
    Pop-Location
}
