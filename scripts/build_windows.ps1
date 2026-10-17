<#
Build an x64 NSIS candidate. -Sign opts into the pinned Azure Artifact Signing
adapter and final-byte updater signature; neither mode authorizes publication.
Model weights and build-only Python never enter the installer.
#>
[CmdletBinding()]
param(
    [string]$Python = $env:SPEAKERDESK_BUILD_PYTHON,
    [string]$TargetDirectory = $env:SPEAKERDESK_NATIVE_TARGET_DIR,
    [string]$RuntimeArchive,
    [string]$OutputDirectory,
    [switch]$Sign,
    [switch]$DeferUpdaterSignature,
    [string]$SigningTools,
    [string]$SigningMetadata
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($DeferUpdaterSignature -and -not $Sign) { throw '-DeferUpdaterSignature requires -Sign.' }
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
$signingEnvironment = @{}
foreach ($name in @('SPEAKERDESK_SIGNING_TOOLS','SPEAKERDESK_SIGNING_METADATA','SPEAKERDESK_SIGNING_PYTHON',
    'SPEAKERDESK_SIGNING_STAGE','SPEAKERDESK_SIGNING_TARGET','SPEAKERDESK_SIGNING_TEMP',
    'SPEAKERDESK_SIGNING_RECEIPTS','SPEAKERDESK_SIGNING_VERSION','TEMP','TMP')) {
    $signingEnvironment[$name] = [Environment]::GetEnvironmentVariable($name)
}
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
    $version = (Get-Content 'desktop/src-tauri/tauri.conf.json' -Raw | ConvertFrom-Json).version
    $updaterConfiguration = $null
    $signer = Join-Path $PSScriptRoot 'windows_signing.ps1'
    $signingRoot = $null
    if ($Sign) {
        $updaterConfiguration = (@(Invoke-Native $Python @('scripts/windows_signing.py', 'configuration')) -join "`n") | ConvertFrom-Json
        if (-not $SigningTools -or -not $SigningMetadata) {
            throw '-Sign requires pinned SigningTools and AzureCli-only SigningMetadata.'
        }
        if (-not $DeferUpdaterSignature -and -not $env:TAURI_SIGNING_PRIVATE_KEY) {
            throw '-Sign requires the protected Tauri updater key unless -DeferUpdaterSignature is explicit.'
        }
        $env:SPEAKERDESK_SIGNING_TOOLS = [IO.Path]::GetFullPath($SigningTools)
        $env:SPEAKERDESK_SIGNING_METADATA = [IO.Path]::GetFullPath($SigningMetadata)
        $env:SPEAKERDESK_SIGNING_PYTHON = (Get-Command $Python).Source
        & $signer -Action Check | Out-Null
        $signingRoot = Join-Path $taskRoot ('.cache/windows-signing/' + [Guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $signingRoot,(Join-Path $signingRoot 'payloads'),(Join-Path $signingRoot 'temp') | Out-Null
        $env:SPEAKERDESK_SIGNING_RECEIPTS = $signingRoot
        $env:SPEAKERDESK_SIGNING_STAGE = $stage
        $env:SPEAKERDESK_SIGNING_TARGET = Join-Path $TargetDirectory "$target/release"
        $env:SPEAKERDESK_SIGNING_TEMP = Join-Path $signingRoot 'temp'
        $env:SPEAKERDESK_SIGNING_VERSION = $version
        $hook = [ordered]@{bundle=@{windows=@{signCommand=@{cmd='powershell.exe';args=@('-NoProfile','-ExecutionPolicy','Bypass','-File',$signer,'-Action','Sign','-Path','%1')}}}}
        Write-Json (Join-Path $signingRoot 'tauri-signing.json') $hook
    }
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
    if ($Sign) { & $signer -Action Sign -Path $helper }
    # Bind admission to the final helper bytes, including Authenticode when requested.
    Write-Json "$helper.capability.json" ([ordered]@{
        helper_sha256 = File-Digest $helper
        export_contract = $description.export_contract
        capability_version = $description.capability_version
        progress_contract = $description.progress_contract
    })
    $echo = Join-Path $taskRoot 'desktop/capture/echo/speakerdesk_echo.dll'
    Assert-RegularFile $echo
    Copy-Item -LiteralPath $echo -Destination (Join-Path $stage 'speakerdesk_echo.dll') -Force
    $model = 'localvqe-v1.2-1.3M-f32.gguf'
    $modelPin = (Get-Content 'desktop/capture/echo/localvqe-lock.json' -Raw | ConvertFrom-Json).artifacts.$model
    $modelSource = Join-Path $taskRoot "desktop/capture/echo/$model"
    Assert-RegularFile $modelSource
    Copy-Item -LiteralPath $modelSource -Destination (Join-Path $stage $model) -Force
    if ((File-Digest (Join-Path $stage $model)) -ne $modelPin.sha256) { throw 'Staged capture model differs from its pin.' }
    # GGUF is pinned data, never an Authenticode signing input.
    if ($Sign) {
        & $signer -Action Sign -Path (Join-Path $stage 'speakerdesk_echo.dll')
        foreach ($name in $runtimeNames | Where-Object { $_.EndsWith('.dll') }) {
            & $signer -Action Sign -Path (Join-Path $runtimeStage $name)
        }
    }
    Invoke-Native $Python @('scripts/native_licenses.py', '--platform', 'windows',
        '--capture-manifest', 'desktop/capture/windows/Cargo.toml',
        '--runtime-directory', $runtimeDirectory, '--destination', (Join-Path $stage 'THIRD_PARTY_NOTICES.txt'))

    Push-Location (Join-Path $taskRoot 'desktop')
    try {
        Invoke-Native 'npm.cmd' @('ci')
        # Use the locked Tauri image converter; all generated icons stay in Windows staging.
        Invoke-Native node @('node_modules/@tauri-apps/cli/tauri.js', 'icon',
            'src-tauri/icons/icon.png', '--output', (Join-Path $stage 'icons'))
        $buildArguments = @('run', 'build', '--', '--target', $target, '--bundles', 'nsis')
        if ($Sign) {
            $installedCli = Get-Content 'node_modules/@tauri-apps/cli/package.json' -Raw | ConvertFrom-Json
            if ($installedCli.version -ne '2.12.1') { throw 'Signed hook inventory requires locked Tauri CLI 2.12.1.' }
            $buildArguments += @('--config', (Join-Path $signingRoot 'tauri-signing.json'))
            # NSIS's temporary uninstaller is the only allowed generated signing input.
            $env:TEMP = $env:SPEAKERDESK_SIGNING_TEMP
            $env:TMP = $env:SPEAKERDESK_SIGNING_TEMP
        }
        Invoke-Native 'npm.cmd' ($buildArguments + @('--', '--locked'))
    } finally { Pop-Location }

    if ($Sign) {
        if ((File-Digest (Join-Path $taskRoot 'desktop/src-tauri/tauri.conf.json')) -ne $updaterConfiguration.configuration_sha256 -or
            (File-Digest (Join-Path $taskRoot 'desktop/src-tauri/tauri.windows.conf.json')) -ne $updaterConfiguration.windows_configuration_sha256) {
            throw 'The compiled updater configuration changed during the signed build.'
        }
    }
    $version = (Get-Content 'desktop/src-tauri/tauri.conf.json' -Raw | ConvertFrom-Json).version
    $installer = Join-Path $TargetDirectory "$target/release/bundle/nsis/Speakerdesk_${version}_x64-setup.exe"
    Assert-RegularFile $installer
    $authenticodeStatus = [string](Get-AuthenticodeSignature -LiteralPath $installer).Status
    if (-not $Sign -and $authenticodeStatus -ne 'NotSigned') {
        throw 'Expected NSIS payload derivation is limited to unsigned candidates. Signed payloads require their own observed inventory.'
    }
    $executable = Join-Path $TargetDirectory "$target/release/speakerdesk.exe"
    Assert-RegularFile $executable
    $executableReceipt = $null
    $signedPayloads = $null
    $updater = $null
    if ($Sign) {
        if ($authenticodeStatus -ne 'Valid') { throw 'Final installer Authenticode is not valid.' }
        & $signer -Action VerifyPackage -Path (Join-Path $env:SPEAKERDESK_SIGNING_TARGET 'nsis/x64/installer.nsi')
        $signedPayloads = (@(Invoke-Native $Python @('scripts/windows_signing.py', 'inventory',
            '--receipts', $signingRoot, '--stage', $stage, '--target', $env:SPEAKERDESK_SIGNING_TARGET,
            '--temporary', $env:SPEAKERDESK_SIGNING_TEMP, '--version', $version)) -join "`n") | ConvertFrom-Json
        if ($DeferUpdaterSignature) {
            $updater = [ordered]@{filename=Split-Path -Leaf $installer; signature_filename=(Split-Path -Leaf $installer)+'.sig';
                pending=$true; verified=$false; require_signed_version=$true;
                configuration_sha256=$updaterConfiguration.configuration_sha256;
                windows_configuration_sha256=$updaterConfiguration.windows_configuration_sha256}
        } else {
            $finalDigest = File-Digest $installer
            Push-Location (Join-Path $taskRoot 'desktop')
            try { Invoke-Native node @('node_modules/@tauri-apps/cli/tauri.js', 'signer', 'sign', '--app-version', $version, $installer) }
            finally { Pop-Location }
            Invoke-Native cargo @('run', '--offline', '--release', '--manifest-path',
                'scripts/windows-updater-verifier/Cargo.toml', '--', $installer, "$installer.sig", 'desktop/src-tauri/tauri.conf.json')
            if ((File-Digest $installer) -ne $finalDigest) { throw 'Updater signing changed finalized Authenticode bytes.' }
            $updater = [ordered]@{filename=Split-Path -Leaf $installer; signature_filename=(Split-Path -Leaf $installer)+'.sig';
                signature=(Get-Content -LiteralPath "$installer.sig" -Raw).Trim(); verified=$true; require_signed_version=$true}
        }
    } else {
        $executableReceipt = (@(Invoke-Native $Python @('scripts/windows_payload.py', '--executable', $executable)) -join "`n") | ConvertFrom-Json
    }
    if (-not $OutputDirectory) {
        $candidate = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + '-' + [Guid]::NewGuid().ToString('N').Substring(0, 8)
        $OutputDirectory = Join-Path $taskRoot "release/windows/$version/$candidate"
    }
    $OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)
    if (Test-Path -LiteralPath $OutputDirectory) { throw "Candidate destination already exists: $OutputDirectory" }
    New-Item -ItemType Directory -Path $OutputDirectory | Out-Null
    $packagedInstaller = Join-Path $OutputDirectory (Split-Path -Leaf $installer)
    Copy-Item -LiteralPath $installer -Destination $packagedInstaller
    if ($Sign) {
        if (-not $DeferUpdaterSignature) { Copy-Item -LiteralPath "$installer.sig" -Destination "$packagedInstaller.sig" }
        $packagedReceipts = Join-Path $OutputDirectory 'signing'
        New-Item -ItemType Directory -Path $packagedReceipts | Out-Null
        Copy-Item -LiteralPath (Join-Path $signingRoot 'payloads') -Destination $packagedReceipts -Recurse
        Get-ChildItem -LiteralPath $signingRoot -Filter '*.json' | Copy-Item -Destination $packagedReceipts
    }
    $inputs = [ordered]@{}
    foreach ($name in @('speakerdesk-capture.exe', 'speakerdesk-capture.exe.capability.json',
        'speakerdesk_echo.dll', 'localvqe-v1.2-1.3M-f32.gguf', 'THIRD_PARTY_NOTICES.txt') + @($runtimeNames | ForEach-Object { "runtime/$_" })) {
        $inputs[$name] = File-Digest (Join-Path $stage $name)
    }
    $inputs['speakerdesk.exe'] = File-Digest $executable
    if (-not $Sign -and $inputs['speakerdesk.exe'] -ne $executableReceipt.input_sha256) {
        throw 'The executable changed after deriving its expected NSIS payload hash.'
    }
    # Tauri restores the loose UNK executable after bundling its NSS-patched copy.
    # These are expected payload digests; installation must verify them separately.
    $installedPayloads = [ordered]@{}
    foreach ($name in $inputs.Keys) { $installedPayloads[$name] = $inputs[$name] }
    if ($Sign) {
        foreach ($entry in $signedPayloads.PSObject.Properties) { $installedPayloads[$entry.Name] = $entry.Value }
    } else { $installedPayloads['speakerdesk.exe'] = $executableReceipt.expected_payload_sha256 }
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
        signing = $(if($Sign){[ordered]@{kind='azure-artifact-signing';payloads_observed_at_hook=$true;receipts='signing'}}else{$null})
        updater_signature = $updater
    })
    Write-Output "Windows candidate (Authenticode: $authenticodeStatus): $OutputDirectory"
} finally {
    $env:CARGO_TARGET_DIR = $previousTarget
    $env:CARGO_BUILD_JOBS = $previousJobs
    foreach ($name in $signingEnvironment.Keys) { [Environment]::SetEnvironmentVariable($name,$signingEnvironment[$name]) }
    Pop-Location
}
