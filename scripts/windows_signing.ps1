<# Build-only Azure Artifact Signing adapter. No login, IAM or publication. #>
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateSet('PrepareTools','PrepareMetadata','Check','Sign','VerifyPackage','FinalizeUpdater')][string]$Action,
    [string]$Path,
    [string]$Signature,
    [string]$Tools = $env:SPEAKERDESK_SIGNING_TOOLS,
    [string]$Metadata = $env:SPEAKERDESK_SIGNING_METADATA,
    [string]$Python = $env:SPEAKERDESK_SIGNING_PYTHON,
    [string]$Endpoint, [string]$Account, [string]$Profile, [string]$Correlation
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) { throw 'Artifact Signing requires Windows.' }
if (-not $Python) { $Python='python' }
$helper=Join-Path $PSScriptRoot 'windows_signing.py'
$contract=Get-Content (Join-Path $PSScriptRoot 'windows-signing.json') -Raw | ConvertFrom-Json
function Invoke-Helper([string[]]$Arguments) {
    $json = & $Python $helper @Arguments
    if($LASTEXITCODE -ne 0){throw 'Windows signing contract rejected the inputs.'}
    return (($json -join "`n") | ConvertFrom-Json)
}
function Write-FreshJson([string]$Destination,$Value) {
    $bytes=[Text.UTF8Encoding]::new($false).GetBytes(($Value | ConvertTo-Json -Depth 12)+"`n")
    $stream=[IO.File]::Open($Destination,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write)
    try{$stream.Write($bytes,0,$bytes.Length)}finally{$stream.Dispose()}
}
if($Action -eq 'PrepareMetadata') {
    $value=Invoke-Helper @('metadata','--endpoint',$Endpoint,'--account',$Account,'--profile',$Profile,'--correlation',$Correlation)
    Write-FreshJson ([IO.Path]::GetFullPath($Path)) $value
    return
}
if($Action -eq 'PrepareTools') {
    if(Test-Path -LiteralPath $Tools){throw 'Use a fresh signer-tools directory.'}
    New-Item -ItemType Directory -Path $Tools | Out-Null
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    foreach($package in $contract.client_packages) {
        $archive=Join-Path $Tools ($package.id+'.nupkg')
        Invoke-WebRequest -UseBasicParsing -Uri $package.url -OutFile $archive
        if((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -cne $package.sha256){throw 'Signer package digest mismatch.'}
        [IO.Compression.ZipFile]::ExtractToDirectory($archive,(Join-Path $Tools $package.id))
    }
    Write-Output "Prepared pinned signer packages in $Tools. Check them with the metadata before signing."
    return
}
if($Action -eq 'FinalizeUpdater') {
    Invoke-Helper @('finalize-updater','--path',$Path,'--signature',$Signature,'--tools',$Tools)
    return
}
$context=Invoke-Helper @('check','--tools',$Tools,'--metadata',$Metadata)
if($Action -eq 'Check'){ $context; return }
foreach($name in @('SPEAKERDESK_SIGNING_STAGE','SPEAKERDESK_SIGNING_TARGET','SPEAKERDESK_SIGNING_TEMP','SPEAKERDESK_SIGNING_RECEIPTS','SPEAKERDESK_SIGNING_VERSION')) {
    if(-not [Environment]::GetEnvironmentVariable($name)){throw "Missing scoped build input $name"}
}
$target=[IO.Path]::GetFullPath($Path)
if($Action -eq 'VerifyPackage') {
    $inputs=@(Invoke-Helper @('package','--path',$target,'--stage',$env:SPEAKERDESK_SIGNING_STAGE,'--target',$env:SPEAKERDESK_SIGNING_TARGET))
    foreach($inputFile in $inputs) {
        if($inputFile.kind -eq 'hook_or_pinned'){continue}
        $signature=Get-AuthenticodeSignature -LiteralPath $inputFile.path
        if($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -cnotmatch '^CN=Microsoft (Corporation|Windows)(?:,|$)'){throw 'Packaged Microsoft input has an unexpected signature.'}
        & $context.signtool verify /pa /all /v $inputFile.path
        if($LASTEXITCODE -ne 0){throw 'Packaged Microsoft input failed native trust verification.'}
        if((Get-FileHash -LiteralPath $inputFile.path -Algorithm SHA256).Hash.ToLowerInvariant() -cne $inputFile.sha256){throw 'Packaged input changed during verification.'}
        $inputFile | Add-Member -NotePropertyName authenticode_status -NotePropertyValue ([string]$signature.Status)
        $inputFile | Add-Member -NotePropertyName publisher_subject -NotePropertyValue $signature.SignerCertificate.Subject
        $inputFile | Add-Member -NotePropertyName verified -NotePropertyValue $true
    }
    $appReceipt=Get-Content -LiteralPath (Join-Path $env:SPEAKERDESK_SIGNING_RECEIPTS 'app.json') -Raw | ConvertFrom-Json
    Write-FreshJson (Join-Path $env:SPEAKERDESK_SIGNING_RECEIPTS 'package.json') ([ordered]@{
        script_sha256=(Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant();
        post_build_inputs=$inputs;app_packaged_sha256=$appReceipt.sha256})
    return
}
$classification=Invoke-Helper @('classify','--path',$target,'--stage',$env:SPEAKERDESK_SIGNING_STAGE,
    '--target',$env:SPEAKERDESK_SIGNING_TARGET,'--temporary',$env:SPEAKERDESK_SIGNING_TEMP,'--version',$env:SPEAKERDESK_SIGNING_VERSION)
$before=(Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
$signature=Get-AuthenticodeSignature -LiteralPath $target
if($classification.disposition -eq 'owned_signed') {
    if($signature.Status -eq 'NotSigned') {
        & $context.signtool sign /v /fd SHA256 /tr $contract.timestamp_url /td SHA256 `
            /dlib $context.dlib /dmdf $Metadata /d 'Speakerdesk' $target
        if($LASTEXITCODE -ne 0){throw "Artifact Signing failed: $($classification.role)"}
    } elseif($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -cnotmatch '^CN=Albert Najjar(?:,|$)') {
        throw 'Refusing to replace an unexpected existing signature.'
    }
    $signature=Get-AuthenticodeSignature -LiteralPath $target
    if($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -cnotmatch '^CN=Albert Najjar(?:,|$)' -or $null -eq $signature.TimeStamperCertificate){throw 'Expected verified publisher and timestamp were not produced.'}
} elseif($classification.disposition -eq 'microsoft_preserved') {
    if($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -cnotmatch '^CN=Microsoft (Corporation|Windows)(?:,|$)'){throw 'Pinned Microsoft signature is not valid.'}
}
if($classification.disposition -ne 'nsis_preserved') {
    & $context.signtool verify /pa /all /v $target
    if($LASTEXITCODE -ne 0){throw 'SignTool trust verification failed.'}
}
$after=(Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
if($classification.disposition -ne 'owned_signed' -and $before -cne $after){throw 'Preserved upstream bytes changed.'}
$role=$classification.role
$receipt=[ordered]@{role=$role;disposition=$classification.disposition;path=$target;input_sha256=$before;sha256=$after;
    authenticode_status=[string]$signature.Status;publisher_subject=$(if($signature.SignerCertificate){$signature.SignerCertificate.Subject}else{$null});
    timestamped=($null -ne $signature.TimeStamperCertificate);metadata_sha256=(Get-FileHash -LiteralPath $Metadata -Algorithm SHA256).Hash.ToLowerInvariant()}
$receiptPath=Join-Path $env:SPEAKERDESK_SIGNING_RECEIPTS ($role+'.json')
if(Test-Path -LiteralPath $receiptPath) {
    $prior=Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
    if($prior.sha256 -cne $after){throw 'A signer role was requested twice with different bytes.'}
} else {
    if($classification.disposition -eq 'owned_signed') {
        $name=switch($role){'capture'{'speakerdesk-capture.exe'} 'echo'{'speakerdesk_echo.dll'} 'app'{'speakerdesk.exe'} 'uninstaller'{'uninstall.exe'} 'installer'{Split-Path -Leaf $target}}
        Copy-Item -LiteralPath $target -Destination (Join-Path $env:SPEAKERDESK_SIGNING_RECEIPTS ('payloads/'+$name))
    }
    Write-FreshJson $receiptPath $receipt
}
Write-Output "Verified signing disposition: $role ($($classification.disposition))."
