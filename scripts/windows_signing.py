"""Build-only signing boundaries and observed NSIS payload inventory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / 'scripts/windows-signing.json').read_text())
EXCLUDED_CREDENTIALS = [
    'EnvironmentCredential', 'WorkloadIdentityCredential', 'ManagedIdentityCredential',
    'SharedTokenCacheCredential', 'VisualStudioCredential', 'VisualStudioCodeCredential',
    'AzurePowerShellCredential', 'AzureDeveloperCliCredential', 'InteractiveBrowserCredential',
]


def require(value, message):
    if not value:
        raise ValueError(message)


def regular(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        require(not item.is_symlink() and not (getattr(item.lstat(), 'st_file_attributes', 0) & 0x400),
                'Signing paths must not traverse links/reparse points')
    require(path.is_file() and path.stat().st_size > 0, 'Missing/nonregular signing input')
    return path.resolve()


def digest(path):
    with regular(path).open('rb') as source:
        return stream_digest(source)


def stream_digest(source):
    value = hashlib.sha256()
    for block in iter(lambda: source.read(1024 * 1024), b''):
        value.update(block)
    return value.hexdigest()


def metadata(endpoint, account, profile, correlation):
    require(re.fullmatch(r'https://[a-z0-9]+\.codesigning\.azure\.net/?', endpoint),
            'Use an exact regional Artifact Signing HTTPS endpoint')
    for value in (account, profile, correlation):
        require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value), 'Invalid signing metadata')
    return dict(Endpoint=endpoint.rstrip('/'), CodeSigningAccountName=account,
                CertificateProfileName=profile, CorrelationId=correlation,
                ExcludeCredentials=EXCLUDED_CREDENTIALS)


def validate_metadata(value):
    expected = metadata(value['Endpoint'], value['CodeSigningAccountName'],
                        value['CertificateProfileName'], value['CorrelationId'])
    require(value == expected, 'Metadata must use AzureCliCredential only, with no extra inputs')


def verify_tools(directory):
    """Verify executable dependencies against the pinned archives, not a mutable receipt."""
    directory = Path(directory).resolve()
    for package in CONTRACT['client_packages']:
        archive = directory / (package['id'] + '.nupkg')
        require(digest(archive) == package['sha256'], 'Signer package digest mismatch')
        extracted = directory / package['id']
        with zipfile.ZipFile(archive) as source:
            members = {item.filename: item for item in source.infolist() if not item.is_dir()}
            require(len(members) == len([i for i in source.infolist() if not i.is_dir()]),
                    'Duplicate signer package entries')
            actual = {p.relative_to(extracted).as_posix() for p in extracted.rglob('*') if p.is_file()}
            require(actual == set(members), 'Signer package file inventory changed')
            for name in members:
                require(not name.startswith('/') and '..' not in Path(name).parts and '\\' not in name,
                        'Unsafe signer package member')
                with source.open(name) as entry:
                    expected = stream_digest(entry)
                require(digest(extracted / name) == expected, 'Extracted signer tool changed: ' + name)
    return {key: str(regular(directory / CONTRACT[key])) for key in ('dlib', 'signtool')}


def classify(path, stage, target, temporary, version):
    path = regular(path)
    stage, target, temporary = (Path(p).resolve() for p in (stage, target, temporary))
    require(re.fullmatch(r'\d+\.\d+\.\d+', version), 'Invalid app version')
    owned = {
        stage / 'speakerdesk-capture.exe': 'capture', stage / 'speakerdesk_echo.dll': 'echo',
        target / 'speakerdesk.exe': 'app',
        target / 'bundle/nsis' / f'Speakerdesk_{version}_x64-setup.exe': 'installer',
    }
    if path in owned:
        role = owned[path]
        if role == 'app':
            data = path.read_bytes()
            require(data.count(b'__TAURI_BUNDLE_TYPE_VAR_NSS') == 1
                    and b'__TAURI_BUNDLE_TYPE_VAR_UNK' not in data,
                    'App must be observed after the NSIS bundle-type patch')
        return dict(role=role, disposition='owned_signed')
    runtime = json.loads((ROOT / 'crates/models/src/windows-runtime.json').read_text())['files']
    for name, pin in runtime.items():
        if name.endswith('.dll') and path == stage / 'runtime' / name:
            require(digest(path) == pin['sha256'], 'Pinned Microsoft runtime changed')
            return dict(role='runtime-' + name, disposition='microsoft_preserved')
    for name, pin in CONTRACT['nsis_plugins'].items():
        if path == target / 'nsis/x64/Plugins/x86-unicode' / name:
            require(digest(path) == pin, 'Pinned NSIS plugin changed')
            return dict(role='plugin-' + Path(name).name, disposition='nsis_preserved')
    # NSIS !uninstfinalize receives its generated temporary PE. TEMP is a fresh,
    # build-owned directory; this is the sole generated-file exception.
    if path.parent == temporary and path.suffix.lower() in ('.tmp', '.exe'):
        data = path.read_bytes()
        require(len(data) >= 64 and data[:2] == b'MZ', 'Generated uninstaller is not PE')
        offset = struct.unpack_from('<I', data, 60)[0]
        require(data[offset:offset + 4] == b'PE\0\0', 'Generated uninstaller is not PE')
        return dict(role='uninstaller', disposition='owned_signed')
    raise ValueError('Signing request outside exact build allowlist: ' + str(path))


def package_inputs(script, stage, target):
    """Read the locked NSIS template's File inputs, failing on new source/destination classes.

    VC redist and WebView2 bypass Tauri's custom signing hook. They must receive
    separate native Microsoft trust verification, never our publisher signature.
    """
    stage, target = Path(stage).resolve(), Path(target).resolve()
    script = regular(script)
    require(script == target / 'nsis/x64/installer.nsi', 'Unexpected generated NSIS script')
    lines = script.read_text(encoding='utf-8-sig').splitlines()
    defines = dict(re.findall(r'^\s*!define\s+(\w+)\s+"([^"$]*)"\s*$', '\n'.join(lines), re.M))
    require(defines.get('INSTALLWEBVIEW2MODE') == 'embedBootstrapper', 'Unexpected WebView2 mode')
    runtime = json.loads((ROOT / 'crates/models/src/windows-runtime.json').read_text())['files']
    expected = {'speakerdesk.exe': target / 'speakerdesk.exe'}
    expected.update({name: stage / name for name in ('speakerdesk-capture.exe',
        'speakerdesk-capture.exe.capability.json', 'speakerdesk_echo.dll', 'THIRD_PARTY_NOTICES.txt')})
    expected.update({'runtime/' + name: stage / 'runtime' / name for name in runtime})
    result, seen = [], set()
    for line in lines:
        if not re.match(r'^\s*File(?:\s|$)', line):
            continue
        quoted = re.findall(r'"([^"\n]*)"', line)
        require(quoted, 'Unrecognized generated NSIS File instruction')
        raw = quoted[-1]
        # The unchanged template retains this inactive branch with no definition.
        if raw == '${WEBVIEW2INSTALLERPATH}' and not defines.get('WEBVIEW2INSTALLERPATH'):
            continue
        raw = re.sub(r'\$\{(\w+)\}', lambda m: defines.get(m[1], m[0]), raw)
        require('$' not in raw, 'Unresolved or escaped NSIS source path')
        source = regular(script.parent / raw)
        oname = [v[len('/oname='):] for v in quoted[:-1] if v.startswith('/oname=')]
        name = oname[0].replace('\\', '/') if oname else source.name
        require(len(oname) <= 1 and name not in seen, 'Ambiguous packaged destination')
        seen.add(name)
        if name in expected:
            require(source == expected[name], 'Packaged source differs from signing inventory')
            kind = 'hook_or_pinned'
        elif name == '$TEMP/MicrosoftEdgeWebview2Setup.exe':
            require(source.name == 'MicrosoftEdgeWebview2Setup.exe'
                    and str(source) == defines.get('WEBVIEW2BOOTSTRAPPERPATH'), 'Unexpected bootstrapper input')
            kind = 'microsoft_bootstrapper'
        else:
            require(source.name == name and re.fullmatch(
                r'(?:msvcp140(?:_1|_2|_atomic_wait|_codecvt_ids)?|vcruntime140(?:_1|_threads)?|concrt140|vccorlib140)\.dll', name, re.I)
                and re.fullmatch(r'Microsoft\.VC\d+\.CRT', source.parent.name)
                and source.parent.parent.name == 'x64', 'Unknown packaged input: ' + name)
            kind = 'microsoft_vc_runtime'
        result.append(dict(path=str(source), name=name, kind=kind, sha256=digest(source)))
    require(set(expected) <= seen and any(v['kind'] == 'microsoft_bootstrapper' for v in result),
            'Incomplete generated payload inventory')
    return result


def observed_payloads(receipts, stage, target, temporary, version):
    receipts, stage = Path(receipts), Path(stage)
    required = {'capture': 'speakerdesk-capture.exe', 'echo': 'speakerdesk_echo.dll',
                'app': 'speakerdesk.exe', 'uninstaller': 'uninstall.exe',
                'installer': f'Speakerdesk_{version}_x64-setup.exe'}
    result = {}
    for role, name in required.items():
        value = json.loads((receipts / (role + '.json')).read_text(encoding='utf-8-sig'))
        payload = receipts / 'payloads' / name
        require(value['role'] == role and value['disposition'] == 'owned_signed'
                and value['authenticode_status'] == 'Valid' and value['timestamped'] is True
                and re.match(r'^CN=Albert Najjar(?:,|$)', value['publisher_subject']),
                'Missing verified publisher/timestamp for ' + role)
        require(digest(payload) == value['sha256'], 'Observed signed payload changed: ' + role)
        if role in ('capture', 'echo'):
            require(digest(stage / name) == value['sha256'], 'Staged owned payload changed')
        if role == 'installer':
            require(digest(Path(target) / 'bundle/nsis' / name) == value['sha256'], 'Installer changed after hook')
        if role == 'app':
            require(classify(payload, stage, receipts / 'payloads', temporary, version)['role'] == 'app',
                    'Invalid observed app')
            restored = regular(Path(target) / 'speakerdesk.exe').read_bytes()
            require(restored.count(b'__TAURI_BUNDLE_TYPE_VAR_UNK') == 1
                    and b'__TAURI_BUNDLE_TYPE_VAR_NSS' not in restored
                    and hashlib.sha256(restored.replace(b'__TAURI_BUNDLE_TYPE_VAR_UNK',
                        b'__TAURI_BUNDLE_TYPE_VAR_NSS')).hexdigest() == value['input_sha256'],
                    'Loose app was not restored to the exact pre-signing input')
        if role != 'installer':
            result[name] = value['sha256']
    capability = json.loads((stage / 'speakerdesk-capture.exe.capability.json').read_text())
    require(capability['helper_sha256'] == result['speakerdesk-capture.exe'],
            'Capture capability receipt does not bind signed bytes')
    runtime = json.loads((ROOT / 'crates/models/src/windows-runtime.json').read_text())['files']
    for name, pin in runtime.items():
        require(digest(stage / 'runtime' / name) == pin['sha256'], 'Runtime was modified during bundling')
        result['runtime/' + name] = pin['sha256']
    for name in ('speakerdesk-capture.exe.capability.json', 'THIRD_PARTY_NOTICES.txt'):
        result[name] = digest(stage / name)
    package = json.loads(regular(receipts / 'package.json').read_text(encoding='utf-8-sig'))
    require(package['script_sha256'] == digest(Path(target) / 'nsis/x64/installer.nsi'),
            'Generated NSIS script changed after verification')
    expected_package = package_inputs(Path(target) / 'nsis/x64/installer.nsi', stage, target)
    require([{k: entry[k] for k in ('path', 'name', 'kind', 'sha256')} for entry in package['post_build_inputs']]
            == expected_package, 'Packaged inventory does not match generated NSIS script')
    require(package['app_packaged_sha256'] == result['speakerdesk.exe'], 'Package app hash does not identify hook payload')
    for entry in package['post_build_inputs']:
        require(digest(entry['path']) == entry['sha256'], 'Packaged source changed after verification')
        if entry['kind'] != 'hook_or_pinned':
            require(entry['authenticode_status'] == 'Valid' and entry['verified'] is True,
                    'Unverified Microsoft packaged input')
            require(re.match(r'^CN=Microsoft (Corporation|Windows|Windows Publisher)(?:,|$)', entry['publisher_subject']),
                    'Unexpected packaged publisher')
        if entry['kind'] == 'microsoft_vc_runtime':
            result[entry['name']] = entry['sha256']
    return result


def updater_configuration():
    """The implicit platform overlay must preserve the verifier's base policy."""
    base = regular(ROOT / 'desktop/src-tauri/tauri.conf.json').read_bytes()
    windows = regular(ROOT / 'desktop/src-tauri/tauri.windows.conf.json').read_bytes()
    overlay = json.loads(windows)
    require('version' not in overlay, 'Windows overlay must not override the updater app version')
    plugins = overlay.get('plugins', {})
    require(isinstance(plugins, dict), 'Windows overlay must not remove the updater policy')
    updater = plugins.get('updater', {})
    require(isinstance(updater, dict) and not {'pubkey', 'requireSignedVersion', 'allowDowngrades'} & updater.keys(),
            'Windows overlay must not override the compiled updater key/version policy')
    return dict(version=json.loads(base)['version'], configuration_sha256=hashlib.sha256(base).hexdigest(),
                windows_configuration_sha256=hashlib.sha256(windows).hexdigest())


def finalize_updater(candidate, signature, tools):
    """Verify public returned bytes; the manifest is the finalization commit point."""
    manifest_path = regular(Path(candidate) / 'artifact-manifest.json')
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    config_path = regular(ROOT / 'desktop/src-tauri/tauri.conf.json')
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    configuration = updater_configuration()
    require(manifest['updater_signature']['configuration_sha256'] == configuration['configuration_sha256']
            and manifest['updater_signature']['windows_configuration_sha256'] == configuration['windows_configuration_sha256'],
            'Trusted updater configuration differs from the build-time pin')
    version = manifest['version']
    require(re.fullmatch(r'\d+\.\d+\.\d+', version) and version == config['version'],
            'Candidate version differs from the trusted compiled updater configuration')
    require(manifest['platform'] == 'windows-x86_64' and manifest['public_ready'] is False
            and manifest['installed_payloads_observed'] is False and manifest['authenticode_status'] == 'Valid'
            and manifest['signing'] == dict(kind='azure-artifact-signing', payloads_observed_at_hook=True, receipts='signing'),
            'Expected an unpromoted Authenticode-signed Windows candidate')
    name = f'Speakerdesk_{version}_x64-setup.exe'
    pin = manifest['installer']
    require(pin['name'] == name and type(pin['bytes']) is int and 0 < pin['bytes'] <= 256 * 1024**2
            and re.fullmatch(r'[0-9a-f]{64}', pin['sha256']), 'Invalid candidate installer pin')
    installer = regular(manifest_path.parent / name)
    require(installer.stat().st_size == pin['bytes'] and digest(installer) == pin['sha256'],
            'Final Authenticode installer differs from its manifest')
    signature = regular(signature)
    require(signature.stat().st_size <= 8192, 'Updater signature exceeds its bounded size')
    signature_bytes = signature.read_bytes()
    signature_text = signature_bytes.decode('ascii').strip()
    require(re.fullmatch(r'[A-Za-z0-9+/]+={0,2}', signature_text), 'Invalid updater signature encoding')
    destination = manifest_path.parent / (name + '.sig')
    completed = dict(filename=name, signature_filename=destination.name, signature=signature_text,
                     verified=True, require_signed_version=True,
                     **{key: value for key, value in configuration.items() if key != 'version'})
    pending = dict(filename=name, signature_filename=destination.name, pending=True,
                   verified=False, require_signed_version=True,
                   **{key: value for key, value in configuration.items() if key != 'version'})
    require(manifest['updater_signature'] in (pending, completed), 'Candidate updater state differs')
    if destination.exists() or destination.is_symlink():
        require(regular(destination).read_bytes() == signature_bytes, 'Refusing a different existing updater signature')
    context = verify_tools(tools)
    # Keep signing credentials out of both verification subprocesses.
    environment = {key: value for key, value in os.environ.items() if key.upper() in {
        'PATH', 'HOME', 'USERPROFILE', 'SYSTEMROOT', 'COMSPEC', 'LOCALAPPDATA', 'APPDATA',
        'TEMP', 'TMP', 'PATHEXT', 'CARGO_HOME', 'RUSTUP_HOME', 'CARGO_TARGET_DIR', 'CARGO_BUILD_JOBS',
        'INCLUDE', 'LIB', 'LIBPATH', 'VSINSTALLDIR', 'VCINSTALLDIR', 'VCTOOLSINSTALLDIR', 'VCTOOLSVERSION',
        'VCTOOLSREDISTDIR', 'WINDOWSSDKDIR', 'WINDOWSSDKVERSION', 'WINDOWSSDKLIBVERSION',
        'WINDOWSSDKBINPATH', 'WINDOWSSDKVERBINPATH', 'UNIVERSALCRTSDKDIR', 'UCRTVERSION',
        'FRAMEWORKDIR', 'FRAMEWORKDIR32', 'FRAMEWORKVERSION', 'FRAMEWORKVERSION32', 'FRAMEWORK40VERSION',
        'VSCMD_ARG_TGT_ARCH', 'VSCMD_ARG_HOST_ARCH', 'VSCMD_VER', 'WINDOWSLIBPATH', 'EXTENSIONSDKDIR',
        'NETFXSDKDIR', 'EXTERNAL_INCLUDE', 'VCIDEINSTALLDIR', 'DEVENVDIR'}}
    environment.update(SPEAKERDESK_FINAL_INSTALLER=str(installer), SPEAKERDESK_FINAL_SIGNTOOL=context['signtool'])
    native = """$ErrorActionPreference='Stop'
$signature=Get-AuthenticodeSignature -LiteralPath $env:SPEAKERDESK_FINAL_INSTALLER
if($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -cnotmatch '^CN=Albert Najjar(?:,|$)' -or $null -eq $signature.TimeStamperCertificate){throw 'Expected verified publisher and timestamp.'}
& $env:SPEAKERDESK_FINAL_SIGNTOOL verify /pa /all /v $env:SPEAKERDESK_FINAL_INSTALLER
if($LASTEXITCODE -ne 0){throw 'Final installer failed native trust verification.'}
"""
    lock = manifest_path.parent / '.updater-finalize.lock'
    temporary_paths, published_signature = [], False
    lock_handle = lock.open('xb')
    try:
        for stage, command in (('Native publisher/timestamp/trust verification',
                                ['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', native]),
                               ('Updater payload/key/version verification',
                                ['cargo', 'run', '--offline', '--release', '--manifest-path',
                         str(ROOT / 'scripts/windows-updater-verifier/Cargo.toml'), '--',
                         str(installer), str(signature), str(config_path)])):
            try:
                subprocess.run(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=180)
            except subprocess.CalledProcessError as error:
                diagnostic = (error.stderr or error.stdout or b'').decode('utf-8', errors='replace')[-4096:]
                raise ValueError(stage + ' failed.\n' + diagnostic) from None
            except subprocess.TimeoutExpired:
                raise ValueError(stage + ' timed out') from None
            except OSError:
                raise ValueError(stage + ' tool could not start') from None
        require(manifest_path.read_bytes() == original and config_path.read_bytes() == config_bytes
                and updater_configuration() == configuration
                and digest(installer) == pin['sha256'] and signature.read_bytes() == signature_bytes,
                'Candidate inputs changed during final updater verification')
        require(not destination.exists() or regular(destination).read_bytes() == signature_bytes,
                'Refusing a different existing updater signature')
        manifest['updater_signature'] = completed
        for path, data in ((destination, signature_bytes),
                           (manifest_path, (json.dumps(manifest, indent=2) + '\n').encode())):
            if path == destination and destination.exists():
                continue
            if path == manifest_path and json.loads(original)['updater_signature'] == completed:
                continue
            with tempfile.NamedTemporaryFile(dir=manifest_path.parent, prefix='.updater-', delete=False) as output:
                temporary_path = Path(output.name); temporary_paths.append(temporary_path)
                output.write(data); output.flush(); os.fsync(output.fileno())
            if path == destination:
                os.link(temporary_path, destination)  # No replacement of any existing signature.
                published_signature = True
            else:
                require(manifest_path.read_bytes() == original, 'Candidate manifest changed before commit')
                os.replace(temporary_path, manifest_path)
        return manifest
    except BaseException:
        if published_signature and manifest_path.read_bytes() == original and destination.read_bytes() == signature_bytes:
            destination.unlink()
        raise
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)
        lock_handle.close()
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('metadata', 'check', 'classify', 'package', 'inventory', 'configuration', 'finalize-updater'))
    parser.add_argument('--path'); parser.add_argument('--tools'); parser.add_argument('--metadata')
    parser.add_argument('--stage'); parser.add_argument('--target'); parser.add_argument('--temporary')
    parser.add_argument('--version'); parser.add_argument('--receipts')
    parser.add_argument('--signature')
    parser.add_argument('--endpoint'); parser.add_argument('--account'); parser.add_argument('--profile'); parser.add_argument('--correlation')
    args = parser.parse_args()
    if args.action == 'metadata':
        result = metadata(args.endpoint, args.account, args.profile, args.correlation)
    elif args.action == 'check':
        validate_metadata(json.loads(regular(args.metadata).read_text(encoding='utf-8-sig')))
        result = verify_tools(args.tools)
    elif args.action == 'classify':
        result = classify(args.path, args.stage, args.target, args.temporary, args.version)
    elif args.action == 'package':
        result = package_inputs(args.path, args.stage, args.target)
    elif args.action == 'finalize-updater':
        result = finalize_updater(args.path, args.signature, args.tools)
    elif args.action == 'configuration':
        result = updater_configuration()
    else:
        result = observed_payloads(args.receipts, args.stage, args.target, args.temporary, args.version)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
