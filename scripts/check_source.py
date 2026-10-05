"""Check tracked payload boundaries, syntax and release version consistency."""
import ast
import json
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
required = {'packaging/overrides/mlx_audio/vad/models/__init__.py',
            'packaging/runtime.spec', 'packaging/sidecar.py',
            'packaging/THIRD_PARTY_NOTICES.txt', 'desktop/capture/MeetingCapture.swift',
            'desktop/capture/Info.plist', 'desktop/src-tauri/icons/icon.icns',
            'desktop/src-tauri/icons/icon.png', 'speakerdesk/redimnet2_artifact.json',
            'packaging/licenses/redimnet2/LICENSE'}
assert required.issubset(set(tracked)), f'Missing packaging source: {sorted(required-set(tracked))}'
prohibited_prefixes = ('models/', 'vendor/', '.venv', '.cache/', 'speakerdesk/data/',
                       'speakerdesk/fixtures/', 'fixtures/', 'evidence/', 'attachments/',
                       'release/', 'desktop/node_modules/', 'desktop/src-tauri/target/',
                       'desktop/src-tauri/binaries/')
prohibited_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.sqlite', '.safetensors',
                         '.dmg', '.zip', '.p12', '.p8', '.pem', '.key', '.pyc'}
for name in filter(None, tracked):
    path = root/name
    assert not name.startswith(prohibited_prefixes), f'Excluded path tracked: {name}'
    assert path.suffix not in prohibited_extensions, f'Excluded file tracked: {name}'
    assert not path.is_symlink(), f'Symlink tracked: {name}'
    if path.suffix == '.py':
        ast.parse(path.read_text(), filename=name)
    if path.suffix in {'.json', '.py', '.rs', '.swift', '.js', '.yml', '.sh', '.md', '.txt'}:
        content = path.read_text()
        assert not re.search(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{30,}', content), f'Credential-like literal: {name}'
versions = [json.loads((root/p).read_text())['version'] for p in
            ['desktop/package.json', 'desktop/package-lock.json', 'desktop/src-tauri/tauri.conf.json']]
versions.append(re.search(r'^version = "([^"]+)"', (root/'desktop/src-tauri/Cargo.toml').read_text(), re.M)[1])
assert len(set(versions)) == 1, f'Version mismatch: {versions}'
lock = (root/'desktop/src-tauri/Cargo.lock').read_text()
assert f'name = "speakerdesk"\nversion = "{versions[0]}"' in lock
print(f'Checked {len(list(filter(None, tracked)))} tracked files; version {versions[0]}; no excluded payload.')
