"""Generate one app notice from locked native dependencies and pinned source notices."""
import hashlib
import json
import subprocess
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def license_paths(source):
    return {path for path in source.iterdir()
            if path.is_file() and path.name.upper().startswith(('LICENSE', 'LICENCE', 'COPYING', 'NOTICE'))} if source.is_dir() else set()


def fetch_text(pin, cache):
    path = cache / pin['sha256']
    if path.is_symlink():
        raise ValueError(f'License cache must not be a symlink: {path}')
    if path.exists():
        data = path.read_bytes()
    else:
        with urlopen(pin['url'], timeout=30) as response:
            data = response.read(1024 * 1024 + 1)
    if len(data) > 1024 * 1024 or hashlib.sha256(data).hexdigest() != pin['sha256']:
        raise ValueError(f"License hash mismatch: {pin['url']}")
    cache.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data.decode('utf-8')


def collect(metadata, destination, pins=None, extra=None):
    pins = pins or {}
    packages = {package['id']: package for package in metadata['packages']}
    nodes = {node['id']: node for node in metadata['resolve']['nodes']}
    pending = [package['id'] for package in packages.values()
               if package['name'] == 'speakerdesk' and package.get('source') is None]
    if len(pending) != 1:
        raise ValueError('Expected one native desktop package')
    entries = {}
    for title, text in (extra or {}).items():
        entries.setdefault(text, []).append(title)
    seen = set()
    missing = []
    while pending:
        identifier = pending.pop()
        if identifier in seen:
            continue
        seen.add(identifier)
        pending.extend(dep['pkg'] for dep in nodes[identifier]['deps']
                       if any(kind['kind'] != 'dev' for kind in dep['dep_kinds']))
        if identifier in metadata['workspace_members']:
            continue
        package = packages[identifier]
        source = Path(package['manifest_path']).parent
        paths = license_paths(source)
        if package.get('license_file'):
            paths.add(source / package['license_file'])
        title = f"{package['name']} {package['version']} ({package.get('license')})"
        if paths:
            for path in sorted(paths):
                if not path.is_file():
                    raise ValueError(f'Missing declared license: {path}')
                entries.setdefault(path.read_text(), []).append(f'{title} / {path.name}')
        elif (pin := pins.get(f"{package['name']}-{package['version']}")):
            title += f" / {pin['url']} / SHA-256: {pin['sha256']}"
            if pin.get('attribution'):
                title += '\n' + pin['attribution']
            entries.setdefault(fetch_text(pin, destination.parent / 'license-sources'), []).append(title)
        else:
            missing.append(f"{package['name']} {package['version']}")
    if missing:
        raise ValueError('Missing license text: ' + ', '.join(sorted(missing)))
    if destination.is_symlink():
        raise ValueError('App notice must not be a symlink')
    destination.parent.mkdir(parents=True, exist_ok=True)
    sections = ['Speakerdesk third-party notices\n']
    for text, titles in sorted(entries.items(), key=lambda entry: sorted(entry[1])):
        sections.append('\n' + '=' * 72 + '\n' + '\n'.join(sorted(titles)) + '\n' + '=' * 72 + '\n\n' + text)
    destination.write_text('\n'.join(sections))


if __name__ == '__main__':
    metadata = json.loads(subprocess.check_output(
        ['cargo', 'metadata', '--locked', '--format-version', '1', '--filter-platform', 'aarch64-apple-darwin'], cwd=ROOT))
    pins = json.loads((ROOT / 'packaging/native-license-sources.json').read_text())
    destination = ROOT / 'target/native/THIRD_PARTY_NOTICES.txt'
    extra = {'Model and source attribution': (ROOT / 'packaging/THIRD_PARTY_NOTICES.md').read_text()}
    for title, pin in pins['sources'].items():
        extra[f"{title} / {pin['url']} / SHA-256: {pin['sha256']}"] = fetch_text(pin, destination.parent / 'license-sources')
    for title, relative in {
        'Apple MLX 0.32.2': 'target/native/mlx-0.32.2/LICENSE',
        'Apple MLX C binding': 'vendor/mlx-sys/src/mlx-c/LICENSE',
        'WebRTC echo component': '.cache/echo-source/src/LICENSE',
        'WebRTC patent grant': '.cache/echo-source/src/PATENTS',
        'WebRTC authors': '.cache/echo-source/src/AUTHORS',
        'Abseil echo support': '.cache/echo-source/src/third_party/abseil-cpp/LICENSE',
    }.items():
        extra[title] = (ROOT / relative).read_text()
    ooura = (ROOT / '.cache/echo-source/src/common_audio/third_party/ooura/fft_size_128/ooura_fft.cc').read_text()
    extra['Ooura FFT in WebRTC'] = ooura[:ooura.index('*/') + 2]
    collect(metadata, destination, pins['packages'], extra)
    print(destination)
