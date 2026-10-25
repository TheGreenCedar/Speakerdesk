"""Generate one app notice from locked native dependencies and pinned source notices."""
import argparse
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


def collect(metadata, destination, pins=None, extra=None, components=()):
    pins = pins or {}
    entries = {}
    for title, text in (extra or {}).items():
        entries.setdefault(text, []).append(title)
    missing = []
    # Separate lockfiles may enable different features of the same crate. Traverse
    # each component's own graph rather than overwriting one graph with the other.
    for component, root_name in [(metadata, 'speakerdesk'), *components]:
        packages = {package['id']: package for package in component['packages']}
        nodes = {node['id']: node for node in component['resolve']['nodes']}
        pending = [package['id'] for package in packages.values()
                   if package['name'] == root_name and package.get('source') is None]
        if len(pending) != 1:
            raise ValueError(f'Expected one native component package: {root_name}')
        seen = set()
        while pending:
            identifier = pending.pop()
            if identifier in seen:
                continue
            seen.add(identifier)
            pending.extend(dep['pkg'] for dep in nodes[identifier]['deps']
                           if any(kind['kind'] != 'dev' for kind in dep['dep_kinds']))
            if identifier in component['workspace_members']:
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
                    entries.setdefault(path.read_text(encoding='utf-8'), []).append(f'{title} / {path.name}')
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
        sections.append('\n' + '=' * 72 + '\n' + '\n'.join(sorted(set(titles))) + '\n' + '=' * 72 + '\n\n' + text)
    destination.write_text('\n'.join(sections), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', choices=('macos', 'windows'), default='macos')
    parser.add_argument('--capture-manifest', type=Path)
    parser.add_argument('--runtime-directory', type=Path)
    parser.add_argument('--destination', type=Path, default=ROOT / 'target/native/THIRD_PARTY_NOTICES.txt')
    args = parser.parse_args(argv)
    from setup_echo import sources
    sources()
    import setup_localvqe
    learned_source = setup_localvqe.sources()
    target = 'x86_64-pc-windows-msvc' if args.platform == 'windows' else 'aarch64-apple-darwin'
    metadata = json.loads(subprocess.check_output(
        ['cargo', 'metadata', '--locked', '--format-version', '1', '--filter-platform', target], cwd=ROOT))
    pins = json.loads((ROOT / 'packaging/native-license-sources.json').read_text(encoding='utf-8'))
    destination = args.destination
    extra = {'Speakerdesk': (ROOT / 'LICENSE').read_text(encoding='utf-8'),
             'Model and source attribution': (ROOT / 'packaging/THIRD_PARTY_NOTICES.md').read_text(encoding='utf-8')}
    for title, pin in pins['sources'].items():
        extra[f"{title} / {pin['url']} / SHA-256: {pin['sha256']}"] = fetch_text(pin, destination.parent / 'license-sources')
    source_notices = {
        'WebRTC echo component': '.cache/echo-source/src/LICENSE',
        'WebRTC patent grant': '.cache/echo-source/src/PATENTS',
        'WebRTC authors': '.cache/echo-source/src/AUTHORS',
        'Abseil echo support': '.cache/echo-source/src/third_party/abseil-cpp/LICENSE',
    }
    extra['LocalVQE v1.2 source and bundled weights (Apache-2.0)'] = (learned_source / 'LICENSE').read_text(encoding='utf-8')
    extra['GGML CPU support (MIT)'] = (setup_localvqe.CACHE / 'ggml/LICENSE').read_text(encoding='utf-8')
    components = []
    if args.platform == 'macos':
        source_notices.update({
            'Apple MLX 0.32.2': '.cache/native/mlx-0.32.2/LICENSE',
            'Apple MLX C binding': 'vendor/mlx-sys/src/mlx-c/LICENSE',
        })
    else:
        from fetch_windows_runtime import verify
        runtime_pin = json.loads((ROOT / 'crates/models/src/windows-runtime.json').read_text(encoding='utf-8'))
        runtime = args.runtime_directory or ROOT / '.cache/native' / f"windows-ml-{runtime_pin['version']}-x64"
        verify(runtime, runtime_pin['files'])
        for name in ('LICENSE.txt', 'ThirdPartyNotices.txt'):
            extra[f"Microsoft Windows ML {runtime_pin['version']} / {name}"] = (runtime / name).read_text(encoding='utf-8')
        manifest = args.capture_manifest or ROOT / 'desktop/capture/windows/Cargo.toml'
        capture = json.loads(subprocess.check_output([
            'cargo', 'metadata', '--locked', '--format-version', '1', '--filter-platform', target,
            '--manifest-path', str(manifest)], cwd=ROOT))
        components.append((capture, 'speakerdesk-capture-windows'))
    for title, relative in source_notices.items():
        extra[title] = (ROOT / relative).read_text(encoding='utf-8')
    ooura = (ROOT / '.cache/echo-source/src/common_audio/third_party/ooura/fft_size_128/ooura_fft.cc').read_text(encoding='utf-8')
    extra['Ooura FFT in WebRTC'] = ooura[:ooura.index('*/') + 2]
    collect(metadata, destination, pins['packages'], extra, components)
    print(destination)


if __name__ == '__main__':
    main()
