"""Collect license text for the locked native Mac dependency graph (build only)."""
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def license_paths(source):
    return {path for path in source.iterdir()
            if path.is_file() and path.name.upper().startswith(('LICENSE', 'LICENCE', 'COPYING', 'NOTICE'))} if source.is_dir() else set()


def collect(metadata, destination, notices=ROOT / 'packaging/licenses/Rust'):
    packages = {package['id']: package for package in metadata['packages']}
    nodes = {node['id']: node for node in metadata['resolve']['nodes']}
    pending = [package['id'] for package in packages.values()
               if package['name'] == 'speakerdesk' and package.get('source') is None]
    if len(pending) != 1:
        raise ValueError('Expected one native desktop package')
    seen = set()
    inventory = []
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
        if not paths:
            # Some published crates omit repository-level licenses. Retained originals
            # are version-specific and include their pinned upstream source in NOTICE.
            paths = license_paths(notices / f"{package['name']}-{package['version']}")
        if not paths:
            missing.append(f"{package['name']} {package['version']}")
            continue
        folder = destination / f"{package['name']}-{package['version']}"
        folder.mkdir(parents=True, exist_ok=True)
        for path in sorted(paths):
            if not path.is_file():
                raise ValueError(f'Missing declared license: {path}')
            shutil.copyfile(path, folder / path.name)
        inventory.append({'name': package['name'], 'version': package['version'],
                          'license': package.get('license'),
                          'files': sorted(path.name for path in paths)})
    if missing:
        raise ValueError('Missing license text: ' + ', '.join(sorted(missing)))
    (destination / 'packages.json').write_text(json.dumps(sorted(inventory, key=lambda p: (p['name'], p['version'])), indent=2) + '\n')


if __name__ == '__main__':
    metadata = json.loads(subprocess.check_output(
        ['cargo', 'metadata', '--locked', '--format-version', '1', '--filter-platform', 'aarch64-apple-darwin'], cwd=ROOT))
    destination = ROOT / 'target/native/licenses'
    if destination.is_symlink():
        raise ValueError('License staging directory must not be a symlink')
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    collect(metadata, destination)
