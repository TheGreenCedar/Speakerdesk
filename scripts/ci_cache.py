"""Cache identities bind toolchains and tracked inputs, never user runtime state."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(root, paths, identity):
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
    for name in sorted(paths):
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or unsafe cache input: {name}")
        digest.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def keys(root, tracked, tools):
    def files(*prefixes):
        return [name for name in tracked if any(name.startswith(p) for p in prefixes)]

    locks = files('requirements-')
    environment = fingerprint(root, [], tools)
    prefix = f"speakerdesk-v1-{environment}"
    rust_inputs = files('desktop/src-tauri/')
    rust_contract = files('.cargo/', 'scripts/build_macos.sh', 'scripts/ci_cache.py')
    # sccache validates each rustc invocation, source, dependency and environment.
    # Version/lock/manifest changes must not discard unchanged dependency objects.
    compiler_prefix = f"{prefix}-compiler-v2-"
    return {
        'uv-key': f"{prefix}-uv-{fingerprint(root, locks, {})}",
        'uv-prefix': f"{prefix}-uv-",
        'cargo-key': f"{prefix}-cargo-{fingerprint(root, files('desktop/src-tauri/Cargo.lock'), {})}",
        'cargo-prefix': f"{prefix}-cargo-",
        'npm-key': f"{prefix}-npm-{fingerprint(root, files('desktop/package-lock.json'), {})}",
        'npm-prefix': f"{prefix}-npm-",
        'compiler-prefix': compiler_prefix,
        'compiler-key': compiler_prefix + fingerprint(root, rust_inputs + rust_contract, {}),
        # Whole components use exact inputs only. No prefix restore for executable outputs.
        'runtime-key': f"{prefix}-runtime-{fingerprint(root, files('speakerdesk/', 'packaging/', 'requirements-', 'scripts/setup.sh', 'scripts/build_macos.sh', 'scripts/ci_cache.py', 'scripts/component_cache.py'), {})}",
        'capture-key': f"{prefix}-capture-{fingerprint(root, files('desktop/capture/', 'desktop/src-tauri/Entitlements.plist', 'scripts/build_macos.sh', 'scripts/ci_cache.py', 'scripts/component_cache.py'), {})}",
    }


def source_changed(paths):
    # Keep the required Source checks job green on documentation-only PRs.
    return any(not (p.startswith('docs/') or p == 'README.md') for p in paths)


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def emit(values):
    print(json.dumps(values, indent=2))
    if os.getenv('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
            for name, value in values.items():
                output.write(f'{name}={value}\n')


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['size']:
        paths = [ROOT / '.cache/uv', ROOT / '.cache/npm', ROOT / '.cache/components',
                 Path(os.environ['SCCACHE_DIR']), Path.home() / '.cargo/registry',
                 Path.home() / '.cargo/git']
        total = sum(p.stat().st_size for directory in paths if directory.exists()
                    for p in directory.rglob('*') if p.is_file() and not p.is_symlink())
        emit({'cache-bytes': total, 'within-limit': str(total <= 2 * 1024**3).lower()})
    elif sys.argv[1:] == ['changes']:
        base = os.getenv('BASE_SHA', '')
        if not base or set(base) == {'0'}:
            changed = True
        else:
            names = command('git', 'diff', '--name-only', '-z', base, 'HEAD').split('\0')
            changed = source_changed(list(filter(None, names)))
        emit({'source': str(changed).lower()})
    else:
        tools = {
            'os': command('sw_vers', '-productVersion'), 'arch': command('uname', '-m'),
            'image': os.getenv('ImageVersion', 'local'), 'target': 'aarch64-apple-darwin',
            'deployment': '15.0', 'rust': command('rustc', '-Vv'),
            'xcode': command('xcodebuild', '-version'), 'swift': command('xcrun', 'swiftc', '--version'),
            'sdk': command('xcrun', '--show-sdk-version'), 'python': command('python3.13', '--version'),
            'uv': command('uv', '--version'), 'node': command('node', '--version'),
            'sccache': command('sccache', '--version'), 'profile': 'release-default-features-adhoc',
        }
        tracked = command('git', 'ls-files', '-z').split('\0')
        emit(keys(ROOT, list(filter(None, tracked)), tools))
