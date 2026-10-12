"""Cache identities bind toolchains and tracked inputs, never user runtime state."""
import hashlib
import json
import os
import platform
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

    environment = fingerprint(root, [], tools)
    prefix = f"speakerdesk-native-v1-{environment}"
    rust_inputs = files('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml', 'crates/', 'vendor/mlx-sys/', 'desktop/src-tauri/', 'speakerdesk/static/', 'speakerdesk/templates/')
    rust_contract = files('.cargo/', 'rust-toolchain.toml', '.github/workflows/',
                          'vendor/mlx-sys/', 'speakerdesk/',
                          'desktop/src-tauri/tauri.conf.json', 'desktop/src-tauri/Info.plist',
                          'desktop/src-tauri/Entitlements.plist', 'desktop/capture/echo/',
                          'scripts/build_macos.sh', 'scripts/fetch_mlx_runtime.sh',
                          'scripts/native_licenses.py', 'scripts/setup_echo.py',
                          'scripts/echo_compile_guard.py', 'scripts/ci_cache.py')
    # sccache validates each rustc invocation, source, dependency and environment.
    # Version/lock/manifest changes must not discard unchanged dependency objects.
    compiler_tools = {name: value for name, value in tools.items() if name not in {'python', 'node', 'profile'}}
    compiler_prefix = f"speakerdesk-compiler-v3-{fingerprint(root, rust_contract, compiler_tools)}-"
    dependency_contract = files('vendor/mlx-sys/', 'scripts/fetch_mlx_runtime.sh',
                                'desktop/capture/echo/', 'scripts/setup_echo.py', 'scripts/echo_compile_guard.py')
    dependency_tools = {name: value for name, value in tools.items() if name not in {'python', 'node'}}
    return {
        'dependency-key': f"speakerdesk-dependencies-v1-{fingerprint(root, dependency_contract, dependency_tools)}",
        'cargo-key': f"{prefix}-cargo-{fingerprint(root, files('Cargo.lock'), {})}",
        'cargo-prefix': f"{prefix}-cargo-",
        'npm-key': f"{prefix}-npm-{fingerprint(root, files('desktop/package-lock.json'), {})}",
        'npm-prefix': f"{prefix}-npm-",
        'compiler-prefix': compiler_prefix,
        'compiler-key': compiler_prefix + fingerprint(root, rust_inputs, {'profile': tools.get('profile', '')}),
        # Whole components use exact inputs only. No prefix restore for executable outputs.
        'capture-key': f"{prefix}-capture-{fingerprint(root, files('desktop/capture/', 'desktop/src-tauri/Entitlements.plist', 'scripts/build_macos.sh', 'scripts/ci_cache.py', 'scripts/component_cache.py'), {})}",
        'echo-key': f"{prefix}-echo-{fingerprint(root, files('desktop/capture/echo/', 'scripts/setup_echo.py', 'scripts/echo_compile_guard.py', 'scripts/ci_cache.py', 'scripts/component_cache.py'), {})}",
    }


def source_changed(paths):
    # Keep the required Source checks job green on documentation-only PRs.
    return any(not (p.startswith('docs/') or p == 'README.md') for p in paths)


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def tool_identity(profile):
    mac = platform.system() == 'Darwin'
    rust = command('rustc', '-Vv')
    tools = {
        'os': command('sw_vers', '-productVersion') if mac else command('uname', '-sr'),
        'arch': command('uname', '-m'), 'image': os.getenv('ImageVersion', 'local'),
        'target': next(line.removeprefix('host: ') for line in rust.splitlines() if line.startswith('host: ')),
        'rust': rust, 'profile': profile, 'python': command('python3', '--version'),
        'node': command('node', '--version'), 'sccache': command('sccache', '--version'),
        'cc': command('cc', '--version'), 'rustflags': os.getenv('RUSTFLAGS', ''),
        'cmake': command('cmake', '--version'), 'ninja': command('ninja', '--version'),
    }
    if mac:
        # The producer sets 15.0 in build_macos.sh; native CI uses Rust's default.
        tools.update(deployment='15.0' if profile == 'release-default-features-adhoc' else os.getenv('MACOSX_DEPLOYMENT_TARGET', 'default'),
                     xcode=command('xcodebuild', '-version'),
                     swift=command('xcrun', 'swiftc', '--version'), sdk=command('xcrun', '--show-sdk-version'),
                     clang=command('xcrun', 'clang', '--version'))
    return tools


def emit(values):
    print(json.dumps(values, indent=2))
    if os.getenv('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
            for name, value in values.items():
                output.write(f'{name}={value}\n')


if __name__ == '__main__':
    import sys
    if sys.argv[1:] == ['size']:
        paths = [ROOT / '.cache/npm', ROOT / '.cache/components',
                 Path(os.environ['SCCACHE_DIR']), Path.home() / '.cargo/registry',
                 Path.home() / '.cargo/git']
        total = sum(p.stat().st_size for directory in paths if directory.exists()
                    for p in directory.rglob('*') if p.is_file() and not p.is_symlink())
        emit({'cache-bytes': total, 'within-limit': str(total <= 2 * 1024**3).lower()})
        if os.getenv('GITHUB_STEP_SUMMARY'):
            with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as summary:
                summary.write(f'- Compiler and component cache bytes: {total}; save limit: {2 * 1024**3}; pruned Cargo archive bytes in action log\n')
    elif sys.argv[1:] == ['changes']:
        base = os.getenv('BASE_SHA', '')
        if not base or set(base) == {'0'}:
            changed = True
        else:
            names = command('git', 'diff', '--name-only', '-z', base, 'HEAD').split('\0')
            changed = source_changed(list(filter(None, names)))
        emit({'source': str(changed).lower()})
    else:
        profile = 'release-default-features-adhoc'
        if sys.argv[1:] == ['rust']:
            profile = 'debug-fixtures-release-default' if platform.system() == 'Darwin' else 'debug-fixtures'
        tracked = command('git', 'ls-files', '-z').split('\0')
        emit(keys(ROOT, list(filter(None, tracked)), tool_identity(profile)))
