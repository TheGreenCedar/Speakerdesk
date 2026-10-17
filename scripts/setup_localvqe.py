"""Pinned build inputs for the bundled capture enhancer; never run upstream hooks."""
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'desktop/capture/echo/localvqe-lock.json'
CACHE = ROOT / '.cache/localvqe'
MODEL = 'localvqe-v1.2-1.3M-f32.gguf'


def artifact(name):
    pin = json.loads(LOCK.read_text())['artifacts'][name]
    path = CACHE / name
    if path.is_symlink():
        raise ValueError(f'Unsafe LocalVQE cache file: {path}')
    if path.exists():
        data = path.read_bytes()
    else:
        with urllib.request.urlopen(pin['url'], timeout=30) as response:
            data = response.read(pin['bytes'] + 1)
    if len(data) != pin['bytes'] or hashlib.sha256(data).hexdigest() != pin['sha256']:
        raise ValueError(f'LocalVQE artifact differs from its pin: {name}')
    CACHE.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open('xb') as output:
            output.write(data)
    return path


def model():
    source = artifact(MODEL)
    destination = ROOT / 'desktop/capture/echo' / MODEL
    if destination.is_symlink():
        raise ValueError('Bundled LocalVQE model must not be a symlink')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return destination


def sources():
    # Recheck extracted files against authenticated archives, including cache hits.
    for name, directory in [('LocalVQE.tar.gz', 'source'), ('ggml.tar.gz', 'ggml')]:
        source = CACHE / directory
        if source.is_symlink():
            raise ValueError('LocalVQE source must not be a symlink')
        total = 0
        expected = set()
        with tarfile.open(artifact(name), mode='r:gz') as archive:
            for entry in archive:
                parts = Path(entry.name).parts
                if not parts or Path(entry.name).is_absolute() or '..' in parts or not (entry.isdir() or entry.isfile()):
                    raise ValueError('Unsafe LocalVQE archive entry')
                relative = Path(*parts[1:])
                path = source / relative
                if path.is_symlink() or any(p.is_symlink() for p in path.parents if p == source or source in p.parents):
                    raise ValueError('Unsafe LocalVQE source path')
                if entry.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                    continue
                total += entry.size
                if total > 128 * 1024**2 or len(expected) > 10000:
                    raise ValueError('LocalVQE extraction budget exceeded')
                expected.add(relative)
                data = archive.extractfile(entry).read()
                if path.exists():
                    if path.read_bytes() != data:
                        raise ValueError(f'Modified pinned LocalVQE source: {path}')
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with path.open('xb') as output:
                        output.write(data)
        if {p.relative_to(source) for p in source.rglob('*') if p.is_file()} != expected:
            raise ValueError(f'Unexpected LocalVQE source files: {source}')
    return CACHE / 'source'


def tree_identity(directory):
    if directory.is_symlink() or getattr(directory, 'is_junction', lambda: False)():
        raise ValueError('Unsafe prepared LocalVQE source')
    result = {}
    for path in directory.rglob('*'):
        if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
            raise ValueError('Unsafe prepared LocalVQE source')
        result[path.relative_to(directory).as_posix()] = ('directory' if path.is_dir()
            else hashlib.sha256(path.read_bytes()).hexdigest())
    return result


def prepare():
    source = sources()
    patch = ROOT / 'desktop/capture/echo/localvqe-portability.patch'
    identity = hashlib.sha256(LOCK.read_bytes() + patch.read_bytes()).hexdigest()
    prepared = CACHE / ('prepared-' + identity)
    incoming = prepared.with_name(prepared.name + '.incoming')
    if incoming.exists() or incoming.is_symlink():
        raise ValueError(f'Incomplete LocalVQE preparation: {incoming}')
    # Reconstruct expected bytes from authenticated inputs on every cache hit.
    # Keep an identical prepared tree in place so native incremental mtimes survive.
    try:
        shutil.copytree(source, incoming)
        shutil.copytree(CACHE / 'ggml', incoming / 'ggml/vendor/ggml', dirs_exist_ok=True)
        subprocess.run(['git', 'apply', '--unsafe-paths', '--directory', (incoming / 'ggml').as_posix(), str(patch)],
                       cwd=ROOT, check=True)
        if prepared.exists() or prepared.is_symlink():
            if tree_identity(prepared) != tree_identity(incoming):
                raise ValueError('Modified, missing or unexpected prepared LocalVQE source')
        else:
            incoming.rename(prepared)
    finally:
        if incoming.is_dir():
            shutil.rmtree(incoming)
    model()
    return prepared / 'ggml'
