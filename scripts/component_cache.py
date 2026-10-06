"""Reuse exact-input ad-hoc components after verifying their stored byte digest."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(8 * 1024**2):
            value.update(block)
    return value.hexdigest()


def restore(directory, key, destination):
    payload, receipt = directory / 'payload', directory / 'receipt.json'
    if not payload.is_file() or not receipt.is_file():
        return False
    if directory.is_symlink() or payload.is_symlink() or receipt.is_symlink():
        raise ValueError('Unsafe component cache path')
    metadata = json.loads(receipt.read_text())
    if metadata != {'key': key, 'sha256': digest(payload), 'bytes': payload.stat().st_size}:
        raise ValueError('Component cache identity or digest mismatch')
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(payload, destination)
    return True


def run(directory, key, destination, command):
    started = time.monotonic()
    # Developer ID builds always sign fresh; they never consume or populate this cache.
    eligible = os.getenv('APPLE_SIGNING_IDENTITY') == '-' and bool(key)
    hit = eligible and restore(directory, key, destination)
    if not hit:
        subprocess.run(command, check=True)
        if eligible:
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, directory / 'payload')
            (directory / 'receipt.json').write_text(json.dumps({
                'key': key, 'sha256': digest(destination), 'bytes': destination.stat().st_size,
            }) + '\n')
    seconds = round(time.monotonic() - started, 3)
    report = f'Component {directory.name}: {"exact verified hit" if hit else "built"}; {seconds}s'
    print(report, flush=True)
    if os.getenv('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as summary:
            summary.write(f'- {report}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('key')
    parser.add_argument('destination', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    run(args.directory, args.key, args.destination, command)
