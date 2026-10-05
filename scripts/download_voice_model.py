"""Explicit post-approval installer for one pinned artifact; never loads the model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request

PIN_PATH = Path(__file__).resolve().parents[1]/'speakerdesk/redimnet2_artifact.json'
DISK_FLOOR = 40_000_000_000


def download(destination):
    pin = json.loads(PIN_PATH.read_text())
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError('Choose a new model directory; existing data will not be replaced.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < DISK_FLOOR+pin['total_bytes']:
        raise ValueError('Keep at least 40 GB free after the bounded model download.')
    with tempfile.TemporaryDirectory(prefix='.voice-download-', dir=destination.parent) as staging:
        root = Path(staging)/'artifact'
        for relative, expected in pin['files'].items():
            path = root/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/{pin['model_id']}/resolve/{pin['revision']}/{relative}"
            digest, size = hashlib.sha256(), 0
            with urllib.request.urlopen(url, timeout=20) as response, path.open('xb') as output:
                while chunk := response.read(min(512*1024, expected['size']-size+1)):
                    size += len(chunk)
                    if size > expected['size']:
                        raise ValueError(f'Voice download exceeded its pinned size: {relative}')
                    digest.update(chunk); output.write(chunk)
            if size != expected['size'] or digest.hexdigest() != expected['sha256']:
                raise ValueError(f'Voice download failed the pinned checksum: {relative}')
        if shutil.disk_usage(destination.parent).free < DISK_FLOOR:
            raise ValueError('Download would breach the 40 GB free-space floor.')
        os.rename(root, destination)
    return pin['total_bytes']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--approved-community-artifact', action='store_true', required=True,
                        help='Use only after explicit approval of the pinned source/license/resource plan.')
    args = parser.parse_args()
    try:
        print(f'Validated {download(args.destination)} downloaded bytes. No model was loaded.')
    except (ValueError, OSError) as error:
        parser.exit(1, f'{error}\n')
