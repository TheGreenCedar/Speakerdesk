"""Fetch the pinned, self-contained x64 Windows ML build dependency and its notices."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open("rb") as source:
        result = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
        return result.hexdigest()


def verify(directory, files):
    for name, pin in files.items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or digest(path) != pin["sha256"]:
            raise ValueError(f"Windows runtime file differs from its pin: {path}")


def fetch(destination, pin, archive=None):
    if destination.is_symlink():
        raise ValueError(f"Windows runtime destination is a link: {destination}")
    if destination.exists():
        verify(destination, pin["files"])
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".windows-ml-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        if archive is None:
            archive = stage / "runtime.nupkg"
            with urllib.request.urlopen(pin["url"], timeout=60) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        if digest(archive) != pin["archive_sha256"]:
            raise ValueError("Windows runtime archive differs from its pin")
        unpacked = stage / "unpacked"
        unpacked.mkdir()
        with zipfile.ZipFile(archive) as package:
            # Extract only named flat resources; archive paths never choose output destinations.
            for name, artifact in pin["files"].items():
                if Path(name).name != name:
                    raise ValueError("Runtime destination must be a filename")
                with package.open(artifact["source"]) as source, (unpacked / name).open("wb") as output:
                    shutil.copyfileobj(source, output)
        verify(unpacked, pin["files"])
        unpacked.rename(destination)
    return destination


if __name__ == "__main__":
    pin = json.loads((ROOT / "crates/models/src/windows-runtime.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Reuse an already downloaded archive; its pin is still verified")
    parser.add_argument("--destination", type=Path, default=ROOT / ".cache/native" / f"windows-ml-{pin['version']}-x64")
    args = parser.parse_args()
    print(fetch(args.destination, pin, args.archive))
