"""Describe expected unsigned NSIS executable bytes without changing build outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


TAURI_CLI = "2.12.1"
UNPATCHED = b"__TAURI_BUNDLE_TYPE_VAR_UNK"
NSIS = b"__TAURI_BUNDLE_TYPE_VAR_NSS"
# patch_binary changes the first marker, then bundle_project restores its input.
SOURCE = ("https://raw.githubusercontent.com/tauri-apps/tauri/tauri-cli-v2.12.1/"
          "crates/tauri-bundler/src/bundle.rs")


def expected_nsis_executable(executable, package_lock, installed_cli):
    lock = json.loads(Path(package_lock).read_text(encoding="utf-8"))
    installed = json.loads(Path(installed_cli).read_text(encoding="utf-8"))
    if (lock["packages"]["node_modules/@tauri-apps/cli"]["version"] != TAURI_CLI
            or installed["version"] != TAURI_CLI):
        raise ValueError("NSIS payload derivation requires locked and installed Tauri CLI 2.12.1")

    executable = Path(executable)
    if executable.is_symlink() or not executable.is_file():
        raise ValueError("The executable input must be a regular file")
    data = executable.read_bytes()
    # Signing after the patch also changes PE checksum/certificate bytes; that
    # future release lane must observe its signed payload instead of deriving it.
    if len(data) < 64 or data[:2] != b"MZ":
        raise ValueError("Expected an unsigned x64 PE executable")
    pe = struct.unpack_from("<I", data, 60)[0]
    optional = pe + 24
    if (data[pe:pe + 4] != b"PE\0\0" or len(data) < optional + 152
            or struct.unpack_from("<H", data, pe + 4)[0] != 0x8664
            or struct.unpack_from("<H", data, pe + 20)[0] < 152
            or struct.unpack_from("<H", data, optional)[0] != 0x20B):
        raise ValueError("Expected an unsigned x64 PE executable")
    if (struct.unpack_from("<I", data, optional + 108)[0] > 4
            and data[optional + 144:optional + 152] != b"\0" * 8):
        raise ValueError("Signed PE payloads cannot be derived from the unsigned marker transformation")
    if data.count(UNPATCHED) != 1 or NSIS in data:
        raise ValueError("Expected exactly one UNK bundle marker and no NSS marker")

    offset = data.index(UNPATCHED)
    payload = hashlib.sha256()
    payload.update(data[:offset])
    payload.update(NSIS)
    payload.update(data[offset + len(UNPATCHED):])
    return {
        "kind": "tauri-cli-2.12.1-unsigned-nsis-marker-v1",
        "observed": False,
        "source": SOURCE,
        "marker_offset": offset,
        "input_marker": UNPATCHED.decode("ascii"),
        "payload_marker": NSIS.decode("ascii"),
        "bytes": len(data),
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "expected_payload_sha256": payload.hexdigest(),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--package-lock", type=Path, default=root / "desktop/package-lock.json")
    parser.add_argument("--installed-cli", type=Path,
                        default=root / "desktop/node_modules/@tauri-apps/cli/package.json")
    args = parser.parse_args()
    print(json.dumps(expected_nsis_executable(args.executable, args.package_lock, args.installed_cli)))


if __name__ == "__main__":
    main()
