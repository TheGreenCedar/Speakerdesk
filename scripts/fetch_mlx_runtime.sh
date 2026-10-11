#!/bin/bash
# Apple's backend-only wheel contains native MLX libraries, headers and Metal kernels.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
destination="$root/target/native/mlx-0.32.2"
sha=55a369250d220b2cf10213a87a2ac1b1a420608c5b35b1df4e7147ac8e32f121
url=https://files.pythonhosted.org/packages/79/ec/34f37376e26d537fadffb99af3a760d6545e37f5e1a30a552baadf237fc5/mlx_metal-0.32.2-py3-none-macosx_15_0_arm64.whl
if [[ -f "$destination/.archive-sha256" ]] && [[ $(cat "$destination/.archive-sha256") == "$sha" ]] && [[ -f "$destination/lib/libmlx.dylib" && -f "$destination/lib/mlx.metallib" ]]; then
    printf '%s\n' "$destination"
    exit 0
fi
if [[ -e "$destination" ]]; then
    printf 'Incomplete or different runtime at %s; remove that build dependency before retrying.\n' "$destination" >&2
    exit 1
fi
mkdir -p "$(dirname "$destination")"
stage=$(mktemp -d "$(dirname "$destination")/.mlx-download.XXXXXX")
trap 'rm -rf "$stage"' EXIT
curl --fail --location --retry 2 --silent --show-error "$url" -o "$stage/runtime.whl"
printf '%s  %s\n' "$sha" "$stage/runtime.whl" | shasum -a 256 --check --status
unzip -q "$stage/runtime.whl" 'mlx/*' 'mlx_metal-*.dist-info/licenses/*' -d "$stage/unpacked"
cp "$stage/unpacked/mlx_metal-0.32.2.dist-info/licenses/LICENSE" "$stage/unpacked/mlx/LICENSE"
printf '%s\n' "$sha" > "$stage/unpacked/mlx/.archive-sha256"
mv "$stage/unpacked/mlx" "$destination"
printf '%s\n' "$destination"
