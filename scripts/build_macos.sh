#!/bin/bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$task_root"
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo 'Apple Silicon macOS is required.' >&2; exit 1; }
: "${APPLE_SIGNING_IDENTITY:?Provide an existing Developer ID identity or '-' for a CI test package.}"
export SPEAKERDESK_SIGNING_IDENTITY="$APPLE_SIGNING_IDENTITY"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$task_root/.cache/uv}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
export RUSTC_WRAPPER="${RUSTC_WRAPPER:-}"
export MACOSX_DEPLOYMENT_TARGET=15.0
export PYINSTALLER_CONFIG_DIR="$task_root/.cache/pyinstaller"
native_root="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
export CARGO_TARGET_DIR="${SPEAKERDESK_NATIVE_TARGET_DIR:-$native_root/speakerdesk-native-target}"
mkdir -p "$CARGO_TARGET_DIR" .cache release
if [[ ! -x .venv-package/bin/python ]]; then ./scripts/setup.sh; else
  uv pip install --python .venv-package/bin/python -r requirements-packaging.lock.txt
  uv pip install --python .venv-package/bin/python --no-deps --require-hashes -r requirements-voice.lock.txt
  bash scripts/setup_alignment.sh
  .venv-package/bin/python - <<'PY'
import shutil,site
from pathlib import Path
for family in ('vad','stt'):
    relative=f'mlx_audio/{family}/models/__init__.py'
    shutil.copyfile(Path('packaging/overrides')/relative,
                   Path(site.getsitepackages()[0])/relative)
PY
fi
uv pip check --python .venv-package/bin/python
.venv-package/bin/python scripts/check_source.py
.venv-package/bin/python -m unittest discover -s tests -v
package_version="$(.venv-package/bin/python -c 'import json;print(json.load(open("desktop/src-tauri/tauri.conf.json"))["version"])')"
.venv-package/bin/python scripts/component_cache.py .cache/components/runtime "${SPEAKERDESK_RUNTIME_KEY:-}" desktop/src-tauri/binaries/speakerdesk-runtime-aarch64-apple-darwin -- .venv-package/bin/python -m PyInstaller --noconfirm --clean --workpath .cache/pyinstaller-build --distpath desktop/src-tauri/binaries packaging/runtime.spec
if [[ "$APPLE_SIGNING_IDENTITY" != '-' ]]; then
  .venv-package/bin/python scripts/runtime_capability.py --runtime desktop/src-tauri/binaries/speakerdesk-runtime-aarch64-apple-darwin --output .cache/model-capability.json
else
  # Hardened ad-hoc Python has no matching Team ID for its nested libraries.
  # Never relax production signature protections to manufacture a green probe.
  echo '{"scope":"capability_unavailable","reason":"Developer ID signed runtime required","models_executed":false,"native_capture":false}' | tee .cache/model-capability.json
fi
.venv-package/bin/python scripts/component_cache.py .cache/components/capture "${SPEAKERDESK_CAPTURE_KEY:-}" desktop/capture/speakerdesk-capture -- xcrun swiftc -O -j 1 -num-threads 1 -target arm64-apple-macos15.0 -module-cache-path "$task_root/.cache/swift-capture" desktop/capture/MeetingCapture.swift -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker desktop/capture/Info.plist -o desktop/capture/speakerdesk-capture
sign_args=(--force --options runtime --entitlements desktop/src-tauri/Entitlements.plist --sign "$APPLE_SIGNING_IDENTITY")
if [[ "$APPLE_SIGNING_IDENTITY" != '-' ]]; then sign_args+=(--timestamp); fi
codesign "${sign_args[@]}" desktop/capture/speakerdesk-capture
cd desktop
npm ci
npm run build -- --target aarch64-apple-darwin --bundles app -- --locked
cd "$task_root"
package_stage="$(mktemp -d "$native_root/speakerdesk-package.XXXXXX")"
trap 'rm -rf "$package_stage"' EXIT
ditto --noextattr --norsrc "$CARGO_TARGET_DIR/aarch64-apple-darwin/release/bundle/macos/Speakerdesk.app" "$package_stage/Speakerdesk.app"
ln -s /Applications "$package_stage/Applications"
codesign "${sign_args[@]}" "$package_stage/Speakerdesk.app"
codesign --verify --deep --strict "$package_stage/Speakerdesk.app"
hdiutil create -volname Speakerdesk -srcfolder "$package_stage" -format UDZO -fs HFS+ -ov "release/Speakerdesk_${package_version}_AppleSilicon.dmg"
dmg_sign_args=(--force --sign "$APPLE_SIGNING_IDENTITY")
if [[ "$APPLE_SIGNING_IDENTITY" != '-' ]]; then dmg_sign_args+=(--timestamp); fi
codesign "${dmg_sign_args[@]}" "release/Speakerdesk_${package_version}_AppleSilicon.dmg"
ditto -c -k --noextattr --norsrc --keepParent "$package_stage/Speakerdesk.app" "release/Speakerdesk_${package_version}_AppleSilicon.app.zip"
if [[ "$APPLE_SIGNING_IDENTITY" == '-' ]]; then echo 'Ad-hoc CI test package built; it is not notarized.'; else echo 'Developer ID package built; notarization remains a separate step.'; fi
