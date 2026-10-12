#!/bin/bash
set -euo pipefail
task_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$task_root"
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo 'Apple Silicon macOS is required.' >&2; exit 1; }
: "${APPLE_SIGNING_IDENTITY:?Provide an existing Developer ID identity or '-' for a CI signing input.}"
export APPLE_SIGNING_IDENTITY
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-2}"
export RUSTC_WRAPPER="${RUSTC_WRAPPER:-}"
export MACOSX_DEPLOYMENT_TARGET=15.0
build_python="${SPEAKERDESK_BUILD_PYTHON:-python3}"
native_root="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
export CARGO_TARGET_DIR="${SPEAKERDESK_NATIVE_TARGET_DIR:-$native_root/speakerdesk-native-target}"
mkdir -p "$CARGO_TARGET_DIR" .cache release
# Python is only a build tool for the pinned native DSP and helper manifest.
export MLX_PREBUILT_ROOT
MLX_PREBUILT_ROOT="$(bash scripts/fetch_mlx_runtime.sh)"
"$build_python" scripts/component_cache.py .cache/components/echo "${SPEAKERDESK_ECHO_KEY:-}" desktop/capture/echo/libspeakerdesk_echo.dylib -- "$build_python" scripts/setup_echo.py
"$build_python" scripts/native_licenses.py
package_version="$(node -p 'require("./desktop/src-tauri/tauri.conf.json").version')"
"$build_python" scripts/component_cache.py .cache/components/capture "${SPEAKERDESK_CAPTURE_KEY:-}" desktop/capture/speakerdesk-capture -- xcrun swiftc -O -j 1 -num-threads 1 -target arm64-apple-macos15.0 -module-cache-path "$task_root/.cache/swift-capture" desktop/capture/MeetingCapture.swift desktop/capture/CaptureAudioConverter.swift desktop/capture/LosslessRedundantChannels.swift desktop/capture/AppleCaptureRoute.swift -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker desktop/capture/Info.plist -o desktop/capture/speakerdesk-capture
"$build_python" scripts/capture_capability.py desktop/capture/speakerdesk-capture
sign_args=(--force --options runtime --entitlements desktop/src-tauri/Entitlements.plist --sign "$APPLE_SIGNING_IDENTITY")
library_sign_args=(--force --options runtime --sign "$APPLE_SIGNING_IDENTITY")
if [[ "$APPLE_SIGNING_IDENTITY" != '-' ]]; then sign_args+=(--timestamp); library_sign_args+=(--timestamp); fi
cd desktop
npm ci
npm run build -- --target aarch64-apple-darwin --bundles app -- --locked
cd "$task_root"
package_stage="$(mktemp -d "$native_root/speakerdesk-package.XXXXXX")"
trap 'rm -rf "$package_stage"' EXIT
ditto --noextattr --norsrc "$CARGO_TARGET_DIR/aarch64-apple-darwin/release/bundle/macos/Speakerdesk.app" "$package_stage/Speakerdesk.app"
ln -s /Applications "$package_stage/Applications"
# Bind the receipt to final signed helper bytes before sealing the outer app.
# Sign only the staged copies; the fetched MLX artifact remains hash-identical.
resources="$package_stage/Speakerdesk.app/Contents/Resources"
codesign "${library_sign_args[@]}" "$package_stage/Speakerdesk.app/Contents/Frameworks/libmlx.dylib"
codesign "${library_sign_args[@]}" "$resources/libspeakerdesk_echo.dylib"
codesign "${sign_args[@]}" "$resources/speakerdesk-capture"
"$build_python" scripts/capture_capability.py "$resources/speakerdesk-capture"
codesign "${sign_args[@]}" "$package_stage/Speakerdesk.app"
codesign --verify --deep --strict "$package_stage/Speakerdesk.app"
cargo xtask verify-bundle "$package_stage/Speakerdesk.app"
hdiutil create -volname Speakerdesk -srcfolder "$package_stage" -format UDZO -fs HFS+ -ov "release/Speakerdesk_${package_version}_AppleSilicon.dmg"
dmg_sign_args=(--force --sign "$APPLE_SIGNING_IDENTITY")
if [[ "$APPLE_SIGNING_IDENTITY" != '-' ]]; then dmg_sign_args+=(--timestamp); fi
codesign "${dmg_sign_args[@]}" "release/Speakerdesk_${package_version}_AppleSilicon.dmg"
ditto -c -k --noextattr --norsrc --keepParent "$package_stage/Speakerdesk.app" "release/Speakerdesk_${package_version}_AppleSilicon.app.zip"
if [[ "$APPLE_SIGNING_IDENTITY" == '-' ]]; then echo 'Ad-hoc CI signing input built; same-team Developer ID signing is required before launch, then notarization.'; else echo 'Developer ID package built; notarization remains a separate step.'; fi
