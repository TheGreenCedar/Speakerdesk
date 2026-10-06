#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo 'Apple Silicon macOS is required.' >&2; exit 1; }
export UV_CACHE_DIR="$PWD/.cache/uv"
uv venv --python "${SPEAKERDESK_PYTHON:-3.13}" .venv-package
uv pip install --python .venv-package/bin/python -r requirements-packaging.lock.txt
uv pip install --python .venv-package/bin/python --no-deps --require-hashes -r requirements-voice.lock.txt
.venv-package/bin/python - <<'PY'
import shutil,site
from pathlib import Path
for family in ('vad','stt'):
    relative=f'mlx_audio/{family}/models/__init__.py'
    shutil.copyfile(Path('packaging/overrides')/relative,
                   Path(site.getsitepackages()[0])/relative)
PY
echo 'Dependencies installed. Start the app, then use its setup control to download models.'
