#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo 'Apple Silicon macOS is required.' >&2; exit 1; }
export UV_CACHE_DIR="$PWD/.cache/uv"
uv venv --python 3.13 .venv-package
uv pip install --python .venv-package/bin/python -r requirements-packaging.lock.txt
.venv-package/bin/python - <<'PY'
import shutil,site
from pathlib import Path
shutil.copyfile('packaging/overrides/mlx_audio/vad/models/__init__.py',
               Path(site.getsitepackages()[0])/'mlx_audio/vad/models/__init__.py')
PY
echo 'Dependencies installed. Start the app, then use its setup control to download models.'
