#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export DIAR_PYTHON="$PWD/.venv-package/bin/python"
export ASR_PYTHON="$DIAR_PYTHON"
export SPEAKERDESK_LIVE_PYTHON="$DIAR_PYTHON"
export SPEAKERDESK_MODELS="$PWD/models"
exec "$DIAR_PYTHON" speakerdesk/app.py
