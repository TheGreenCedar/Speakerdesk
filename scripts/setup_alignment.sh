#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Existing voice runtime supplies protobuf 7.36.2. Do not downgrade it.
uv pip install --python .venv-package/bin/python --no-deps --require-hashes -r requirements-alignment.lock.txt
uv pip install --python .venv-package/bin/python --no-deps --require-hashes --no-build-isolation -r requirements-alignment-ctc.lock.txt
.venv-package/bin/python - <<'PY'
import importlib.metadata as metadata
import importlib.machinery as machinery
import importlib.util
for name, version in {'onnxruntime':'1.30.0','flatbuffers':'25.12.19','Cython':'3.1.4','ctc-segmentation':'1.7.4','numpy':'2.5.3','setuptools':'84.0.0','protobuf':'7.36.2'}.items():
    assert metadata.version(name)==version, f'Alignment dependency differs: {name}'
package=importlib.util.find_spec('ctc_segmentation')
native=machinery.PathFinder.find_spec('ctc_segmentation.ctc_segmentation_dyn',package.submodule_search_locations)
assert native and isinstance(native.loader,machinery.ExtensionFileLoader), 'Precompiled CTC runtime missing'
import onnxruntime, ctc_segmentation
onnxruntime.disable_telemetry_events()
PY
