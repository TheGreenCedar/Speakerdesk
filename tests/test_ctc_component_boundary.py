"""Private CTC dependency boundary; no neural models executed by these tests."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import alignment_model
from mlx_ctc_components import source_files, BASE_SOURCE_SHA256


class ComponentBoundaryTests(unittest.TestCase):
    def test_selected_sources_are_byte_identical_to_calibrated_components(self):
        files, base = source_files()
        expected = alignment_model.manifest()['runtime']['component_source_sha256']
        self.assertEqual(set(files), set(expected))
        for name, path in files.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected[name])
        self.assertEqual(hashlib.sha256(base.read_bytes()).hexdigest(), BASE_SOURCE_SHA256)

    def test_unknown_component_inventory_and_changed_base_are_rejected(self):
        metadata = alignment_model.manifest()
        versions = metadata['runtime']
        with patch('importlib.metadata.version', side_effect=lambda name: versions[name]):
            changed = copy.deepcopy(metadata)
            changed['runtime']['component_source_sha256']['unexpected.py'] = 'wrong'
            with self.assertRaisesRegex(RuntimeError, 'inventory differs'):
                alignment_model.require_runtime(changed)
            real_digest = alignment_model.digest_file
            with patch.object(alignment_model, 'digest_file', side_effect=lambda path:
                              'wrong' if Path(path).name == 'base.py' else real_digest(path)):
                with self.assertRaisesRegex(RuntimeError, 'base component differs'):
                    alignment_model.require_runtime(metadata)

    def test_selected_imports_never_enter_public_STT_and_preserve_warm_namespaces(self):
        # Real private source imports run in fresh children. Only MLX and the
        # unused downloader are peers; no model construction/forward occurs.
        script = r'''
import importlib.abc, json, sys
from types import ModuleType
from unittest.mock import MagicMock
sys.path.insert(0, sys.argv[1])
class RefusePublicSTT(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('mlx_audio.stt') or fullname.startswith('transformers'):
            raise AssertionError('Entered unrelated public STT initializer: '+fullname)
sys.meta_path.insert(0, RefusePublicSTT())
if sys.argv[2] == 'warm':
    for name in ('mlx_audio', 'mlx_audio.stt', 'mlx_audio.stt.models'):
        module = ModuleType(name);module.__path__ = [];module.preserved = object()
        sys.modules[name] = module
before = {name: module for name, module in sys.modules.items() if name.startswith('mlx_audio')}
mx = ModuleType('mlx.core');mx.__getattr__ = lambda name: MagicMock(name='mx.'+name)
nn = ModuleType('mlx.nn');nn.Module = type('Module', (), {})
nn.__getattr__ = lambda name: MagicMock(name='nn.'+name)
mlx = ModuleType('mlx');mlx.__path__ = [];mlx.core = mx;mlx.nn = nn
hub = ModuleType('huggingface_hub')
def never_download(*args, **kwargs):raise AssertionError('Unexpected model download')
hub.snapshot_download = never_download
sys.modules.update({'mlx':mlx, 'mlx.core':mx, 'mlx.nn':nn, 'huggingface_hub':hub})
from mlx_ctc_components.models.mms.mms import Model
from mlx_ctc_components.models.wav2vec.wav2vec import ModelConfig
after = {name: module for name, module in sys.modules.items() if name.startswith('mlx_audio')}
assert before.keys() == after.keys()
assert all(after[name] is module for name, module in before.items())
assert Model.__module__ == 'mlx_ctc_components.models.mms.mms'
assert ModelConfig.__module__ == 'mlx_ctc_components.models.wav2vec.wav2vec'
assert not any(name.startswith('transformers') for name in sys.modules)
print(json.dumps({'private_imports':True,'public_namespaces_unchanged':True,'neural_calls':0}))
'''
        root = Path(__file__).resolve().parents[1] / 'speakerdesk'
        for state in ('cold', 'warm'):
            with self.subTest(state=state):
                run = subprocess.run([sys.executable, '-B', '-c', script, str(root), state],
                                     capture_output=True, text=True, timeout=5)
                self.assertEqual(run.returncode, 0, run.stderr)
                self.assertTrue(json.loads(run.stdout)['public_namespaces_unchanged'])

    def test_frozen_inventory_contains_private_sources_and_license(self):
        root = Path(__file__).resolve().parents[1]
        spec = (root / 'packaging/runtime.spec').read_text()
        self.assertIn("'mlx_ctc_components':'pyz+py'", spec)
        self.assertIn('licenses/mlx-ctc-components', spec)
        self.assertNotIn("collect_submodules('mlx_audio.stt.models.mms')", spec)
        self.assertNotIn("collect_submodules('mlx_audio.stt.models.wav2vec')", spec)
        self.assertIn('Copyright (c) 2024 Prince Canuma',
                      (root / 'packaging/licenses/mlx-ctc-components/LICENSE').read_text())


if __name__ == '__main__':
    unittest.main()
