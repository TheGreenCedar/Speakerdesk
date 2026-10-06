"""CPU frontend contracts using real scripts and a small DOM model, no browser."""
import shutil
import subprocess
import unittest
from pathlib import Path


class FrontendContracts(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node is needed for frontend CPU contracts')
    def test_frontend_contracts(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [shutil.which('node'), '--test', 'tests/frontend_performance.test.cjs', 'tests/negative_audio_presentation.test.cjs', 'tests/optional_model_label.test.cjs'],
            cwd=root, capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
