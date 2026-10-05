"""Hostile controls for privilege, cancellation and persisted-path policy."""
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from check_workflows import validate


class WorkflowTests(unittest.TestCase):
    def test_existing_workflows_obey_policy(self):
        validate(ROOT / '.github/workflows')

    def test_unsafe_workflow_mutations_are_rejected(self):
        cases = [
            ('ci.yml', "github.ref == 'refs/heads/main'", "github.ref != 'refs/heads/main'"),
            ('ci.yml', 'contents: read', 'contents: write'),
            ('ci.yml', 'pull_request:', 'pull_request_target:'),
            ('ci.yml', "${{ github.event_name == 'pull_request' }}", 'true'),
            ('apple-build.yml', '.cache/uv', '~/.cache/huggingface'),
            ('apple-build.yml', "APPLE_SIGNING_IDENTITY: '-'", 'APPLE_SIGNING_IDENTITY: ${{ secrets.APPLE_SIGNING_IDENTITY }}'),
            ('apple-build.yml', 'cancel-in-progress: false', 'cancel-in-progress: true'),
            ('apple-build.yml', 'actions/cache/restore@caa296126883cff596d87d8935842f9db880ef25', 'actions/cache/restore@v5'),
        ]
        for name, before, after in cases:
            with self.subTest(mutation=after), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                for source in (ROOT / '.github/workflows').glob('*.yml'):
                    shutil.copy2(source, directory / source.name)
                path = directory / name
                original = path.read_text()
                self.assertIn(before, original)
                path.write_text(original.replace(before, after))
                with self.assertRaises(AssertionError):
                    validate(directory)


if __name__ == '__main__':
    unittest.main()
