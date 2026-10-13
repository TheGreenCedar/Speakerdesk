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
            ('ci.yml', "contains(fromJSON('[\"push\", \"workflow_dispatch\"]'), github.event_name)",
             "contains(fromJSON('[\"push\", \"pull_request\"]'), github.event_name)"),
            ('ci.yml', "github.event.inputs.cache != 'false'", 'true'),
            ('ci.yml', '${{ runner.temp }}/speakerdesk-sccache', '${{ runner.temp }}/speakerdesk-native-target'),
            ('ci.yml', 'cache-workspace-crates: true', 'cache-workspace-crates: false'),
            ('apple-build.yml', 'cache-workspace-crates: true', 'cache-workspace-crates: false'),
            ('ci.yml', 'cache-bin: false', 'cache-bin: true'),
            ('ci.yml', 'save-if: ${{', 'save-if: true # ${{'),
            ('ci.yml', '${{ steps.identity.outputs.dependency-key }}', 'unbound-target'),
            ('ci.yml', '.cache/components/echo', '~/.cache/huggingface'),
            ('ci.yml', "SCCACHE_GHA_ENABLED: 'false'", "SCCACHE_GHA_ENABLED: 'true'"),
            ('ci.yml', "CARGO_INCREMENTAL: '0'", "CARGO_INCREMENTAL: '1'"),
            ('ci.yml', "RUSTC_WRAPPER: ${{ github.event.inputs.cache != 'false' && 'sccache' || '' }}",
             'RUSTC_WRAPPER: sccache'),
            ('ci.yml', "steps.budget.outputs.within-limit == 'true'", 'true'),
            ('ci.yml', "if: always() && steps.identity.outcome == 'success' && github.event.inputs.cache != 'false'",
             "if: always() && steps.identity.outcome == 'success'"),
            ('apple-build.yml', '.cache/npm', '~/.cache/huggingface'),
            ('apple-build.yml', "github.event.inputs.cache != 'false'", 'true'),
            ('apple-build.yml', "SPEAKERDESK_CAPTURE_KEY: ${{ github.event.inputs.cache != 'false' && steps.identity.outputs.capture-key || '' }}",
             'SPEAKERDESK_CAPTURE_KEY: ${{ steps.identity.outputs.capture-key }}'),
            ('apple-build.yml', "SPEAKERDESK_ECHO_KEY: ${{ github.event.inputs.cache != 'false' && steps.identity.outputs.echo-key || '' }}",
             'SPEAKERDESK_ECHO_KEY: ${{ steps.identity.outputs.echo-key }}'),
            ('apple-build.yml', "- name: Finish and report compiler cache\n        if: github.event.inputs.cache != 'false'",
             '- name: Finish and report compiler cache'),
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
