"""Hostile controls for privilege, cancellation and persisted-path policy."""
import shutil
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from check_workflows import validate


class WorkflowTests(unittest.TestCase):
    def test_existing_workflows_obey_policy(self):
        validate(ROOT / '.github/workflows')

    def test_macos_debug_commands_reuse_one_graph_and_keep_native_probes(self):
        workflow = yaml.load((ROOT / '.github/workflows/ci.yml').read_text(), Loader=yaml.BaseLoader)
        job = workflow['jobs']['rust-macos']
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            record = directory / 'commands.jsonl'
            cargo = directory / 'cargo'
            cargo.write_text('#!' + sys.executable + '\n'
                             'import json, os, sys\n'
                             'with open(os.environ["CI_TEST_COMMANDS"], "a") as output:\n'
                             '    output.write(json.dumps(sys.argv[1:]) + "\\n")\n')
            cargo.chmod(0o755)
            compiler = directory / 'cc'
            compiler.write_text('#!/bin/sh\nexit 0\n')
            compiler.chmod(0o755)
            environment = dict(os.environ, **job['env'],
                               PATH=str(directory) + os.pathsep + os.environ['PATH'],
                               CI_TEST_COMMANDS=str(record), RUNNER_TEMP=temp,
                               GITHUB_STEP_SUMMARY=str(directory / 'summary'))
            for step in job['steps']:
                script = step.get('run', '')
                if 'cargo test ' in script or 'cargo build ' in script:
                    result = subprocess.run(['bash', '-e', '-c', script], cwd=ROOT,
                                            env=dict(environment, **step.get('env', {})),
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
            commands = [json.loads(line) for line in record.read_text().splitlines()]
        debug = [args for args in commands if args[0] == 'test' and '--release' not in args]
        def selection(args):
            packages = frozenset(args[i + 1] for i, arg in enumerate(args[:-1]) if arg in ('-p', '--package'))
            features = frozenset(args[i + 1] for i, arg in enumerate(args[:-1]) if arg == '--features')
            return packages, features
        graphs = {selection(args) for args in debug}
        self.assertEqual(len(graphs), 1, f'Debug commands selected different dependency graphs: {graphs}')
        packages, features = graphs.pop()
        self.assertEqual(packages, {'speakerdesk-core', 'speakerdesk-host', 'speakerdesk-models',
                                    'speakerdesk-echo', 'xtask', 'speakerdesk'})
        self.assertEqual(features, {'speakerdesk-host/fixtures'})
        self.assertTrue(all('--locked' in args for args in commands))
        self.assertEqual(sum('--ignored' not in args for args in debug), 1,
                         'The complete debug suite must run once, including desktop tests')
        probes = {arg for args in debug if '--ignored' in args for arg in args
                  if arg.startswith(('raw_dual_capture_', 'native_echo_bounds_', 'native_hybrid_',
                                     'windows_capture_routes_'))}
        self.assertEqual(probes, {'raw_dual_capture_opens_and_preserves_partial_pause_tail',
                                 'native_echo_bounds_queues_and_drains_partial_frames_exactly',
                                 'windows_capture_routes_selected_sources_and_retains_failure_endpoints_on_reopen',
                                 'native_hybrid_all_frame_residues',
                                 'native_hybrid_warmed_eof',
                                 'native_hybrid_model_admission'})
        self.assertTrue(any(args[:2] == ['build', '--release'] and 'speakerdesk' in args
                            for args in commands), 'The default-feature release compile remains required')

    def test_unsafe_workflow_mutations_are_rejected(self):
        cases = [
            ('ci.yml', "- name: Prepare stable vendored native source timestamps\n        if: github.event.inputs.cache != 'false'",
             '- name: Prepare stable vendored native source timestamps'),
            ('apple-build.yml', "- name: Prepare stable vendored native source timestamps\n        if: github.event.inputs.cache != 'false'",
             '- name: Prepare stable vendored native source timestamps'),
            ('ci.yml', 'SPEAKERDESK_DEPENDENCY_KEY: ${{ steps.identity.outputs.dependency-key }}',
             'SPEAKERDESK_DEPENDENCY_KEY: unrelated-source'),
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
