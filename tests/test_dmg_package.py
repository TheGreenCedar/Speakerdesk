"""The real packaging path retries only transient DMG creation failures."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUSY = 'hdiutil: create failed - Resource busy'
COMMAND = r'''
import json, os, shutil, stat, sys
from pathlib import Path
name, args = Path(sys.argv[0]).name, sys.argv[1:]
record = Path(os.environ['DMG_TEST_RECORD'])
events = [json.loads(line) for line in record.read_text().splitlines()] if record.exists() else []
event = {'command': name, 'args': args}
code = 0
if name == 'hdiutil':
    source = Path(args[args.index('-srcfolder') + 1])
    event['source'] = str(source)
    event['files'] = sorted(str(path.relative_to(source)) for path in source.rglob('*'))
    output_stat = os.fstat(1)
    event['regular_log'] = stat.S_ISREG(output_stat.st_mode)
    event['log'] = next((str(path) for path in Path(os.environ['native_root']).glob('speakerdesk-dmg.*')
                         if path.stat().st_ino == output_stat.st_ino), None)
    attempt = sum(item['command'] == name for item in events)
    code, message = json.loads(os.environ['DMG_TEST_OUTCOMES'])[attempt]
    Path(args[-1]).write_text('complete fixture image' if code == 0 else 'incomplete fixture image')
    print(message, file=sys.stderr if code else sys.stdout)
elif name == 'ditto':
    if '-c' in args:
        Path(args[-1]).write_text('fixture app archive')
    else:
        shutil.copytree(args[-2], args[-1])
with record.open('a') as output:
    output.write(json.dumps(event) + '\n')
raise SystemExit(code)
'''


class DmgPackageTests(unittest.TestCase):
    def run_package(self, outcomes, exit_code, sleeps):
        with tempfile.TemporaryDirectory(prefix='speakerdesk package ') as temporary:
            root = Path(temporary).resolve()
            native_root = root / 'native temp'
            native_root.mkdir()
            app = root / 'target/aarch64-apple-darwin/release/bundle/macos/Speakerdesk.app'
            for relative in ['Contents/Frameworks/libmlx.dylib',
                             'Contents/Resources/libspeakerdesk_echo.dylib',
                             'Contents/Resources/speakerdesk-capture']:
                path = app / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('Independent app fixture')
            (root / 'release').mkdir()
            (root / 'scripts').mkdir()
            (root / 'scripts/capture_capability.py').write_text('')
            tools = root / 'tools'
            tools.mkdir()
            for name in ['hdiutil', 'ditto', 'codesign', 'cargo', 'sleep']:
                command = tools / name
                command.write_text('#!' + sys.executable + '\n' + COMMAND)
                command.chmod(0o755)
            record = root / 'commands.jsonl'
            environment = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ['PATH'],
                               native_root=str(native_root), CARGO_TARGET_DIR=str(root / 'target'),
                               APPLE_SIGNING_IDENTITY='-', build_python=sys.executable,
                               package_version='0.7.1', DMG_TEST_RECORD=str(record),
                               DMG_TEST_OUTCOMES=json.dumps(outcomes))
            script = (ROOT / 'scripts/build_macos.sh').read_text()
            # Execute the actual stage/copy/sign/create/archive path, without rebuilding its inputs.
            packaging = script[script.index('package_stage="$(mktemp -d'):]
            result = subprocess.run(['bash', '-c', 'set -euo pipefail\nsign_args=()\nlibrary_sign_args=()\n' + packaging],
                                    cwd=root, env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, exit_code, result.stdout + result.stderr)
            events = [json.loads(line) for line in record.read_text().splitlines()]
            creates = [event for event in events if event['command'] == 'hdiutil']
            self.assertEqual(len(creates), len(outcomes))
            self.assertEqual([event['args'] for event in events if event['command'] == 'sleep'],
                             [[str(delay)] for delay in sleeps])
            self.assertEqual(len({event['source'] for event in creates}), 1)
            self.assertEqual(sum(event['command'] == 'cargo' for event in events), 1)
            for event in creates:
                self.assertTrue(event['regular_log'])
                self.assertEqual(Path(event['log']).parent, native_root)
                self.assertIn('Speakerdesk.app/Contents/Resources/speakerdesk-capture', event['files'])
                self.assertEqual(event['args'][:3], ['create', '-volname', 'Speakerdesk'])
                self.assertEqual(event['args'][5:9], ['-format', 'UDZO', '-fs', 'HFS+'])
                self.assertEqual(event['args'][-2], '-ov')
            self.assertEqual(list(native_root.iterdir()), [])
            dmg = 'release/Speakerdesk_0.7.1_AppleSilicon.dmg'
            signed_dmg = any(event['command'] == 'codesign' and event['args'][-1] == dmg for event in events)
            zipped = (root / 'release/Speakerdesk_0.7.1_AppleSilicon.app.zip').exists()
            self.assertEqual(signed_dmg, exit_code == 0)
            self.assertEqual(zipped, exit_code == 0)
            for code, message in outcomes:
                if code:
                    self.assertIn(message, result.stderr)

    def test_success_packages_without_retry(self):
        self.run_package([(0, 'created fixture image')], 0, [])

    def test_busy_then_success_reuses_verified_stage(self):
        self.run_package([(1, BUSY), (0, 'created fixture image')], 0, [2])

    def test_last_attempt_success_packages(self):
        self.run_package([(1, BUSY), (1, BUSY), (0, 'created fixture image')], 0, [2, 4])

    def test_busy_exhaustion_fails_without_signing_or_archiving(self):
        self.run_package([(1, BUSY)] * 3, 1, [2, 4])

    def test_other_errors_keep_exit_status_and_do_not_retry(self):
        for message in ['hdiutil: create failed - Permission denied', BUSY + ' while reading source']:
            with self.subTest(message=message):
                self.run_package([(23, message)], 23, [])
