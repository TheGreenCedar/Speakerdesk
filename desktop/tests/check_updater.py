#!/usr/bin/env python3
"""Run the real native admission/export unit tests without building or opening Tauri.

Optional disposable mutation controls demonstrate that the assertions detect
missing shutdown, preparation and export protections. These are source-contract
tests; they do not claim coverage of plugin networking or the macOS installer.
"""
import argparse
import json
import re
from pathlib import Path
import subprocess
import tempfile


def run(source, binary, test=None):
    subprocess.run(['rustc', '--edition=2021', '--test', str(source), '-o', str(binary)], check=True)
    command = [str(binary)] + ([test] if test else [])
    return subprocess.run(command, text=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--negative-controls', action='store_true')
    parser.add_argument('--location', action='store_true',
                        help='Also compile libc and the real macOS location preflight, without Tauri.')
    args = parser.parse_args()
    native = Path(__file__).resolve().parents[1] / 'src-tauri' / 'src'
    sources = (native / 'update_control.rs', native / 'exports.rs')
    results = {'scope': 'real native admission/framing and export filesystem unit tests; no Tauri build or app',
               'positive': [], 'mutation_controls': []}
    with tempfile.TemporaryDirectory(prefix='speakerdesk-updater-tests-') as directory:
        temporary = Path(directory)
        for source in sources:
            outcome = run(source, temporary / source.stem)
            print(outcome.stdout, end='')
            if outcome.returncode:
                print(outcome.stderr, end='')
                raise SystemExit(outcome.returncode)
            results['positive'].append({'source': source.name, 'passed': True})
        if args.negative_controls:
            controls = (
                ('shutdown acknowledgement', sources[0],
                 'self.shutdown_ready && self.terminated_clean', 'self.terminated_clean',
                 'lost_ack_or_timeout_never_installs_on_late_process_exit'),
                ('preparation ownership', sources[0],
                 '&& preparation == Some(self.preparation)', '',
                 'late_editor_or_shutdown_ack_cannot_authorize_a_retried_preparation'),
                ('pending export refusal', sources[1],
                 'if id == 0 || state.stopped || state.pending.is_some()', 'if id == 0 || state.stopped',
                 'update_refuses_download_chooser_and_write_without_cancelling_export'),
            )
            for index, (name, source, before, after, test) in enumerate(controls):
                text = source.read_text()
                if text.count(before) != 1:
                    raise SystemExit(f'Mutation target changed: {name}')
                mutant = temporary / f'control_{index}.rs'
                mutant.write_text(text.replace(before, after, 1))
                outcome = run(mutant, temporary / f'control_{index}', test)
                if outcome.returncode == 0 or '1 failed' not in outcome.stdout:
                    raise SystemExit(f'Protection assertion did not reject removed {name}')
                results['mutation_controls'].append({'removed': name, 'assertion_failed_as_expected': True})
        if args.location:
            lock = (native.parent / 'Cargo.lock').read_text()
            version = re.search(r'name = "libc"\nversion = "([^"]+)"', lock)
            if not version:
                raise SystemExit('The pinned libc resolution is missing')
            manifest = temporary / 'Cargo.toml'
            manifest.write_text('[package]\nname = "speakerdesk-location-check"\nversion = "0.1.0"\nedition = "2021"\n'
                                '[lib]\npath = ' + json.dumps(str(native / 'update_location.rs')) + '\n'
                                '[dependencies]\nlibc = "=' + version.group(1) + '"\n')
            subprocess.run(['cargo', 'test', '--offline', '--manifest-path', str(manifest)], check=True)
            results['positive'].append({'source': 'update_location.rs', 'passed': True})
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
