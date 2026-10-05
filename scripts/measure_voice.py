"""Bounded real-model smoke/calibration run after artifact/runtime approval."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from voice_coreml import COMPUTE_UNITS, ReDimNet2CoreML, file_digest
from voice_measurement import calibrate, extract_groups, fixture_groups


def measure(args):
    manifest = json.loads(args.fixtures.read_text())
    groups = fixture_groups(manifest, args.fixtures, maximum_clips=8 if args.smoke else 200)
    if args.smoke and any(g['split'] != 'smoke' for g in groups):
        raise ValueError('Use a dedicated smoke fixture manifest; it never produces calibration.')
    if not args.smoke and any(g['split'] == 'smoke' for g in groups):
        raise ValueError('Smoke queries cannot be reused as held-out calibration evidence.')
    backend = ReDimNet2CoreML(args.model_dir, args.compute_units)
    backend.validate()
    report = {'schema_version': 1, 'approved_for_recognition': False, 'fixture_source': manifest['fixture_source'],
              'model': backend.model.payload(), 'compute_units': args.compute_units,
              'model_dir': os.path.relpath(args.model_dir.resolve(), args.output.resolve().parent),
              'dataset_id': manifest['dataset_id'], 'mode': 'smoke' if args.smoke else 'calibration'}
    report['fixture_manifest_sha256'] = file_digest(args.fixtures)
    report['fixture_provenance'] = [{'group': g['id'], 'recording_id': g['recording_id'], 'split': g['split'],
                                    'speaker_id': g['speaker_id'], 'audio_sha256': g['audio_sha256'],
                                    'clips': [c.payload() for c in g['selected']]} for g in groups]
    if args.smoke:
        first = groups[0]
        warmups = [backend.embed(first['audio'], first['selected'][0]) for _ in range(2)]
        if any(e.clean is not True for e in warmups):
            raise ValueError('Smoke warmup clip failed waveform quality checks.')
        report['warmup_metrics'] = [e.metrics for e in warmups]
    results, report['clip_metrics'] = extract_groups(backend, groups)
    if args.smoke:
        first = results[0]
        repeated = [backend.embed(first['audio'], first['selected'][0]) for _ in range(2)]
        report['repeat_cosines'] = [sum(a*b for a, b in zip(first['vectors'][0], e.vector)) for e in repeated]
        report['repeat_metrics'] = [e.metrics for e in repeated]
        pairs = []
        flattened = [(g['speaker_id'], clip.segment_id, vector) for g in results
                     for clip, vector in zip(g['selected'], g['vectors'])]
        for i, (speaker, sid, vector) in enumerate(flattened):
            for other, oid, candidate in flattened[i+1:]:
                pairs.append({'clips': [sid, oid], 'same_label': speaker == other and speaker is not None,
                              'cosine': sum(a*b for a, b in zip(vector, candidate))})
        report['diagnostic_pairs'] = pairs
        report['limitation'] = 'Extraction smoke only. No calibrated policy or cross-meeting accuracy claim.'
    else:
        if args.maximum_far is None or args.maximum_frr is None:
            raise ValueError('Calibration requires explicit reviewed maximum FAR and FRR limits.')
        policy, report['held_out'] = calibrate(backend.model, results, manifest['dataset_id'],
                                             maximum_far=args.maximum_far, maximum_frr=args.maximum_frr)
        report['calibration'] = {k: v for k, v in asdict(policy).items() if k != 'model'}
        report['accepted_error_limits'] = {'maximum_far': args.maximum_far, 'maximum_frr': args.maximum_frr}
        report['limitation'] = 'Observed labelled-fixture error rates; review trial counts and deployment coverage before enabling.'
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['peak_rss_bytes'] = rss if sys.platform == 'darwin' else rss*1024
    return report


def bounded_worker(args):
    """Supervisor kills native work on timeout/RSS/disk breach; no enrollment side effects."""
    if shutil.disk_usage(args.output.resolve().parent).free < 40_000_000_000:
        raise ValueError('Keep at least 40 GB free before native model work.')
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker']
    started, peak = time.monotonic(), 0
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(command, stdout=output, stderr=output)
        try:
            while process.poll() is None:
                elapsed = time.monotonic()-started
                if elapsed > args.timeout_seconds:
                    raise ValueError('Voice measurement exceeded its wall-time budget.')
                try:
                    sample = subprocess.run(['ps', '-o', 'rss=', '-p', str(process.pid)], capture_output=True, text=True, timeout=2)
                except subprocess.TimeoutExpired:
                    raise ValueError('Resident-memory monitoring is unavailable; native voice work was stopped.') from None
                rss_text = sample.stdout.strip()
                if (sample.returncode or not rss_text) and process.poll() is None:
                    raise ValueError('Resident-memory monitoring is unavailable; native voice work was stopped.')
                if rss_text:
                    peak = max(peak, int(rss_text)*1024)
                if peak > args.maximum_rss_mib*1024*1024:
                    raise ValueError('Voice measurement exceeded its resident-memory budget.')
                if shutil.disk_usage(args.output.resolve().parent).free < 40_000_000_000:
                    raise ValueError('Voice measurement breached the 40 GB disk floor.')
                time.sleep(.5)
        finally:
            if process.poll() is None:
                process.kill(); process.wait()
        output.seek(0)
        body = output.read().decode(errors='replace')
        if process.returncode:
            raise ValueError(body[-4000:] or 'Voice measurement failed.')
    return {'seconds': time.monotonic()-started, 'supervised_peak_rss_bytes': peak}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--fixtures', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--compute-units', choices=COMPUTE_UNITS, default='ALL')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--maximum-far', type=float)
    parser.add_argument('--maximum-frr', type=float)
    parser.add_argument('--timeout-seconds', type=int, default=60)
    parser.add_argument('--maximum-rss-mib', type=int, default=1024)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.timeout_seconds <= 60 or not 64 <= args.maximum_rss_mib <= 1024:
        parser.error('Choose at most 60 seconds and 1024 MiB of resident memory.')
    owns_output = False
    try:
        if args.worker:
            report = measure(args)
            if report['peak_rss_bytes'] > args.maximum_rss_mib*1024*1024:
                raise ValueError('Voice measurement exceeded its peak resident-memory budget.')
            with args.output.open('x') as output:
                output.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
        else:
            if args.output.exists():
                raise ValueError('Choose a new measurement report path.')
            owns_output = True
            budget = bounded_worker(args)
            report = json.loads(args.output.read_text())
            report['run_budget'] = budget
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
            print(f'Measured report: {args.output}. Recognition remains disabled pending review.')
    except (ValueError, OSError, KeyError, TypeError) as error:
        if owns_output and args.output.exists():
            # The outer supervisor owns only this previously nonexistent report.
            # Keep failed native work from leaving a success-looking report.
            if not args.worker:
                args.output.unlink()
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
