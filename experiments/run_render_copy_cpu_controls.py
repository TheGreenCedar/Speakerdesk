"""Retained public PCM comparisons; no model imports, devices or audio writes."""
import hashlib
import json
from pathlib import Path
import wave
import numpy as np
from render_copy_diagnostic import diagnose

NEW = Path('/private/tmp/speakerdesk-attribution-private-9yk6lg11/source-echo-repeat-controls-27e5bfbfc5d5/INPUTS.json')
OLD = Path('/private/tmp/speakerdesk-attribution-private-9yk6lg11/public-channel-study-eight-v1/INPUTS.json')
REVIEW = Path('/private/tmp/speakerdesk-render-certificate-review-task10-45uqgm4e/REPORT.json')
REVIEW_SHA = 'bb78554c87e625cc8e3504c285be85ba3f9f2cbae6b704dd773813cdf299ec21'
PINS = {NEW: 'f1574eefdc5eabba2207e88d686ea326d6e05ad2e74e21739968c6dc3bff64ee',
        OLD: '85636c7ab995c99cb7acca56290342bf797dcae9ac5d31c62eded3a57312aeed'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pcm(path):
    with wave.open(str(path), 'rb') as source:
        assert (source.getnchannels(), source.getsampwidth(), source.getframerate()) == (1, 2, 16000)
        count = source.getnframes()
        result = np.frombuffer(source.readframes(count), dtype='<i2').copy()
        assert len(result) == count
        return result


def fixed_mono(raw, far, start, end, lag=1232, numerator=18, denominator=100):
    """Arithmetic only; this mono measurement is not the stereo diagnostic."""
    if not 0 <= start-lag < end-lag <= len(far):
        return {'reason': 'missing_causal_reference', 'production_admissible': False}
    x, y = far[start-lag:end-lag].astype(np.int64), raw[start:end].astype(np.int64)
    error = 2*np.abs(denominator*y-numerator*x)
    bound = denominator+abs(numerator)
    failed = np.flatnonzero(error > bound)
    return {'production_admissible': False, 'scope': 'fixed_declared_mono_fixture_path_arithmetic',
            'anchor': [start, end], 'lag_samples': lag, 'numerator': numerator,
            'denominator': denominator, 'fixed_rounding_bound_LSB': bound/(2*denominator),
            'maximum_error_LSB': float(error.max())/(2*denominator),
            'fits_per_sample_bound': not len(failed),
            'first_unexplained_sample': start+int(failed[0]) if len(failed) else None,
            'reference_energy_LSB_squared': int(x@x)}


def run():
    for path, expected in PINS.items():
        assert sha(path) == expected
    assert sha(REVIEW) == REVIEW_SHA
    new, old = json.loads(NEW.read_bytes()), json.loads(OLD.read_bytes())
    for case in new['cases']:
        for item in case['files'].values():
            assert sha(item['path']) == item['sha256']
    echo = next(c for c in new['cases'] if c['name'] == 'echo_only')
    root = Path(echo['directory'])
    raw, far = pcm(root/'microphone.wav'), pcm(root/'system.wav')
    a, b = 14208, 99968
    rows = {'actual_echo_only_full_anchor': fixed_mono(raw, far, a, b),
            'actual_control_missing_bound_stereo_provenance': diagnose(raw, far, {}, {})}
    modified = raw.copy(); modified[a] = 4
    rows['actual_anchor_single_4_LSB_context_innovation'] = fixed_mono(modified, far, a, b)
    modified = raw.copy()
    modified[a:b] = np.rint(.184*far[a-1232:b-1232]).astype(np.int16)
    rows['above_resolution_collinear_near_against_prior_018'] = fixed_mono(modified, far, a, b)
    rows['same_collinear_world_against_free_0184'] = fixed_mono(modified, far, a, b, numerator=184, denominator=1000)
    repeat = next(c for c in new['cases'] if c['name'] == 'genuine_repeat')
    root = Path(repeat['directory']); raw, far = pcm(root/'microphone.wav'), pcm(root/'system.wav')
    for i, event in enumerate(repeat['expected']['microphone_clean']):
        a, b = event['start_sample'], event['end_sample']
        rows[f'actual_genuine_repeat_{i}_fixed_causal_path'] = fixed_mono(raw, far, a, b)
        if i == 0:
            rows['actual_genuine_repeat_0_unsafe_nine_second_reference'] = fixed_mono(raw, far, a, b, lag=144000, numerator=1, denominator=1)
    pins = {path: digest for path, digest in old['input_artifact_sha256'].items() if path.endswith('.wav')}
    for path, digest in pins.items():
        assert sha(path) == digest
    folder = Path(next(p for p in pins if p.endswith('/microphone.wav'))).parent
    raw, far = pcm(folder/'microphone.wav'), pcm(folder/'system.wav')
    for condition in ('normal', 'quiet'):
        case = next(c for c in old['cases'] if c['id'] == 'raw_mic-'+condition)
        rows[f'actual_retained_{condition}_overlap'] = fixed_mono(raw, far, *case['anchor_samples'])
    assert rows['actual_echo_only_full_anchor']['fits_per_sample_bound']
    assert not rows['actual_anchor_single_4_LSB_context_innovation']['fits_per_sample_bound']
    assert not rows['above_resolution_collinear_near_against_prior_018']['fits_per_sample_bound']
    assert rows['same_collinear_world_against_free_0184']['fits_per_sample_bound']
    assert rows['actual_genuine_repeat_0_unsafe_nine_second_reference']['maximum_error_LSB'] == 0
    for key in ('actual_genuine_repeat_0_fixed_causal_path', 'actual_genuine_repeat_1_fixed_causal_path',
                'actual_retained_normal_overlap', 'actual_retained_quiet_overlap'):
        assert not rows[key]['fits_per_sample_bound']
    return {'CPU_arithmetic_only': True, 'model_calls': 0, 'device_calls': 0,
            'private_audio': False, 'production_admission_changed': False,
            'input_sha256': {str(path): sha(path) for path in PINS},
            'independent_review_sha256': REVIEW_SHA, 'rows': rows,
            'note': 'Declared synthetic mono-path arithmetic is separate from the stereo diagnostic. No production predicate receives known-near truth.'}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
