"""Bounded native ABI-5 DSP probe, without devices, models or fitted output gain.

Use the two immutable public WAVs retained by host/tests/echo_quality.rs in
--voices, the actual --library, and a fresh --output directory. This is a
simulated-room regression, not physical speaker/microphone qualification.
"""
import argparse
import array
import ctypes
import hashlib
import json
import math
import sys
import wave
from pathlib import Path

RATE, FRAME, DELAY = 16000, 160, 160
SAMPLES, WARMUP = 320073, 64000
VOICES = {
    'far': ('c7a783b45d4fac62025b17c4f43f8b72bfc32b09eb3ad4c3d57ceda4bfc76c49', 90880),
    'near': ('1a0efbcb86807b14d1667d91fcfb38502a8b380ee1c8c0e3448db8b29bad82bf', 40240),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def voice(root, name):
    pin, samples = VOICES[name]
    path = root / (pin + '.wav')
    assert sha(path.read_bytes()) == pin
    with wave.open(str(path)) as reader:
        assert (reader.getnchannels(), reader.getframerate(), reader.getsampwidth(), reader.getnframes()) == (1, RATE, 2, samples)
        pcm = array.array('h', reader.readframes(samples))
    if sys.byteorder != 'little':
        pcm.byteswap()
    values = [v / 32768. for v in pcm]
    gain = .08 / math.sqrt(sum(v*v for v in values) / len(values))
    return [values[i % (samples + 3200)] * gain if i % (samples + 3200) < samples else 0. for i in range(SAMPLES)]


def process(library, microphone, reference):
    state = library.speakerdesk_echo_create()
    assert state
    output = []
    try:
        for start in range(0, len(microphone) + DELAY + FRAME, FRAME):
            mic = microphone[start:start+FRAME]
            far = reference[start:start+FRAME]
            # Match EchoMixer's native-input contract; retained raw PCM is unchanged.
            mic = [max(-1., min(1., value)) for value in mic]
            a = (ctypes.c_float * FRAME)(*(mic + [0.] * (FRAME-len(mic))))
            b = (ctypes.c_float * (FRAME*2))(*(value for v in far + [0.] * (FRAME-len(far)) for value in (v, 0.)))
            cleaned = (ctypes.c_float * FRAME)()
            # No external delay: the internal estimator must work from startup.
            assert library.speakerdesk_echo_process(state, b, a, cleaned, FRAME, -1) == 0
            output.extend(cleaned)
    finally:
        library.speakerdesk_echo_destroy(state)
    return output


def metrics(output, microphone, target, start):
    clean = output[start:]
    mic, near = microphone[start:], target[start:]
    error = sum((a-b)**2 for a, b in zip(clean, near))
    original_error = sum((a-b)**2 for a, b in zip(mic, near))
    target_energy = sum(v*v for v in near)
    clean_energy = sum(v*v for v in clean)
    cross = sum(a*b for a, b in zip(clean, near))
    return {
        'error_reduction_db': 10 * math.log10(max(original_error, 1e-30) / max(error, 1e-30)),
        'near_projection_gain': cross/target_energy if target_energy else None,
        'near_correlation': cross/math.sqrt(clean_energy*target_energy) if clean_energy*target_energy else None,
        'near_error_nrms': math.sqrt(error/target_energy) if target_energy else None,
        'clean_rms': math.sqrt(clean_energy/len(clean)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--voices', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir()
    report = {
        'passed': False, 'abi': 5, 'sample_rate': RATE, 'samples': SAMPLES,
        'latency_samples': DELAY, 'noise_policy': 'upstream NsConfig default k12dB',
        'gain_control': False, 'external_delay_ms': -1,
        'native_input': 'Same [-1,1] clamp as EchoMixer; raw input artifacts retain unclamped values',
        'source_sha256': sha(Path(__file__).read_bytes()), 'library_sha256': sha(args.library.read_bytes()),
        'limits': 'Fixed public-voice simulated FIR/noise regression; no physical-room or intelligibility qualification.',
        'cases': [],
        'gates': {'far_only_db': 6, 'noise_only_db': 3, 'near_only_correlation': .9,
                  'near_projection_gain': [.5, 1.2], 'noise_voice_and_double_talk': 'target error improves; projection gain >= .5',
                  'double_talk_startup': 'first4s target error non-worsening and near projection gain >= .8',
                  'geometry': 'finite, exact length; measured delay 160; no fitted lag/gain applied to quality metrics'},
    }
    try:
        library = ctypes.CDLL(str(args.library.resolve()))
        library.speakerdesk_echo_create.restype = ctypes.c_void_p
        library.speakerdesk_echo_destroy.argtypes = [ctypes.c_void_p]
        library.speakerdesk_echo_process.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_int]
        assert library.speakerdesk_echo_abi() == 5
        far, near = voice(args.voices, 'far'), voice(args.voices, 'near')
        noise, seed = [], 7
        for _ in range(SAMPLES):
            seed = (1664525 * seed + 1013904223) & 0xffffffff
            noise.append(((seed >> 8) / 16777216. - .5) * math.sqrt(12) * .03)
        echo = [sum(far[i-d]*g for d, g in [(1120, .55), (1200, -.15), (1376, .08), (1696, .04)] if i >= d) for i in range(SAMPLES)]
        zero = [0.] * SAMPLES
        for name in ['far_only', 'near_only', 'noise_only', 'noise_voice', 'double_talk']:
            target = zero if name in ('far_only', 'noise_only') else near
            reference = far if name in ('far_only', 'double_talk') else zero
            microphone = [target[i] + (echo[i] if reference is far else 0.) + (noise[i] if name.startswith('noise_') else 0.) for i in range(SAMPLES)]
            def save(label, values):
                data = array.array('f', values)
                if sys.byteorder != 'little': data.byteswap()
                raw = data.tobytes()
                path = args.output / (name+'-'+label+'.f32')
                path.write_bytes(raw)
                return {'file': path.name, 'bytes': len(raw), 'sha256': sha(raw)}
            result = {'case': name, 'passed': False,
                      'clamped_input_samples': sum(abs(v) > 1 for v in microphone),
                      'microphone': save('microphone', microphone), 'reference': save('reference', reference),
                      'target': save('target', target)}
            report['cases'].append(result)
            raw_output = process(library, microphone, reference)
            clean = raw_output[DELAY:DELAY+SAMPLES]
            geometry = len(clean) == SAMPLES and all(math.isfinite(v) for v in clean)
            result.update(geometry_finite=geometry, clean=save('clean', clean))
            result.update(metrics(clean, microphone, target, WARMUP))
            if name in ('near_only', 'double_talk'):
                result['startup_first_four_seconds'] = metrics(clean[:WARMUP], microphone[:WARMUP], target[:WARMUP], 0)
            if name == 'near_only':
                # Measure latency independently; never fit alignment of quality metrics.
                points = range(WARMUP, WARMUP+8000, 8)
                scores = [sum(microphone[i]*raw_output[i+d] for i in points) for d in range(384)]
                result['measured_delay_samples'] = max(range(384), key=scores.__getitem__)
                passed = result['measured_delay_samples'] == DELAY and result['near_correlation'] >= .9 and .5 <= result['near_projection_gain'] <= 1.2
            elif name in ('far_only', 'noise_only'):
                passed = result['error_reduction_db'] >= (6 if name == 'far_only' else 3)
            else:
                passed = result['error_reduction_db'] > 0 and .5 <= result['near_projection_gain'] <= 1.2
            if name == 'double_talk':
                startup = result['startup_first_four_seconds']
                passed = passed and startup['error_reduction_db'] >= 0 and startup['near_projection_gain'] >= .8
            result['passed'] = geometry and passed
        report['passed'] = all(case['passed'] for case in report['cases'])
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        (args.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'passed': report['passed'], 'report': str(args.output/'report.json')}))
    assert report['passed'], 'Native DSP regression failed; retained raw results must be reviewed'


if __name__ == '__main__':
    main()
