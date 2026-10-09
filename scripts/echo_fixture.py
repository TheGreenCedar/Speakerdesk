"""Deterministic, file-backed capture fixtures. Never opens an audio device.

All signals are generated locally. The expected near-end signal and acoustic
path are independent of the capture implementation under test.
"""
import json
import argparse
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'speakerdesk'))
from audio import pcm16_bytes
from live_meeting import SourceMixer, RATE


def speech(seed, seconds=12):
    rng = np.random.default_rng(seed)
    n = round(seconds * RATE)
    t = np.arange(n) / RATE
    # Aperiodic excitation with independent harmonic and syllabic components.
    noise = rng.normal(0, 1, n)
    colored = np.convolve(noise, np.array([.2, .5, .2, -.1]), mode='same')
    voiced = sum(np.sin(2*np.pi*(113+seed*7)*k*t + .5*np.sin(2*np.pi*2*t))/k
                 for k in range(1, 9))
    envelope = (.15 + .85*np.sin(2*np.pi*2.7*t)**2) * (np.sin(2*np.pi*.7*t) > -.5)
    return ((.045*colored + .04*voiced)*envelope).astype('<f4')


def fixtures():
    remote = speech(7)
    near = speech(19)
    # First four seconds far-only, then double talk, then near-only.
    near[:4*RATE] = 0
    remote[9*RATE:] = 0
    indexes = np.arange(len(remote))
    cases = {}
    for name, delay, drift in [('echo', .063, 0), ('latency', .237, 0), ('drift', .081, 180e-6)]:
        echo = np.zeros_like(remote)
        for tail, gain in [(0, .52), (.013, -.19), (.047, .11), (.091, .04)]:
            echo += gain*np.interp(indexes*(1+drift)-(delay+tail)*RATE,
                                  indexes, remote, left=0, right=0)
        cases[name] = (remote, near, echo)
    cases['no_echo'] = (remote, near, np.zeros_like(remote))
    cases['near_only'] = (np.zeros_like(remote), near, np.zeros_like(remote))
    return cases


def replay(remote, microphone, *, baseline=False):
    blocks = []
    options={'echo_factory':None} if baseline else {}
    mixer = SourceMixer(['microphone', 'system'], lambda mixed, tracks: blocks.append(mixed.copy()),**options)
    # Deliberately asynchronous 31/47 ms packets; timestamp rounding is the
    # same transport boundary as the native helper. Flush after both arrive.
    for start in range(0, len(remote), RATE//4):
        end = min(start+RATE//4, len(remote))
        for source, audio, packet in [('microphone', microphone, 496), ('system', remote, 752)]:
            for pos in range(start, end, packet):
                mixer.add(source, pos/RATE, audio[pos:min(pos+packet,end)].astype('<f4').tobytes(),
                          audio.shape[1] if audio.ndim==2 else 1)
        mixer.flush(end/RATE)
    mixer.flush(len(remote)/RATE, final=True)
    if mixer.echo:mixer.echo.close()
    return np.concatenate(blocks)


def write_wav(path, audio):
    with wave.open(str(path), 'wb') as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(RATE)
        out.writeframes(pcm16_bytes(audio))


def read_wav(path):
    with wave.open(str(path), 'rb') as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (1, 2, RATE):
            raise ValueError('Expected canonical PCM16 mono fixture.')
        return (np.frombuffer(source.readframes(source.getnframes()), dtype='<i2')/32768).astype('<f4')


def run(destination, *, baseline=False):
    destination.mkdir(parents=True, exist_ok=False)
    metrics = {}
    for name, (remote, near, echo) in fixtures().items():
        case = destination/name; case.mkdir(exist_ok=True)
        for track, audio in [('system',remote),('microphone',near+echo),('near_expected',near)]:
            write_wav(case/(track+'.wav'), audio)
        # Exercise transport with bytes read from the persisted PCM fixtures.
        remote = read_wav(case/'system.wav')
        mic = read_wav(case/'microphone.wav')
        near = read_wav(case/'near_expected.wav')
        echo = mic-near  # Includes the independently quantized capture noise.
        mixed = replay(remote, mic,baseline=baseline)
        residual = mixed-remote-near
        segment = slice(2*RATE, 4*RATE)
        erle = 10*np.log10((np.sum(echo[segment]**2)+1e-20)/(np.sum(residual[segment]**2)+1e-20))
        segment = slice(5*RATE, 8*RATE)
        gain = np.dot(mixed[segment]-remote[segment], near[segment])/np.dot(near[segment], near[segment])
        snr = 10*np.log10((np.sum(near[segment]**2)+1e-20)/(np.sum(residual[segment]**2)+1e-20))
        metrics[name] = {'far_only_echo_reduction_db':float(erle), 'double_talk_near_gain':float(gain),
                         'double_talk_near_snr_db':float(snr),
                         'expected_mix_error_rms':float(np.sqrt(np.mean(residual**2))), 'samples':len(mixed)}
        for track, audio in [('echo_expected',echo),('mixed',mixed)]:
            write_wav(case/(track+'.wav'), audio)
    (destination/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    return metrics


def require_clean(metrics):
    failures = []
    for name, result in metrics.items():
        if result['samples'] != 12*RATE:
            failures.append(f'{name}: timeline changed')
        if not .95 <= result['double_talk_near_gain'] <= 1.05:
            failures.append(f'{name}: near-end gain changed')
        minimum_snr = 30 if name in ('no_echo', 'near_only') else 15
        if result['double_talk_near_snr_db'] < minimum_snr:
            failures.append(f'{name}: near-end residual SNR below {minimum_snr} dB')
        if name not in ('no_echo', 'near_only') and result['far_only_echo_reduction_db'] < 20:
            failures.append(f'{name}: echo reduction below 20 dB')
    if failures:
        raise AssertionError('; '.join(failures))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path, help='New local-only directory; existing outputs are never overwritten.')
    parser.add_argument('--baseline',action='store_true',help='Replay the original uncorrected mixer for comparison only.')
    parser.add_argument('--assert-clean', action='store_true', help='Enforce the frozen echo/near-end contract (fails on the uncorrected mixer).')
    args = parser.parse_args()
    metrics = run(args.destination,baseline=args.baseline)
    print(json.dumps(metrics, indent=2), flush=True)
    if args.assert_clean:
        require_clean(metrics)
