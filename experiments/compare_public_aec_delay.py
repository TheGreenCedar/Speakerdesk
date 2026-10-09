"""Owned public CPU DSP matrix; no product edits, model/device/network calls."""
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch
import wave
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'speakerdesk'), str(ROOT/'scripts')]
import capture_echo
from capture_echo import EchoMixer, ReferenceClock, RATE, library_path
from live_meeting import SourceMixer
from audio import pcm16_bytes
from echo_fixture import speech
from echo_fixture import fixtures
from aec_delay_latch import DelayLatch
from run_render_copy_cpu_controls import NEW, OLD, PINS, pcm, sha

NATIVE_SHA = '6bf686f5ddd0e32dfa49d2127cf41c4bce293e015d2fa9621c54dc951e9addc6'
TRUTH = Path('/tmp/speakerdesk-speech-operator/public-sanity-fixture/TRUTH.json')


def strongest_clock():
    source = inspect.getsource(ReferenceClock)
    marker = 'if earlier:peak = earlier[-1]'
    assert source.count(marker) == 1
    scope = dict(np=np, RATE=RATE, LOOKAHEAD=capture_echo.LOOKAHEAD)
    exec(source.replace(marker, 'pass  # Isolated strongest-peak comparison only.'), scope)
    return scope['ReferenceClock']


def replay(far, mic, variant):
    outputs = []; history = []; bypass = []; sent_delays = []
    def factory(sink):
        echo = EchoMixer(sink)
        native = echo.lib.speakerdesk_echo_process; latch = DelayLatch()
        last_sent = None
        def call(state, render, near, clean, samples, delay):
            nonlocal last_sent
            c = echo.clock
            center = c.matches[-1][0] if c.matches else None
            applied = delay
            if variant in ('fixed77_oracle', 'fixed75') and delay >= 0:
                applied = 77 if variant == 'fixed77_oracle' else 75
            elif variant == 'block_hysteresis':
                applied = latch.observe(delay, center, c.end, c)
            refused = variant == 'block_hysteresis' and applied < 0
            key = (applied, center, c.echo, refused)
            if key != last_sent:
                sent_delays.append(dict(processed_start_sample=echo.position,
                    input_observed_sample=c.end, output_start_sample=max(0, echo.position-64),
                    match_center=center, sent_native_delay_ms=applied,
                    sent_native_block=applied//4 if applied >= 0 else None,
                    clock_echo_validated=c.echo, latch=latch.evidence() if variant == 'block_hysteresis' else None))
                last_sent = key
            result = native(state, render, near, clean, samples, applied)
            if refused:
                clean[:] = np.concatenate((echo.raw_delay, near))[:samples]
            a, b = max(0, echo.position-64), min(len(mic), echo.position+samples-64)
            raw = not c.echo or refused
            if a < b:
                if bypass and bypass[-1]['raw_bypass'] == raw and bypass[-1]['end_sample'] == a:
                    bypass[-1]['end_sample'] = b
                else: bypass.append(dict(start_sample=a, end_sample=b, raw_bypass=raw))
            return result
        echo.lib.speakerdesk_echo_process = call
        return echo
    clock = strongest_clock() if variant == 'strongest_peak' else ReferenceClock
    with patch.object(capture_echo, 'ReferenceClock', clock):
        mixer = SourceMixer(['microphone', 'system'],
            lambda mixed, tracks: outputs.append(tracks['microphone_clean'].copy()), echo_factory=factory)
        try:
            for start in range(0, len(far), RATE//4):
                end = min(len(far), start+RATE//4)
                for name, samples, packet in [('system', far, 752), ('microphone', mic, 496)]:
                    for a in range(start, end, packet):
                        b = min(end, a+packet)
                        mixer.add(name, a/RATE, samples[a:b].astype('<f4').tobytes())
                mixer.flush(end/RATE)
                c = mixer.echo.clock
                history.append(dict(observed_sample=end, echo_validated=c.echo,
                    delay_samples=c.matches[-1][1] if c.matches else None, skew=c.skew))
            mixer.flush(len(far)/RATE, final=True)
            assert mixer.late_samples == 0
        finally:
            mixer.echo.close()
    output = np.concatenate(outputs)
    assert len(output) == len(mic)
    assert bypass[0]['start_sample'] == 0 and bypass[-1]['end_sample'] == len(mic)
    assert sum(v['end_sample']-v['start_sample'] for v in bypass) == len(mic)
    return output, history, dict(output_raw_bypass_intervals=bypass, native_delay_events=sent_delays)


def metrics(clean, raw, far, near, first, last):
    y = clean[first:last].astype(np.float64); target = near[first:last].astype(np.float64)
    original = raw[first:last].astype(np.float64)
    error = y-target
    near_energy = float(target@target)
    gain = float(y@target/near_energy) if near_energy else None
    orthogonal = y-gain*target if near_energy else y
    a = max(first, 1232)
    x = far[a-1232:last-1232].astype(np.float64); residual = error[a-first:]
    reference_energy = float(x@x)
    coherent_gain = float(x@residual/reference_energy) if reference_energy else None
    coherent_fraction = (float((x@residual)**2/(reference_energy*(residual@residual)))
                         if reference_energy and residual@residual else None)
    def db(top, bottom):
        return 10*math.log10(max(float(top), 1e-30)/max(float(bottom), 1e-30))
    return dict(samples=[first, last], cleaned_RMS=float(np.sqrt(np.mean(y*y))),
        cleaned_peak=float(np.max(np.abs(y))), residual_RMS=float(np.sqrt(np.mean(error*error))),
        ERLE_dB=db((original-target)@(original-target), error@error),
        near_projection_gain=gain, near_orthogonal_distortion_RMS=float(np.sqrt(np.mean(orthogonal**2))),
        near_SNR_dB=db(near_energy, error@error) if near_energy else None,
        residual_reference_coherent_gain=coherent_gain, residual_reference_coherent_fraction=coherent_fraction)


def main():
    os.umask(0o077)
    assert sha(library_path()) == NATIVE_SHA
    for path, expected in PINS.items(): assert sha(path) == expected
    old, new = json.loads(OLD.read_bytes()), json.loads(NEW.read_bytes())
    cases = []
    for item in new['cases']:
        for value in item['files'].values(): assert sha(value['path']) == value['sha256']
        directory = Path(item['directory'])
        raw, far, near = [pcm(directory/(name+'.wav')).astype(np.float32)/32768
                          for name in ('microphone', 'system', 'known_near')]
        windows = []
        for i, event in enumerate(item['expected']['system']):
            a, b = event['start_sample']+1232, event['end_sample']+1232
            windows += [(f'remote{i}_startup1s', a, min(a+RATE, b)), (f'remote{i}_whole', a, b)]
        for i, event in enumerate(item['expected']['microphone_clean']):
            windows.append((f'genuine_near_repeat{i}', event['start_sample'], event['end_sample']))
        cases.append((item['name'], raw, far, near, windows, ['baseline', 'fixed77_oracle', 'fixed75', 'strongest_peak', 'block_hysteresis']))
    assert sha(TRUTH) == old['input_artifact_sha256'][str(TRUTH)]
    truth = json.loads(TRUTH.read_bytes())
    assert sha(truth['near_source']) == truth['near_source_sha256']
    pins = {p: h for p, h in old['input_artifact_sha256'].items() if p.endswith('.wav')}
    for p, h in pins.items(): assert sha(p) == h
    folder = Path(next(p for p in pins if p.endswith('/microphone.wav'))).parent
    raw, far = [pcm(folder/(name+'.wav')).astype(np.float32)/32768 for name in ('microphone', 'system')]
    near = np.zeros_like(raw); clip = pcm(truth['near_source'])[:truth['near_source_frames']].astype(np.float32)/32768
    for event in truth['near_insertions']:
        a = round(event['at']*RATE); near[a:a+len(clip)] += clip*np.float32(event['gain'])
    windows = [('remote_startup1s', RATE+1232, 2*RATE+1232), ('remote_before_near', RATE+1232, 6*RATE)]
    for condition in ('normal', 'quiet'):
        selected = next(c for c in old['cases'] if c['id'] == 'raw_mic-'+condition)
        windows.append((condition+'_overlap', *selected['near_interval_samples']))
    windows.append(('normal_near_solo', 7*RATE, 7*RATE+len(clip)))
    cases.append(('public28_overlap', raw, far, near, windows, ['baseline', 'fixed77_oracle', 'fixed75', 'strongest_peak', 'block_hysteresis']))
    # Protect the separately justified weak-direct/strong-reflection case.
    far = speech(11); far[9*RATE:] = 0
    near = speech(19)*.25; near[:4*RATE] = 0
    index = np.arange(len(far))
    echo = sum(g*np.interp(index-delay*RATE, index, far, left=0, right=0)
               for delay, g in [(.040, .3), (.045, .7), (.080, -.1)])
    cases.append(('earlier_path_generated_guard', (near+echo).astype('<f4'), far, near,
                  [('far_only', 2*RATE, 4*RATE), ('quiet_overlap', 5*RATE, 8*RATE)],
                  ['baseline', 'strongest_peak', 'block_hysteresis']))
    for name, far, near, echo in [('plus180ppm_generated_guard', *fixtures()['drift']),
                                 ('doubletalk_from_start_guard', *fixtures()['echo'])]:
        if name == 'doubletalk_from_start_guard': near = speech(19)
        cases.append((name, (near+echo).astype('<f4'), far, near,
                      [('startup', 0, RATE), ('far_only', 2*RATE, 4*RATE), ('overlap', 5*RATE, 8*RATE)],
                      ['baseline', 'block_hysteresis']))
    far = speech(11); far[9*RATE:] = 0; near = speech(41); near[:4*RATE] = 0
    index = np.arange(len(far))
    echo = sum(g*np.interp(index*(1-220e-6)-(.312+tail)*RATE, index, far, left=0, right=0)
               for tail, g in ((0, .62), (.011, .15), (.067, -.08), (.109, .035)))
    cases.append(('minus220ppm_generated_guard', (near+echo).astype('<f4'), far, near,
                  [('far_only', 3*RATE, 4*RATE), ('overlap', 5*RATE, 8*RATE)], ['baseline', 'block_hysteresis']))
    far = speech(7); near = speech(19)*.25; near[:4*RATE] = 0
    index = np.arange(len(far)); delayed = np.where(index < 6*RATE, index-.063*RATE, index-.237*RATE)
    echo = .52*np.interp(delayed, index, far, left=0, right=0)
    cases.append(('real_route_jump_generated_guard', (near+echo).astype('<f4'), far, near,
                  [('transition', 6*RATE, 7*RATE), ('reacquired', 8*RATE, 10*RATE)], ['baseline', 'block_hysteresis']))
    parent = Path('/private/tmp/speakerdesk-attribution-private-9yk6lg11')
    assert parent.stat().st_uid == os.getuid() and parent.stat().st_mode & 0o777 == 0o700 and not parent.is_symlink()
    output = Path(tempfile.mkdtemp(prefix='public-aec-delay-matrix-', dir=parent)); rows = []
    started = time.monotonic()
    for name, raw, far, near, windows, variants in cases:
        for variant in variants:
            cleaned, history, timing = replay(far, raw, variant)
            path = output/(name+'-'+variant+'.wav')
            with wave.open(str(path), 'wb') as target:
                target.setparams((1, 2, RATE, 0, 'NONE', 'none')); target.writeframes(pcm16_bytes(cleaned))
            rows.append(dict(case=name, variant=variant, frames=len(cleaned), history=history, **timing,
                output_path=str(path), output_sha256=sha(path),
                windows={label: metrics(cleaned, raw, far, near, a, b) for label, a, b in windows}))
    result = dict(scope='CPU_DSP_PUBLIC_DELAY_DISCRIMINATOR', model_calls=0, device_calls=0,
        product_source_changed=False, actual_stereo_provenance=False,
        reference_scope='Existing mono public controls; native receives duplicated mono, not claimed physical stereo.',
        output_directory=str(output), duration_seconds=time.monotonic()-started, native_sha256=NATIVE_SHA,
        source_sha256=sha(__file__), latch_source_sha256=sha(ROOT/'experiments/aec_delay_latch.py'), capture_echo_sha256=sha(ROOT/'speakerdesk/capture_echo.py'),
        input_sha256={str(p): sha(p) for p in [NEW, OLD, TRUTH]}, rows=rows,
        limits=['fixed77 uses fixture truth and is an oracle, never production calibration',
                'Native replay starts from decoded retained PCM16; earlier production capture used unquantized Float32, so exact output equality is not expected.',
                'ABI3 exports no ERL/ERLE/delay metrics; reported quantities are independent waveform diagnostics.',
                'No transcript, VAD/diarizer, real hardware or private speech result is supplied.'])
    report = output/'REPORT.json'; report.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(report=str(report), sha256=sha(report), seconds=result['duration_seconds'],
        summary=[dict(case=r['case'], variant=r['variant'], windows=r['windows']) for r in rows]), indent=2))


if __name__ == '__main__': main()
