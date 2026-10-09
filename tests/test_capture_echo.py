"""File/DSP-only regressions. Never starts capture, a model, or audio playback."""
import ctypes
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'speakerdesk'),str(ROOT/'scripts')]
from capture_echo import EchoMixer, ReferenceClock, library_path, RATE, FRAME, DELAY, LOOKAHEAD
from live_meeting import SourceMixer
from echo_fixture import fixtures, replay, require_clean, run, speech


@unittest.skipUnless(library_path().is_file(),'Build the pinned macOS DSP with scripts/setup_echo.py')
class NativeEchoTests(unittest.TestCase):
    def test_clean_microphone_retains_compensated_samples_even_when_mix_clips(self):
        events=[];processor=EchoMixer(lambda mixed,tracks:events.append((mixed.copy(),{s:a.copy() for s,a in tracks.items()})))
        self.addCleanup(processor.close)
        mic=np.full(5003,.5,dtype=np.float32);far=np.full(5003,.8,dtype=np.float32)
        processor.add({'microphone':mic[:3001],'system':far[:3001]})
        processor.add({'microphone':mic[3001:],'system':far[3001:]},final=True)
        mixed=np.concatenate([e[0] for e in events])
        clean=np.concatenate([e[1]['microphone_clean'] for e in events])
        raw=np.concatenate([e[1]['microphone'] for e in events])
        system=np.concatenate([e[1]['system'] for e in events])
        np.testing.assert_array_equal(raw,mic);np.testing.assert_array_equal(system,far)
        np.testing.assert_array_equal(clean,mic);np.testing.assert_array_equal(mixed,np.ones_like(mic))
        self.assertTrue(np.any(clean != mixed-system))

    def test_file_backed_latency_drift_and_speech_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            require_clean(run(Path(folder)/'fixtures'))

    def test_native_delay_is_measured_and_api_rejects_invalid_frames(self):
        processor=EchoMixer(lambda mixed,tracks:None)
        self.addCleanup(processor.close)
        lib=processor.lib
        far=np.zeros(FRAME*2,np.float32)
        mic=np.zeros(FRAME,np.float32);mic[0]=.3
        first=np.empty(FRAME,np.float32);second=np.empty(FRAME,np.float32)
        self.assertEqual(lib.speakerdesk_echo_process(processor.state,far,mic,first,FRAME,0),0)
        self.assertEqual(lib.speakerdesk_echo_process(processor.state,far,np.zeros_like(mic),second,FRAME,0),0)
        actual=np.concatenate((first,second))
        self.assertEqual(int(np.argmax(np.abs(actual))),DELAY)
        self.assertAlmostEqual(float(actual[DELAY]),.3,places=5)
        self.assertEqual(lib.speakerdesk_echo_process(processor.state,far,mic,first,FRAME-1,0),-1)
        mic[0]=np.nan
        self.assertEqual(lib.speakerdesk_echo_process(processor.state,far,mic,first,FRAME,0),-1)

    def test_partial_frame_pause_resume_and_empty_stop_keep_every_original_sample(self):
        blocks=[];originals=[]
        mixer=SourceMixer(['microphone','system'],lambda mixed,tracks:blocks.append(mixed.copy()),
                          raw_sink=lambda tracks:originals.append(tracks['microphone'].copy()))
        self.addCleanup(mixer.echo.close)
        rng=np.random.default_rng(531)
        a=(rng.normal(0,.03,1237)).astype('<f4');b=(rng.normal(0,.03,283)).astype('<f4')
        mixer.add('microphone',0,a.tobytes());mixer.flush(len(a)/RATE,final=True)
        mixer.add('microphone',len(a)/RATE,b.tobytes());mixer.flush((len(a)+len(b))/RATE,final=True)
        mixer.flush((len(a)+len(b))/RATE,final=True)
        np.testing.assert_array_equal(np.concatenate(blocks),np.concatenate((a,b)))
        np.testing.assert_array_equal(np.concatenate(originals),np.concatenate((a,b)))
        self.assertEqual(mixer.cursor,len(a)+len(b))

    def test_independent_stereo_speaker_paths_are_cancelled_before_mono_mix(self):
        left=speech(7);right=speech(29);left[9*RATE:]=0;right[9*RATE:]=0
        near=speech(19);near[:4*RATE]=0
        reference=np.column_stack((left,right)).astype('<f4')
        index=np.arange(len(left));echo=np.zeros_like(left)
        for signal,delay,gain in [(left,.071,.65),(right,.071,.3),(left,.098,-.12),(right,.143,.06)]:
            echo+=gain*np.interp(index-delay*RATE,index,signal,left=0,right=0)
        mic=(near+echo).astype('<f4');blocks=[]
        mixer=SourceMixer(['microphone','system'],lambda mixed,tracks:blocks.append(mixed.copy()))
        self.addCleanup(mixer.echo.close)
        for start in range(0,len(left),4000):
            end=start+4000
            mixer.add('microphone',start/RATE,mic[start:end].tobytes())
            mixer.add('system',start/RATE,reference[start:end].tobytes(),2)
            mixer.flush(end/RATE)
        mixer.flush(len(left)/RATE,final=True)
        mixed=np.concatenate(blocks);remote=reference.mean(axis=1);residual=mixed-remote-near
        far=slice(2*RATE,4*RATE);double=slice(5*RATE,8*RATE)
        erle=10*np.log10(np.sum(echo[far]**2)/np.sum(residual[far]**2))
        gain=np.dot(mixed[double]-remote[double],near[double])/np.dot(near[double],near[double])
        self.assertGreater(erle,20);self.assertGreater(gain,.95);self.assertLess(gain,1.05)
        self.assertEqual(len(mixed),len(left))

    def test_double_talk_from_start_preserves_near_speech_and_improves_residual(self):
        for name in ('echo','latency','drift'):
            with self.subTest(name=name):
                far,_,echo=fixtures()[name];near=speech(19);mic=near+echo
                mixed=replay(far,mic);original=replay(far,mic,baseline=True)
                segment=slice(5*RATE,8*RATE)
                gain=np.dot(mixed[segment]-far[segment],near[segment])/np.dot(near[segment],near[segment])
                residual=mixed[segment]-far[segment]-near[segment]
                baseline=original[segment]-far[segment]-near[segment]
                self.assertGreater(gain,.95);self.assertLess(gain,1.05)
                # There is no far-only training period: require measurable
                # improvement without disguising the harder residual as clean.
                self.assertLess(np.sum(residual**2),.5*np.sum(baseline**2))

    def test_first_path_before_stronger_reflection_preserves_quiet_double_talk(self):
        far = speech(11); far[9*RATE:] = 0
        near = speech(19)*.25; near[:4*RATE] = 0
        index = np.arange(len(far))
        echo = sum(gain*np.interp(index-delay*RATE, index, far, left=0, right=0)
                   for delay, gain in [(.040, .3), (.045, .7), (.080, -.1)])
        mixed = replay(far, (near+echo).astype('<f4'))
        residual = mixed-far-near
        segment = slice(2*RATE, 4*RATE)
        erle = 10*np.log10(np.sum(echo[segment]**2)/np.sum(residual[segment]**2))
        self.assertGreater(erle, 20)
        segment = slice(5*RATE, 8*RATE)
        gain = np.dot(mixed[segment]-far[segment], near[segment])/np.dot(near[segment], near[segment])
        self.assertGreater(gain, .95); self.assertLess(gain, 1.05)

    def test_no_echo_different_speakers_never_changes_canonical_audio(self):
        far=speech(7)
        for seed in (11,19,29,41):
            with self.subTest(seed=seed):
                near=speech(seed)
                np.testing.assert_array_equal(replay(far,near),np.clip(far+near,-1,1))

    def test_quiet_near_speech_is_preserved_during_double_talk(self):
        far,near,echo=fixtures()['echo']
        segment=slice(5*RATE,8*RATE)
        for scale in (.5,.25,.1):
            with self.subTest(scale=scale):
                local=near*scale
                mixed=replay(far,local+echo)
                gain=np.dot(mixed[segment]-far[segment],local[segment])/np.dot(local[segment],local[segment])
                self.assertGreater(gain,.95);self.assertLess(gain,1.05)

    def test_correlated_tones_are_not_evidence_of_acoustic_echo(self):
        t=np.arange(12*RATE)/RATE
        far=(.12*np.sin(2*np.pi*437*t)).astype('<f4')
        local=(.07*np.sin(2*np.pi*437*t+.4)).astype('<f4')
        np.testing.assert_array_equal(replay(far,local),np.clip(far+local,-1,1))

    def test_phase_inverted_stereo_retains_remote_and_local_content(self):
        far=speech(7);near=speech(19)
        reference=np.column_stack((far,-far)).astype('<f4')
        np.testing.assert_array_equal(replay(reference,near),np.clip(far+near,-1,1))

    def test_other_voices_long_room_tail_and_negative_clock_drift(self):
        far=speech(11);far[9*RATE:]=0;near=speech(41);near[:4*RATE]=0
        index=np.arange(len(far));double=slice(5*RATE,8*RATE);trained=slice(3*RATE,4*RATE)
        for delay,drift in ((.017,0),(.312,-220e-6)):
            with self.subTest(delay=delay,drift=drift):
                echo=sum(gain*np.interp(index*(1+drift)-(delay+tail)*RATE,index,far,left=0,right=0)
                         for tail,gain in ((0,.62),(.011,.15),(.067,-.08),(.109,.035)))
                mixed=replay(far,(near+echo).astype('<f4'));residual=mixed-far-near
                gain=np.dot(mixed[double]-far[double],near[double])/np.dot(near[double],near[double])
                self.assertGreater(gain,.95);self.assertLess(gain,1.05)
                # Additional voices/longer room tail use a later settling window;
                # the original frozen five-case thresholds remain unchanged.
                erle=10*np.log10(np.sum(echo[trained]**2)/np.sum(residual[trained]**2))
                snr=10*np.log10(np.sum(near[double]**2)/np.sum(residual[double]**2))
                self.assertGreater(erle,15);self.assertGreater(snr,15)

    def test_inverted_acoustic_polarity_still_cancels_echo(self):
        far,near,positive=fixtures()['echo'];echo=-positive
        mixed=replay(far,(near+echo).astype('<f4'));residual=mixed-far-near
        far_only=slice(2*RATE,4*RATE);double=slice(5*RATE,8*RATE)
        erle=10*np.log10(np.sum(echo[far_only]**2)/np.sum(residual[far_only]**2))
        gain=np.dot(mixed[double]-far[double],near[double])/np.dot(near[double],near[double])
        self.assertGreater(erle,20);self.assertGreater(gain,.95);self.assertLess(gain,1.05)

    def test_format_boundaries_drain_pending_audio_without_loss(self):
        blocks=[]
        mixer=SourceMixer(['microphone','system'],lambda mixed,tracks:blocks.append(mixed.copy()))
        self.addCleanup(mixer.echo.close)
        local=np.random.default_rng(51).normal(0,.03,24000).astype('<f4')
        # Boundaries are deliberately inside a tile and a native frame.
        for boundary in (1237,10003):mixer.format_changed('system',boundary/RATE)
        for start in range(0,len(local),4000):
            mixer.add('microphone',start/RATE,local[start:start+4000].tobytes())
            mixer.flush((start+4000)/RATE)
        mixer.flush(len(local)/RATE,final=True)
        np.testing.assert_array_equal(np.concatenate(blocks),local)
        self.assertEqual(mixer.reset_points,[])

    def test_original_tracks_survive_a_dsp_failure(self):
        raw=[]
        mixer=SourceMixer(['microphone','system'],lambda mixed,tracks:None,
                          raw_sink=lambda tracks:raw.append(tracks['microphone'].copy()))
        self.addCleanup(mixer.echo.close)
        mic=np.linspace(-.3,.3,4000,dtype='<f4')
        mixer.add('microphone',0,mic.tobytes())
        with patch.object(mixer.echo,'add',side_effect=RuntimeError('injected DSP failure')):
            with self.assertRaisesRegex(RuntimeError,'DSP failure'):mixer.flush(.25,final=True)
        np.testing.assert_array_equal(np.concatenate(raw),mic)
        self.assertEqual(mixer.cursor,4000)

    def test_streaming_memory_and_clock_phase_are_bounded(self):
        processor=EchoMixer(lambda mixed,tracks:None);self.addCleanup(processor.close)
        zero=np.zeros(4000,np.float32)
        for _ in range(160):
            processor.add({'system':zero,'microphone':zero})
            self.assertLessEqual(len(processor.clock.far),RATE*2)
            self.assertLessEqual(len(processor.input_mic),LOOKAHEAD+FRAME)
            self.assertLessEqual(len(processor.pending['microphone']),LOOKAHEAD+FRAME+DELAY)
        processor.add({'system':zero[:0],'microphone':zero[:0]},final=True)


class EchoBoundaryTests(unittest.TestCase):
    def test_missing_component_fails_closed_before_any_audio(self):
        with patch('capture_echo.library_path',return_value=Path('/missing/speakerdesk_echo.dylib')):
            with self.assertRaisesRegex(RuntimeError,'missing from this build'):
                SourceMixer(['microphone','system'],lambda mixed,tracks:self.fail('Audio unexpectedly emitted'))
        # Single-source capture needs no external playback reference.
        self.assertIsNone(SourceMixer(['microphone'],lambda mixed,tracks:None).echo)

    def test_stereo_payload_shape_is_validated_before_alignment(self):
        mixer=SourceMixer(['microphone','system'],lambda mixed,tracks:None,echo_factory=None)
        with self.assertRaises(ValueError):mixer.add('system',0,np.zeros(3,np.float32).tobytes(),2)
        with self.assertRaises(ValueError):mixer.add('microphone',0,np.zeros(4,np.float32).tobytes(),2)
        with self.assertRaises(ValueError):mixer.add('system',0,np.zeros(4,np.float32).tobytes(),True)

    def test_silence_cannot_create_an_echo_or_clock_match(self):
        clock=ReferenceClock()
        for _ in range(12):clock.append(np.zeros((4000,2),np.float32),np.zeros(4000,np.float32))
        self.assertFalse(clock.echo);self.assertEqual(clock.matches,[]);self.assertEqual(clock.skew,0.)


class ReferenceClockTests(unittest.TestCase):
    def test_changing_strongest_speaker_does_not_invent_clock_drift(self):
        rng = np.random.default_rng(946)
        size = RATE*8
        far = rng.normal(0, .03, (size, 2)).astype(np.float32)
        # Change the active speaker every two seconds. Two stationary transducers have
        # different path lengths, but neither sample clock is drifting.
        for start in range(0, size, 4000):
            far[start:start+4000, 1-(start//4000//8)%2] = 0
        microphone = np.zeros(size, np.float32)
        for channel, delay in [(0, 715), (1, 717)]:
            microphone[delay:] += .5*far[:-delay, channel]
        clock = ReferenceClock()
        measured = []
        for start in range(0, size, 4000):
            clock.append(far[start:start+4000], microphone[start:start+4000])
            measured.append(clock.skew)
        self.assertTrue(clock.echo)
        self.assertLess(max(abs(value) for value in measured), 3e-6)
        self.assertEqual({round(match[1]) for match in clock.matches}, {715, 717})

    def test_stereo_path_offsets_preserve_a_real_clock_drift_estimate(self):
        rng = np.random.default_rng(218)
        size = RATE*16
        far = rng.normal(0, .03, (size, 2)).astype(np.float32)
        for start in range(0, size, 4000):
            far[start:start+4000, 1-(start//4000//8)%2] = 0
        index = np.arange(size)
        for drift in (180e-6, -220e-6):
            with self.subTest(drift=drift):
                microphone = sum(.5*np.interp(index*(1+drift)-delay, index, far[:,channel], left=0, right=0)
                                 for channel, delay in [(0, 715), (1, 717)]).astype(np.float32)
                clock = ReferenceClock()
                for start in range(0, size, 4000):
                    clock.append(far[start:start+4000], microphone[start:start+4000])
                self.assertTrue(clock.echo)
                self.assertAlmostEqual(clock.skew, drift, delta=30e-6)


if __name__=='__main__':unittest.main()
