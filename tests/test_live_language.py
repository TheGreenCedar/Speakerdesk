"""Live configuration through real pipes/SQLite and real worker code, CPU only."""
import base64
import contextlib
import io
import json
import queue
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from pcm_peer import varying_pcm, SpeechFramePeer, SpeechEvidencePeer

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from app import create_app
from language_detection import SpeechTranscriber
from live_language import LanguageEpochs, validate_language_segment
from support.meeting_harness import MeetingHarness
from test_language_detection import cohere_model, model_modules, scores


class LiveLanguageAPITests(MeetingHarness,unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.scenario='language_inflight'

    def change(self,jid,language,revision):
        return self.client.patch(f'/api/meetings/{jid}/language',headers=self.headers,
                                 json={'language':language,'language_revision':revision})

    def paused_inflight(self):
        jid=self.start(['microphone'])
        self.wait_for(lambda:self.job(jid)['status']=='recording')
        self.control(jid,'pause')
        self.wait_for(lambda:self.job(jid)['status']=='paused' and self.manager.duration==.1)
        self.wait_for(lambda:(self.root/jid/'worker-inflight').exists())
        return jid

    def test_rapid_changes_preserve_inflight_epoch_and_apply_after_boundary_through_resume_and_restart(self):
        jid=self.paused_inflight()
        for revision,language in enumerate(['fr','auto','fr']):
            response=self.change(jid,language,revision)
            self.assertEqual(response.status_code,200,response.json)
            self.assertEqual(response.json['language_revision'],revision+1)
            self.assertEqual(response.json['language_history'][-1]['start_sample'],1600)
        state=self.client.get('/api/meeting').json
        self.assertEqual((state['language'],state['language_revision'],state['language_acknowledged_revision']),('fr',3,0))
        (self.root/jid/'worker-release').touch()
        self.wait_for(lambda:self.job(jid)['language_acknowledged_revision']==3)
        self.wait_for(lambda:len(self.job(jid)['document']['segments'])==1)
        self.control(jid,'resume')
        self.wait_for(lambda:self.job(jid)['status']=='recording')
        self.control(jid,'pause')
        self.wait_for(lambda:self.job(jid)['status']=='paused' and self.manager.duration==.35)
        self.wait_for(lambda:len(self.job(jid)['document']['segments'])==3)
        self.control(jid,'stop');self.wait_for(lambda:self.manager.jid is None)
        job=self.job(jid)
        segments=[s for s in job['document']['segments'] if s.get('language_generation') is not None]
        self.assertEqual([(s['start'],s['end'],s['language'],s['language_generation']) for s in segments],
                         [(0,.1,'en',0),(.1,.25,'fr',3),(.25,.35,'fr',3)])
        self.assertEqual(segments[0]['text'],'Synthetic en words.')
        commands=[json.loads(line) for line in (self.root/jid/'worker-commands.jsonl').read_text().splitlines()]
        self.assertEqual([c['generation'] for c in commands if c['type']=='language'],[1,2,3])
        self.assertEqual(len(self.samples(jid,'audio.wav')),5600)
        reopened=create_app(self.root)
        try:
            restored=reopened.test_client().get(f'/api/jobs/{jid}').json
            self.assertEqual(restored['language_history'],job['language_history'])
            self.assertEqual(restored['document'],job['document'])
            self.assertEqual(restored['language'],'fr')
        finally:
            reopened.extensions['speakerdesk']['meetings'].close()
            reopened.extensions['speakerdesk']['executor'].shutdown(wait=True,cancel_futures=True)
        # The successful active override is also the new default for the next meeting.
        self.scenario='normal'
        response=self.client.post('/api/meetings',headers=self.headers,json={'sources':['microphone']})
        self.assertEqual(response.status_code,201,response.json)
        self.assertEqual(response.json['language'],'fr')  # Last successful mode is the persisted new default.
        self.assertEqual(response.json['language_revision'],0)
        self.control(response.json['id'],'stop');self.wait_for(lambda:self.manager.jid is None)

    def test_conflicting_invalid_unauthenticated_or_unavailable_changes_leave_current_epoch_unchanged(self):
        jid=self.paused_inflight()
        self.assertEqual(self.client.patch(f'/api/meetings/{jid}/language',json={'language':'fr','language_revision':0}).status_code,403)
        for language,revision in [('unknown',0),([],0),('fr',True),('fr',None),('fr',-1)]:
            self.assertEqual(self.change(jid,language,revision).status_code,400)
        self.assertEqual(self.change(jid,'fr',0).status_code,200)
        self.assertEqual(self.change(jid,'en',0).status_code,409)
        self.assertEqual(self.change(jid,'fr',1).json['language_revision'],1) # no-op
        with patch('live_meeting.preflight',return_value=['Detector unavailable']):
            self.assertEqual(self.change(jid,'auto',1).status_code,409)
        self.assertEqual(self.job(jid)['language_revision'],1)
        self.assertEqual(self.job(jid)['language'],'fr')

    def test_recording_change_uses_saved_frontier_and_preserves_both_selected_sources(self):
        jid=self.start(['microphone','system'])
        self.wait_for(lambda:self.job(jid)['status']=='recording')
        self.assertEqual(self.manager.duration,0) # Alignment tail has not reached the saved watermark.
        response=self.change(jid,'fr',0)
        self.assertEqual(response.status_code,200,response.json)
        self.assertEqual(response.json['language_history'][-1]['start_sample'],0)
        self.control(jid,'pause')
        self.wait_for(lambda:(self.root/jid/'worker-inflight').exists())
        (self.root/jid/'worker-release').touch()
        self.wait_for(lambda:len(self.job(jid)['document']['segments'])==1)
        self.control(jid,'stop');self.wait_for(lambda:self.manager.jid is None)
        job=self.job(jid)
        segment=job['document']['segments'][0]
        self.assertEqual((segment['start'],segment['end'],segment['language'],segment['language_generation']),(0,.1,'fr',1))
        self.assertEqual(job['sources'],['microphone','system'])
        for source,value in [('microphone',.2),('system',.3),('audio',.5)]:
            np.testing.assert_array_equal(self.samples(jid,f'{source}.wav'),
                                          np.rint(np.full(1600,value,dtype=np.float32).astype(np.float64)*32768).astype('<i2'))

    def test_failed_persistence_and_full_queue_never_acknowledge_or_enqueue_a_change(self):
        jid=self.paused_inflight()
        original=self.job(jid)
        with patch.object(self.manager,'put',side_effect=OSError('Synthetic write failure')),self.assertLogs(self.app.logger,level='ERROR'):
            self.assertEqual(self.change(jid,'fr',0).status_code,500)
        self.assertEqual(self.job(jid),original)
        with self.manager.lock:
            original_queue=self.manager.packets
            blocked=queue.Queue(maxsize=1);blocked.put({'type':'audio'})
            self.manager.packets=blocked
            try:self.assertEqual(self.change(jid,'fr',0).status_code,429)
            finally:self.manager.packets=original_queue
        self.assertEqual(self.job(jid),original)
        self.assertEqual(self.change(jid,'fr',0).status_code,200)

    def test_wrong_manual_language_result_is_rejected_and_audio_and_earlier_words_survive(self):
        self.scenario='language_wrong_epoch'
        jid=self.paused_inflight()
        self.assertEqual(self.change(jid,'fr',0).status_code,200)
        (self.root/jid/'worker-release').touch()
        self.wait_for(lambda:self.job(jid)['language_acknowledged_revision']==1)
        self.control(jid,'resume');self.wait_for(lambda:self.job(jid)['status']=='recording')
        self.control(jid,'pause');self.wait_for(lambda:self.manager.jid is None)
        job=self.job(jid)
        self.assertEqual(job['status'],'failed')
        inferred=[s for s in job['document']['segments'] if 'language_generation' in s]
        self.assertEqual([s['text'] for s in inferred],['Synthetic en words.'])
        self.assertGreaterEqual(len(self.samples(jid,'audio.wav')),5600)
        self.assertEqual(self.change(jid,'auto',1).status_code,409)


class LiveLanguageWorkerTests(unittest.TestCase):
    def run_worker(self,messages,detector_scores=None):
        import live_worker
        class Diarizer:
            def set_streaming_config(self,preset):pass
            def init_streaming_state(self):return types.SimpleNamespace(frames_processed=0)
            def feed(self,pcm,state,**kwargs):
                start=state.frames_processed*.01;state.frames_processed+=len(pcm)//160
                return types.SimpleNamespace(segments=[types.SimpleNamespace(start=start,end=state.frames_processed*.01,speaker=0)] if len(pcm) else []),state
        output=io.StringIO();asr=cohere_model();detector=Mock()
        detector.detect.side_effect=detector_scores or []
        with patch.dict(sys.modules,model_modules(asr,Diarizer())),patch('inference_worker.check_memory'), \
             patch('speech_admission.SileroModel',side_effect=lambda path:SpeechFramePeer()), \
             patch('language_detection.WhisperLanguageDetector',return_value=detector), \
             patch('sys.stdin',io.StringIO('\n'.join(json.dumps(m) for m in messages)+'\n')),contextlib.redirect_stdout(output):
            from support.resident_fixture import run_messages
            events=run_messages({'language':'en','diar_path':'unused','cohere_path':'unused','lid_path':'approved-local'},messages)
        return events,asr

    def audio(self,value=.1,count=16000):
        return {'type':'audio','pcm':base64.b64encode(varying_pcm(count,value,dtype='<f4').tobytes()).decode()}

    def test_uncommitted_phrase_is_split_at_control_boundary_and_never_relabelled(self):
        messages=[self.audio() for _ in range(2)]
        messages+=[{'type':'language','generation':1,'start_sample':32000,'language':'fr'}]
        messages+=[self.audio(.2) for _ in range(4)]+[{'type':'stop'}]
        events,asr=self.run_worker(messages)
        segments=[e['segment'] for e in events if e['type']=='segment']
        self.assertEqual([(s['start'],s['end'],s['language_generation'],s['language']) for s in segments],[(0,2,0,'en'),(2,6,1,'fr')])
        self.assertEqual([c.kwargs['language'] for c in asr.transcribe.call_args_list],['en','fr'])
        np.testing.assert_allclose(asr.transcribe.call_args_list[0].args[0],varying_pcm(32000,.1,dtype=np.float32),atol=1/32767)
        np.testing.assert_allclose(asr.transcribe.call_args_list[1].args[0],varying_pcm(64000,.2,dtype=np.float32),atol=1/32767)

    def test_rapid_zero_length_epochs_do_not_duplicate_audio_or_generate_wrong_language(self):
        messages=[self.audio()]
        messages+=[{'type':'language','generation':i,'start_sample':16000,'language':language}
                   for i,language in enumerate(['fr','auto','fr'],1)]
        messages+=[self.audio()]+[{'type':'stop'}]
        events,asr=self.run_worker(messages)
        segments=[e['segment'] for e in events if e['type']=='segment']
        self.assertEqual([(s['start'],s['end'],s['language_generation'],s['language']) for s in segments],[(0,1,0,'en'),(1,2,3,'fr')])
        self.assertEqual(asr.transcribe.call_count,2)

    def test_new_auto_epoch_clears_hysteresis_and_reuses_detector_after_manual_override(self):
        detector=Mock();detector.detect.side_effect=[scores('en'),scores('fr',.92)]
        with patch('language_detection.WhisperLanguageDetector',return_value=detector) as load:
            transcriber=SpeechTranscriber(cohere_model(),'auto','local-approved',speech_evidence=SpeechEvidencePeer())
            pcm=varying_pcm(16000,.1,dtype=np.float32)
            self.assertEqual(transcriber.transcribe(pcm,16000,('speaker_0',))[0]['language'],'en')
            transcriber.set_language('fr')
            self.assertEqual(transcriber.transcribe(pcm,16000,('speaker_0',))[0]['language'],'fr')
            transcriber.set_language('auto')
            self.assertEqual(transcriber.transcribe(pcm,16000,('speaker_0',))[0]['language'],'fr')
            self.assertEqual(load.call_count,1)

    def test_epoch_protocol_rejects_out_of_order_generation_wrong_cursor_and_cross_boundary_output(self):
        epochs=LanguageEpochs('en')
        for generation,boundary in [(2,16000),(1,0),(True,16000)]:
            with self.assertRaises(ValueError):epochs.register({'generation':generation,'start_sample':boundary,'language':'fr'},16000)
        epochs.register({'generation':1,'start_sample':16000,'language':'fr'},16000)
        for segment in [{'start':0,'end':1.1,'language_mode':'en','language':'en','language_generation':0},
                        {'start':1,'end':2,'language_mode':'fr','language':'en','language_generation':1},
                        {'start':1,'end':2,'language_mode':'fr','language':'fr','language_generation':2}]:
            with self.assertRaises(ValueError):validate_language_segment(segment,epochs.history)


if __name__=='__main__':unittest.main()
