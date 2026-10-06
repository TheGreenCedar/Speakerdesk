"""Production engine/host contracts with explicit synthetic CPU model outputs."""
import copy
import re
import sys
import tempfile
import threading
import unittest
import wave
from unittest.mock import Mock,patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from app import create_app
from live_refinement import Engine, Inbox, Models, align_tracks, context_regions, read_audio
from meeting_refinement import RefinementController, split_blank_placeholders
from rolling_refinement import RATE, RollingPlan, reconcile_window, segment_version, split_same_origin


class CPUModels:
    def __init__(self,config=None):self.received=0;self.calls=[];self.batch_calls=[]
    def feed(self,audio,final=False):
        start=self.received/RATE;self.received+=len(audio);end=self.received/RATE
        return ([{'start':start,'end':end,'speaker':'speaker_0'}] if len(audio) else []),end
    def transcribe(self,audio,language,names,overlap=False):
        seconds=len(audio)/RATE;self.calls.append((seconds,language,overlap))
        return [{'start':0,'end':seconds,'text':' '.join(['token']*max(1,round(seconds))),
                 'language':language if language!='auto' else 'en','review':len(names)>1}]
    def batch_turns(self,audio):
        self.batch_calls.append(len(audio)/RATE)
        return [{'start':0,'end':len(audio)/RATE,'speaker':'speaker_7'}]
    def metrics(self):return {'peak_mlx_bytes':0,'peak_process_rss_bytes':0}


def wav_file(path,seconds=80):
    with wave.open(str(path),'wb') as wav:
        wav.setparams((1,2,RATE,0,'NONE','not compressed'));wav.writeframes(b'\x01\x00'*round(seconds*RATE))


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)/'audio.wav';wav_file(self.path)
        self.models=CPUModels();self.events=[];self.inbox=Inbox()
        self.engine=Engine({'audio_path':str(self.path),'language':'en'},self.models,self.events.append,self.inbox)
    def tearDown(self):self.temp.cleanup()
    def feed(self,start,end,mode='en',epoch=0):
        for n in range(start,end):self.engine.handle({'type':'audio','start_sample':n*RATE,'end_sample':(n+1)*RATE,'language':mode,'language_epoch':epoch})
    def test_six_second_midword_is_replaced_at_same_audio_origin(self):
        def transcribe(audio,language,names,overlap=False):
            duration=len(audio)/RATE
            return [{'start':0,'end':duration,'text':'Please review the secs' if duration==6 else 'Please review the section together.','language':'en'}]
        self.models.transcribe=transcribe;self.feed(0,6)
        first=self.engine.document['segments'][0]['id'];self.feed(6,9)
        rows=self.engine.document['segments'];self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['id'],first);self.assertEqual(rows[0]['text'],'Please review the section together.')
        self.assertEqual(rows[0]['refinement_state'],'provisional');self.assertFalse(rows[0].get('review',False))
    def test_long_continuous_speech_carries_context_preserves_real_repeats_and_hard_caps(self):
        self.feed(0,80);self.engine.handle({'type':'stop'})
        full={'speakers':{},'segments':[]}
        for event in self.events:
            if event['type']=='provisional_revision':full=reconcile_window(full,event['window'],event['expected'],event['candidates'],event['speakers'])['document']
        rows=full['segments'];self.assertEqual(sum(len(s['text'].split()) for s in rows),80)
        self.assertEqual([(s['start'],s['end']) for s in rows],[(0,18),(18,36),(36,54),(54,72),(72,80)])
        self.assertTrue(all(s['finalized'] for s in rows));self.assertLessEqual(max(s[0] for s in self.models.calls),24.5)
        self.assertLessEqual(len(self.engine.document['segments']),4)
    def test_stop_flushes_multiple_remaining_cores_and_short_right_context(self):
        self.engine.received=20*RATE;self.engine.processed=20
        self.engine.turns=[{'start':0,'end':20,'speaker':'speaker_0'}]
        self.engine.commit(final=True)
        self.assertEqual(self.engine.cursor,20)
        self.assertEqual(sum(len(s['text'].split()) for s in self.engine.document['segments']),20)
    def test_language_boundary_and_pause_flush_never_relabel_earlier_audio(self):
        self.feed(0,2);self.engine.handle({'type':'language','generation':1,'language':'fr','start_sample':2*RATE})
        self.feed(2,6,'fr',1);self.engine.handle({'type':'flush'})
        rows=self.engine.document['segments'];self.assertEqual([(s['start'],s['end'],s['language_generation'],s['language']) for s in rows],[(0,2,0,'en'),(2,6,1,'fr')])
        self.feed(6,10,'fr',1);self.engine.handle({'type':'stop'})
        self.assertEqual(self.engine.cursor,10);self.assertEqual(self.events[-1]['type'],'capture_finished')
    def test_ambiguous_context_keeps_legible_words_and_full_candidate(self):
        counter=[0]
        def transcribe(audio,language,names,overlap=False):
            duration=len(audio)/RATE;counter[0]+=1
            text='stable previous words' if duration<=18 else 'different contextual hypothesis'
            return [{'start':0,'end':duration,'text':text,'language':'en'}]
        self.models.transcribe=transcribe;self.feed(0,21)
        self.assertTrue(any(e['type']=='boundary_candidate' for e in self.events))
        self.assertIn('stable previous words',[s['text'] for s in self.engine.document['segments']])
        self.assertTrue(any(s.get('boundary_unresolved') for s in self.engine.document['segments']))
    def test_inbox_coalesces_large_audio_backlog_and_cancels_stale_language_operations(self):
        for n in range(10000):self.inbox.push({'type':'audio','start_sample':n*RATE,'end_sample':(n+1)*RATE,'language':'en','language_epoch':0})
        self.assertEqual(len(self.inbox.messages),1);self.assertEqual(self.inbox.take()['end_sample'],RATE)
        request={'operation_id':'old','language_epoch':0}
        self.inbox.push({'type':'language','generation':1,'start_sample':10000*RATE,'language':'fr'})
        self.assertTrue(self.inbox.cancelled_request(request))
        self.inbox.push({'type':'shutdown'});self.assertEqual(self.inbox.take()['type'],'shutdown')
    def test_batch_slots_map_by_time_and_ambiguous_slots_stay_unknown(self):
        refs=[{'start':0,'end':5,'speaker_candidates':['speaker_0']},{'start':5,'end':10,'speaker_candidates':['speaker_1']}]
        turns=[{'start':0,'end':5,'speaker':'speaker_7'},{'start':5,'end':10,'speaker':'speaker_2'}]
        self.assertEqual(align_tracks(turns,refs),{'speaker_7':'speaker_0','speaker_2':'speaker_1'})
        mixed=[{'start':0,'end':10,'speaker_candidates':['speaker_0','speaker_1']}]
        self.assertEqual(align_tracks(turns,mixed),{})
    def test_micro_overlap_is_one_shared_phrase_without_erasing_voice_candidates(self):
        turns=[{'start':0,'end':10,'speaker':'speaker_0'},{'start':4,'end':4.1,'speaker':'speaker_1'}]
        self.assertEqual(context_regions(turns),[{'start':0,'end':10,'speakers':['speaker_0','speaker_1']}])
        self.engine.received=10*RATE;self.engine.processed=10;self.engine.turns=turns;self.engine.commit(final=True)
        rows=self.engine.document['segments'];self.assertEqual(len(rows),1)
        self.assertFalse(rows[0]['voice_eligible']);self.assertTrue(rows[0]['review'])
        self.assertEqual(rows[0]['speaker_candidates'],['speaker_0','speaker_1'])
    def test_lexical_continuity_keeps_repeats_sentence_punctuation_and_code_switch(self):
        for old,new,want in [('yes yes','yes yes yes',('yes yes','yes')),('Done. Next secs','Done. Next section now.',('Done. Next section','now.')),('Done. Next secs.','Done. Next section now.',('Done. Next section','now.')),('Done.','Done. Next sentence.',('Done.','Next sentence.')),('bonjour hello','bonjour hello encore',('bonjour hello','encore'))]:
            self.assertEqual(split_same_origin(old,new),want)
        self.assertIsNone(split_same_origin('repeat the sentence','a different sentence repeat the sentence'))
        with self.assertRaises(ValueError):read_audio(self.path,0,25*RATE)
    def test_offline_diarization_restores_live_preset_even_when_model_fails(self):
        models=Models.__new__(Models);models.check_memory=Mock();models.diar=Mock()
        models.diar.generate.side_effect=RuntimeError('CPU injected failure')
        with self.assertRaises(RuntimeError):models.batch_turns([])
        self.assertEqual([call.args[0] for call in models.diar.set_streaming_config.call_args_list],['offline','low'])


class HostTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.app=create_app(self.root)
        self.manager=self.app.extensions['speakerdesk']['meetings'];self.manager.duration=self.manager.processed=30
        self.client=self.app.test_client();token=re.search(r'name="speakerdesk-token" content="([^"]+)"',self.client.get('/').get_data(as_text=True)).group(1)
        self.headers={'X-Speakerdesk-Token':token};self.jid='c'*32
        self.job={'id':self.jid,'name':'CPU contract','kind':'meeting','language':'en','duration':30,'revision':0,'status':'ready',
                  'document':{'speakers':{'speaker_0':'Confirmed Albert'},'segments':[], 'provenance':{'kind':'local_inference'}}}
        self.manager.refinement.initialize(self.job);self.manager.put(self.job)
    def tearDown(self):
        self.manager.close();self.app.extensions['speakerdesk']['executor'].shutdown(wait=True,cancel_futures=True);self.temp.cleanup()
    def provisional(self,end,text,seq=1):
        job=self.manager.get(self.jid)
        result={'window':{'id':'fast','kind':'fast_tail','start_sample':0,'end_sample':end*RATE},'fast_sequence':seq,'language_epoch':0,
                'expected':{s['id']:segment_version(s) for s in job['document']['segments']},'speakers':{'speaker_0':'Guessed name'},
                'candidates':[{'start':0,'end':end,'text':text,'speaker':'speaker_0','language':'en','language_epoch':0,
                               'language_generation':0,'language_mode':'en','fast_origin_sample':0,'refinement_state':'provisional'}]}
        self.manager.refinement.provisional(self.jid,result);return result
    def test_saved_prefix_does_not_block_new_tail_or_get_overwritten(self):
        self.provisional(6,'Please review the secs');row=self.manager.get(self.jid)['document']['segments'][0]
        response=self.client.patch(f'/api/jobs/{self.jid}/segments/{row["id"]}',headers=self.headers,
                                   json={'segment_revision':row['machine_revision'],'changes':{'text':'My saved correction'}})
        self.assertEqual(response.status_code,200,response.json)
        result=self.provisional(9,'Please review the section together.',2)
        self.provisional(12,'Please review the section together. Yes yes.',3)
        rows=self.manager.get(self.jid)['document']['segments']
        self.assertEqual([s['text'] for s in rows],['My saved correction','together. Yes yes.'])
        self.assertEqual([(s['start'],s['end']) for s in rows],[(0,6),(6,12)])
        before=copy.deepcopy(rows);self.manager.refinement.provisional(self.jid,result)
        self.assertEqual(self.manager.get(self.jid)['document']['segments'],before)
        self.assertEqual(self.manager.get(self.jid)['document']['speakers']['speaker_0'],'Confirmed Albert')
        history=self.client.get(f'/api/jobs/{self.jid}/refinement/fast-revisions/0').json
        self.assertTrue(history['versions']);self.assertEqual(history['versions'][0]['segments'][0]['text'],'My saved correction')
        self.assertEqual(history['versions'][0]['candidates'][0]['text'],'Please review the section together.')
    def test_targeted_edit_cas_csrf_and_whole_save_cannot_clear_protection(self):
        self.provisional(6,'First words');job=self.manager.get(self.jid);row=job['document']['segments'][0];url=f'/api/jobs/{self.jid}/segments/{row["id"]}'
        body={'segment_revision':row['machine_revision'],'changes':{'text':'saved edit'}}
        self.assertEqual(self.client.patch(url,json=body).status_code,403)
        self.assertEqual(self.client.patch(url,json=body,headers=self.headers).status_code,200)
        self.assertEqual(self.client.patch(url,json=body,headers=self.headers).status_code,409)
        job=self.manager.get(self.jid);incoming=copy.deepcopy(job['document']);incoming['segments'][0].update(protected_fields=[],machine_revision=900,source_start=99,language_generation=99)
        saved=self.client.put(f'/api/jobs/{self.jid}/transcript',headers=self.headers,json={'revision':job['revision'],'document':incoming})
        self.assertEqual(saved.status_code,200,saved.json);self.assertEqual(saved.json['document']['segments'][0]['protected_fields'],['text'])
        self.assertEqual(saved.json['document']['segments'][0]['language_generation'],0)
    def test_long_blank_recovery_can_fill_disjoint_windows_without_erasing_words(self):
        job=self.manager.get(self.jid);job['document']['segments']=[{'id':'blank','start':0,'end':30,'text':'','speaker':'speaker_0'}];self.manager.put(job)
        self.manager.refinement.enabled=True;self.manager.refinement.force=True;self.manager.refinement.schedule(self.jid)
        request=self.manager.worker_controls.get_nowait();self.assertEqual(request['window']['end_sample'],18*RATE)
        self.manager.refinement.result(self.jid,{'operation_id':request['operation_id'],'language_epoch':0,'window':request['window'],
            'candidates':[{'start':0,'end':18,'text':'recovered words','speaker':'speaker_0','language_generation':0,'language_mode':'en','language':'en'}]})
        job=self.manager.get(self.jid);self.assertEqual([s['text'] for s in job['document']['segments']],['recovered words',''])
        self.assertEqual(job['rolling_refinement']['completed_sample'],18*RATE)
        self.assertIn(request['window']['id'],job['refinement_history'])
    def test_cancelled_late_result_and_new_epoch_cannot_replace_current_words(self):
        self.provisional(6,'Retain words');self.manager.refinement.enabled=True;self.manager.refinement.force=True;self.manager.refinement.schedule(self.jid)
        request=self.manager.worker_controls.get_nowait();self.manager.refinement.pause(self.jid)
        before=self.manager.get(self.jid)['document'];self.manager.refinement.result(self.jid,{**request,'candidates':[]})
        self.assertEqual(self.manager.get(self.jid)['document'],before)
        self.assertEqual(self.manager.get(self.jid)['refinement_status'],'paused')
    def test_startup_keeps_audio_history_and_pauses_pending_work_without_capture(self):
        self.provisional(6,'Retain words');self.manager.refinement.enabled=True;self.manager.refinement.force=True;self.manager.refinement.schedule(self.jid)
        (self.root/self.jid).mkdir();wav_file(self.root/self.jid/'audio.wav',30);before=(self.root/self.jid/'audio.wav').read_bytes()
        reopened=create_app(self.root)
        try:
            restored=reopened.extensions['speakerdesk']['meetings'].get(self.jid)
            self.assertEqual(restored['refinement_status'],'paused');self.assertEqual(restored['rolling_refinement']['pending'][0]['status'],'queued')
            self.assertEqual(restored['document']['segments'][0]['text'],'Retain words')
            self.assertIsNone(reopened.extensions['speakerdesk']['meetings'].capture)
            self.assertEqual((self.root/self.jid/'audio.wav').read_bytes(),before)
        finally:reopened.extensions['speakerdesk']['executor'].shutdown(wait=True,cancel_futures=True)
    def test_two_failed_attempts_retain_words_and_bookmark_explicit_retry(self):
        self.provisional(6,'Retain words');self.manager.refinement.enabled=True;self.manager.refinement.force=True
        self.manager.refinement.schedule(self.jid)
        for _ in range(2):
            request=self.manager.worker_controls.get_nowait()
            self.manager.refinement.result(self.jid,{'operation_id':request['operation_id'],'language_epoch':0,'window':request['window'],'error':'CPU injected failure'})
        job=self.manager.get(self.jid);self.assertEqual(job['document']['segments'][0]['text'],'Retain words')
        self.assertTrue(job['refinement_unresolved']);self.assertEqual(job['document']['segments'][0]['refinement_state'],'unresolved')
        self.manager.refinement.pause(self.jid);self.manager.refinement.resume(self.jid)
        job=self.manager.get(self.jid);self.assertEqual(job['rolling_refinement']['completed_sample'],0)
    def test_initial_three_section_delay_is_measured_from_meeting_creation(self):
        job=self.manager.get(self.jid);job.pop('rolling_refinement');job['duration']=11
        job['rolling_sources']={str(n):{'id':str(n),'start':start,'end':end,'text':'CPU words','speaker':'speaker_0','finalized':True}
                                for n,(start,end) in enumerate([(0,3),(3,6),(6,8)])}
        with patch('meeting_refinement.time.monotonic',return_value=100):self.manager.refinement.initialize(job)
        self.manager.put(job);self.manager.processed=11;self.manager.refinement.enabled=True
        with patch('meeting_refinement.time.monotonic',return_value=107):self.manager.refinement.schedule(self.jid)
        self.assertTrue(self.manager.worker_controls.empty())
        with patch('meeting_refinement.time.monotonic',return_value=108):self.manager.refinement.schedule(self.jid)
        self.assertEqual(self.manager.worker_controls.get_nowait()['window']['end_sample'],8*RATE)


if __name__=='__main__':unittest.main()
