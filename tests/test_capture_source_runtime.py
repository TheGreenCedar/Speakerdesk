"""Source/clock/CAS/voice regressions with explicit CPU peers, not acoustic proof."""
import copy
import json
import sys
import tempfile
import unittest
import uuid
import wave
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'speakerdesk'),str(Path(__file__).parent)]
from test_canonical_runtime import Peer
from app import create_app
from capture_sources import RATE,SOURCE_IDS,catalog,binding,utterance_id,validate_inspections
from admission_receipt import execution,retained_pcm_digest
from source_runtime import SourceRuntime
from live_refinement import Inbox,Models
from meeting_refinement import activity_references
from voice_profiles import VoiceClip,ClipEmbedding
from source_voice import voice_clip_choices,clips_current,extract
from voice_source_audio import validate_voice_audio,clip_pcm_digest


def wav(path,seconds,value):
    with wave.open(str(path),'wb') as recording:
        recording.setparams((1,2,RATE,0,'NONE','none'))
        recording.writeframes(value.to_bytes(2,'little',signed=True)*round(seconds*RATE))


class SourcePeer(Peer):
    def __init__(self,config):
        super().__init__();self.config=config;self.admission_execution=execution(config['job_id'],uuid.uuid4().hex)
        self.text='  go go café 👩🏽‍💻  ';self.inputs=[];self.closed=False
    def feed(self,audio,final=False):
        self.inputs.extend(audio.tolist());self.closed=final
        return super().feed(audio,final=final)
    def inspection_receipt(self,phase,request_id):
        endpoint=self.speech_live.evidence.end_sample
        return {**self.admission_execution,'inspection_state':'observed_prefix','phase':phase,
            'request_id':request_id,'received_sample':self.received,'start_sample':0,'end_sample':endpoint,
            'speech_samples':endpoint,'uncertain_samples':0,'negative_constant_samples':0,
            'model_negative_samples':0,'closed':self.closed,'audio_encoding':'pcm_s16le',
            'pcm_sha256':retained_pcm_digest(self.config['audio_path'],endpoint),'decision':'speech'}


class SourceRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.jid='a'*32;self.folder=self.root/self.jid;self.folder.mkdir()
        for source,value in [('audio',333),('microphone_clean',111),('system',222)]:
            wav(self.folder/(source+'.wav'),12,value)
        self.catalog=catalog(self.jid)
        (self.folder/'capture-sources.json').write_text(json.dumps(self.catalog))
        self.peers={};self.events=[];self.inbox=Inbox()
        owner=type('Owner',(),{})()
        def make(config):
            peer=SourcePeer(config);self.peers[config['capture_source']['source_id']]=peer;return peer
        owner.for_source=make
        self.config={'audio_path':str(self.folder/'audio.wav'),'language':'en','job_id':self.jid,
                     'canonical_utterances':True,'capture_source_catalog':self.catalog}
        self.runtime=SourceRuntime(self.config,owner,self.events.append,self.inbox)
    def feed(self,end=7):
        self.runtime.handle({'type':'audio','start_sample':self.runtime.received,'end_sample':end*RATE,
                             'language':'en','language_epoch':0})
    def rows(self):
        return {event['candidate']['id']:event['candidate'] for event in self.events
                if event['type']=='canonical_revision'}
    def job(self,rows=None):
        return {'id':self.jid,'name':'CPU source contracts','status':'ready','kind':'meeting',
                'created':1,'language':'en','revision':0,'duration':7,'canonical_utterances':True,
                'capture_source_catalog':self.catalog,
                'document':{'speakers':{'speaker_0':'Speaker 1','speaker_32':'Speaker 33'},
                            'segments':list((rows or self.rows()).values()),'warnings':[],
                            'provenance':{'kind':'local_inference'}}}
    def test_same_time_identical_words_survive_as_two_original_source_rows(self):
        self.feed();self.runtime.handle({'type':'stop','request_id':'stop'})
        rows=list(self.rows().values());self.assertEqual(len(rows),2)
        self.assertEqual([r['text'] for r in rows],['  go go café 👩🏽‍💻  ']*2)
        self.assertEqual({r['speaker'] for r in rows},{'speaker_0','speaker_32'})
        self.assertEqual({(r['start_sample'],r['end_sample']) for r in rows},{(0,7*RATE)})
        self.assertEqual({r['id'] for r in rows},{utterance_id(self.jid,0,0,binding(self.catalog,s)) for s in SOURCE_IDS})
        self.assertNotIn(utterance_id(self.jid,0,0),{r['id'] for r in rows})
        for source,value in [('microphone_clean',111),('system',222)]:
            self.assertEqual(set(self.peers[source].inputs),{value/32768})
        seq=[e['fast_sequence'] for e in self.events if e['type']=='canonical_revision']
        self.assertEqual(seq,list(range(1,len(seq)+1)))

    def test_development_catalog_binds_selected_input_for_replay_and_voice(self):
        from capture_sources import validate_catalog
        self.catalog=catalog(self.jid,innovation=True)
        (self.folder/'capture-sources.json').write_text(json.dumps(self.catalog))
        wav(self.folder/'microphone_native.wav',12,999)
        owner=type('Owner',(),{})()
        def make(config):
            peer=SourcePeer(config);self.peers[config['capture_source']['source_id']]=peer;return peer
        owner.for_source=make
        self.config['capture_source_catalog']=self.catalog
        self.runtime=SourceRuntime(self.config,owner,self.events.append,self.inbox)
        self.feed();self.runtime.handle({'type':'stop','request_id':'stop'})
        self.assertEqual(set(self.peers['microphone_clean'].inputs),{111/32768})
        clip=next(iter(voice_clip_choices(self.job(),'speaker_0').values()))
        selected=self.folder/'microphone_clean.wav'
        self.assertEqual(validate_voice_audio(self.root,selected,clip),selected.resolve())
        with self.assertRaises(ValueError):validate_voice_audio(self.root,self.folder/'microphone_native.wav',clip)
        self.assertNotEqual(binding(self.catalog,'microphone_clean'),binding(catalog(self.jid),'microphone_clean'))
        self.runtime.restore_saved()
        self.assertEqual(self.runtime.received,12*RATE)
        self.assertEqual(set(self.peers['microphone_clean'].inputs),{111/32768})
        altered=copy.deepcopy(self.catalog);altered['derivation']['render']='elsewhere.wav'
        with self.assertRaises(ValueError):validate_catalog(altered,self.jid)

    def test_real_source_context_factory_shares_weights_without_crossing_requests(self):
        from types import SimpleNamespace
        from final_asr_reuse import FinalAsrReuse
        owner=object.__new__(Models)  # Never calls the neural-loading constructor.
        owner.diar=SimpleNamespace(init_streaming_state=lambda: {'frames':0})
        owner.asr=object();owner.speech=SimpleNamespace(session=lambda **kw:SimpleNamespace(history=[]))
        owner.final_asr_reuse=FinalAsrReuse({},mode='observe');owner.detector=None;owner.coarse_aligner=None
        a=owner.for_source({**self.config,'audio_path':str(self.folder/'microphone_clean.wav')})
        b=owner.for_source({**self.config,'audio_path':str(self.folder/'system.wav')})
        self.assertIs(a.asr,b.asr);self.assertIs(a.diar,b.diar);self.assertIs(a.speech,b.speech)
        a.state['frames']=100;a.speech_live.history.append('source-A')
        a.final_asr_reuse.begin({'binding':{'id':'source-A'}});a.set_language_context({'language':'en'},100)
        self.assertEqual(b.state['frames'],0);self.assertEqual(b.speech_live.history,[])
        self.assertIsNone(b.final_asr_reuse.scope);self.assertIsNone(b.language_context)
        self.assertNotEqual(a.admission_execution['execution_id'],b.admission_execution['execution_id'])

    def test_bounded_startup_publication_and_saved_restore_consume_identical_selected_sources(self):
        import numpy as np
        from render_startup import StartupSourceSink,StartupReceiptWriter
        from audio import pcm16_bytes
        self.catalog=catalog(self.jid,innovation=True,startup=True)
        (self.folder/'capture-sources.json').write_text(json.dumps(self.catalog))
        self.config['capture_source_catalog']=self.catalog
        owner=type('Owner',(),{})()
        def make(config):
            peer=SourcePeer(config);self.peers[config['capture_source']['source_id']]=peer;return peer
        owner.for_source=make
        self.runtime=SourceRuntime(self.config,owner,self.events.append,self.inbox)
        handles={};recordings={};writer=StartupReceiptWriter(self.folder)
        for name in ('audio','system','microphone_clean'):
            handles[name]=(self.folder/(name+'.wav')).open('w+b')
            recordings[name]=wave.open(handles[name],'wb');recordings[name].setparams((1,2,RATE,0,'NONE','none'))
        far=np.random.default_rng(17).normal(0,.1,64000).astype(np.float32)
        raw=np.zeros_like(far);raw[1232:]=.18*far[:-1232];published=[]
        def publish(mix,tracks):
            start=writer.end;writer.write(start,tracks)
            for name,values in (('audio',mix),('system',tracks['system']),('microphone_clean',tracks['microphone_clean'])):
                recordings[name].writeframes(pcm16_bytes(values));handles[name].flush()
            published.append((start,writer.end))
            self.runtime.handle({'type':'audio','start_sample':start,'end_sample':writer.end,'language':'en','language_epoch':0})
        try:
            with patch('render_startup.threading.Timer'):
                sink=StartupSourceSink(publish)
                for a in range(0,len(raw),4000):
                    b=a+4000
                    sink.retain(dict(microphone=raw[a:b],system_reference=np.repeat(far[a:b,None],2,axis=1),
                        microphone_present=np.ones(4000,bool),system_present=np.ones(4000,bool)),a,0)
                    sink.emit(raw[a:b]+far[a:b],dict(microphone=raw[a:b],system=far[a:b],microphone_clean=raw[a:b]))
                sink.flush()
            self.runtime.handle({'type':'stop','request_id':'startup-stop'})
            live={s:list(p.inputs) for s,p in self.peers.items()}
        finally:
            for w in recordings.values():w.close()
            for h in handles.values():h.close()
            writer.close()
        self.assertEqual(published[0],(0,14000));self.assertEqual(self.runtime.received,len(raw))
        self.runtime.restore_saved()
        self.assertEqual({s:list(p.inputs) for s,p in self.peers.items()},live)
        self.assertEqual(len(live['microphone_clean']),len(raw))
        self.assertTrue(all(v==0 for v in live['microphone_clean'][1232:]))
    def test_missing_or_short_source_fails_before_advancing_either_stream(self):
        wav(self.folder/'system.wav',1,222)
        with self.assertRaisesRegex(ValueError,'shared endpoint'):self.feed()
        self.assertEqual([peer.received for peer in self.peers.values()],[0,0])
    def test_pause_stop_use_both_exact_source_receipts_not_mixed_pcm(self):
        self.feed();self.runtime.handle({'type':'flush','request_id':'pause','through_sample':7*RATE})
        result=self.events[-1];job=self.job();job['source_admission_executions']=self.runtime.admission_executions
        actual=validate_inspections(job,self.folder,result['source_admission_receipts'],phase='pause',
            request_id='pause',received_sample=7*RATE,observed_sample=result['speech_observed_sample'])
        self.assertEqual(set(actual),set(SOURCE_IDS))
        self.runtime.handle({'type':'stop','request_id':'stop'});result=self.events[-1]
        validate_inspections(job,self.folder,result['source_admission_receipts'],phase='stop',
            request_id='stop',received_sample=7*RATE,observed_sample=result['canonical_observed_sample'],uncertain_samples=0)
        wrong=copy.deepcopy(result['source_admission_receipts']);wrong['system']=wrong['microphone_clean']
        with self.assertRaises(ValueError):validate_inspections(job,self.folder,wrong,phase='stop',request_id='stop',
            received_sample=7*RATE,observed_sample=7*RATE)
    def test_language_epochs_and_saved_refinement_keep_source_and_clock(self):
        self.feed(2);self.runtime.handle({'type':'language','generation':1,'language':'fr','start_sample':2*RATE})
        self.inbox.latest_epoch=1
        self.runtime.handle({'type':'audio','start_sample':2*RATE,'end_sample':7*RATE,'language':'fr','language_epoch':1})
        self.runtime.handle({'type':'stop','request_id':'stop'})
        self.assertEqual({(r['language_epoch'],r['start_sample'],r['end_sample']) for r in self.rows().values()},
                         {(0,0,2*RATE),(1,2*RATE,7*RATE)})
        row=next(r for r in self.rows().values() if r['language_epoch']==0)
        req={'type':'refine','operation_id':'refine','canonical':row,'capture_source':row['capture_source'],
             'language_epoch':0,'language':'en','window':{'id':row['id'],'start_sample':0,'end_sample':2*RATE},
             'references':activity_references(self.rows().values(),0,2*RATE,row['capture_source'])}
        wrong=copy.deepcopy(req);wrong['capture_source']=binding(self.catalog,'system' if row['capture_source']['source_id']=='microphone_clean' else 'microphone_clean')
        with self.assertRaisesRegex(ValueError,'dispatch'):self.runtime.handle(wrong)
        self.runtime.handle(req);result=self.events[-1]
        self.assertEqual(result['canonical_candidate']['id'],row['id'])
        self.assertEqual(result['canonical_candidate']['capture_source'],row['capture_source'])
        self.runtime.restore_saved();self.assertEqual(self.runtime.received,12*RATE)
        wav(self.folder/'system.wav',11,222)
        with self.assertRaisesRegex(ValueError,'shared clock'):self.runtime.restore_saved()
    def test_host_merge_and_cas_reject_cross_source_revisions(self):
        self.feed();self.runtime.handle({'type':'stop','request_id':'stop'})
        app=create_app(self.root/'app');manager=app.extensions['speakerdesk']['meetings'];manager.duration=7
        self.addCleanup(manager.close);self.addCleanup(app.extensions['speakerdesk']['executor'].shutdown,wait=True,cancel_futures=True)
        job=self.job();job['document']['segments']=[];manager.refinement.initialize(job);manager.put(job)
        for event in self.events:
            if event['type']=='canonical_revision':manager.refinement.canonical(self.jid,event)
        saved=manager.get(self.jid);self.assertEqual(len(saved['document']['segments']),2)
        row=copy.deepcopy(saved['document']['segments'][0]);row['capture_source']=binding(self.catalog,
            'system' if row['capture_source']['source_id']=='microphone_clean' else 'microphone_clean')
        with self.assertRaisesRegex(ValueError,'identity'):manager.refinement.canonical(self.jid,
            {'candidate':row,'fast_sequence':saved['last_fast_sequence']+1})
    def test_voice_choice_preview_and_extraction_use_exact_owned_source(self):
        self.feed();self.runtime.handle({'type':'stop','request_id':'stop'});job=self.job()
        choices=voice_clip_choices(job,'speaker_0');self.assertTrue(choices)
        clip=next(iter(choices.values()));path=self.folder/'microphone_clean.wav'
        self.assertEqual(validate_voice_audio(self.root,path,clip),path.resolve())
        with self.assertRaisesRegex(ValueError,'outside'):validate_voice_audio(self.root,self.folder/'audio.wav',clip)
        with self.assertRaises(ValueError):validate_voice_audio(self.root,path,replace(clip,audio_revision=None))
        self.assertNotEqual(clip_pcm_digest(path,clip),clip_pcm_digest(self.folder/'audio.wav',clip))
        calls=[]
        backend=type('Backend',(),{'model':type('Model',(),{'dimension':2})(),
            'embed':lambda _self,a,c:(calls.append((a,c)) or ClipEmbedding((1.,0.),True))})()
        extract(backend,self.folder/'audio.wav',[clip]);self.assertEqual(calls[0][0],path.resolve())
        self.assertTrue(clips_current(job,[clip]))
        wav(path,12,112)
        with self.assertRaisesRegex(ValueError,'PCM'):extract(backend,self.folder/'audio.wav',[clip])
        self.assertEqual(len(calls),1)
        next(row for row in job['document']['segments'] if row['id']==clip.segment_id)['audio_revision']+=1
        self.assertFalse(clips_current(job,[clip]))
        legacy=VoiceClip(self.jid,'speaker_0','legacy',0.,3.)
        self.assertEqual(validate_voice_audio(self.root,self.folder/'audio.wav',legacy),(self.folder/'audio.wav').resolve())
        with self.assertRaisesRegex(ValueError,'outside'):validate_voice_audio(self.root,path,legacy)

    def test_wrong_source_timing_receipt_is_rejected_before_word_binding(self):
        runtime=self.runtime.engines['microphone_clean'].canonical;peer=self.peers['microphone_clean']
        with patch.object(peer,'alignment_supported',return_value=True,create=True),patch.object(peer,'align_canonical',
                return_value={'capture_source':binding(self.catalog,'system')},create=True):
            with self.assertRaisesRegex(ValueError,'Timing evidence'):
                runtime.align_reading({'start_sample':0,'end_sample':RATE},'unchanged',
                    {'language':'en','language_detection':{'reason':'override'}})

    def test_preview_and_remember_use_current_source_and_explicit_consent(self):
        from test_people import SyntheticBackend,POLICY
        import re
        self.feed(12);self.runtime.handle({'type':'stop','request_id':'stop'});job=self.job()
        near=next(r for r in job['document']['segments'] if r['speaker']=='speaker_0')
        job['duration']=12
        backend=SyntheticBackend();app=create_app(self.root/'preview-app',voice_backend=backend,voice_calibration=POLICY)
        manager=app.extensions['speakerdesk']['meetings'];self.addCleanup(manager.close)
        self.addCleanup(app.extensions['speakerdesk']['executor'].shutdown,wait=True,cancel_futures=True)
        dest=manager.folder(self.jid);dest.mkdir()
        for name in ('audio.wav','microphone_clean.wav','system.wav','capture-sources.json'):
            (dest/name).write_bytes((self.folder/name).read_bytes())
        manager.put(job);client=app.test_client()
        headers={'X-Speakerdesk-Token':re.search(r'name="speakerdesk-token" content="([^"]+)"',client.get('/').text)[1]}
        choices=client.get(f'/api/jobs/{self.jid}/speakers/speaker_0/voice-clips').json['clips']
        self.assertEqual(len(choices),2)
        preview=client.get(f'/api/jobs/{self.jid}/speakers/speaker_0/voice-preview',query_string={'clip':choices[0]['id'],'revision':0})
        self.assertEqual(preview.status_code,200);self.assertEqual(preview.data,(dest/'microphone_clean.wav').read_bytes())
        preview.close()
        self.assertEqual(client.get(f'/api/jobs/{self.jid}/speakers/speaker_0/voice-preview',
            query_string={'clip':choices[0]['id'],'revision':1}).status_code,409)
        person=client.post('/api/people',headers=headers,json={'name':'CPU Fixture Person'}).json['id']
        assigned=client.post(f'/api/jobs/{self.jid}/speakers/speaker_0/identity',headers=headers,
            json={'revision':0,'person_id':person});self.assertEqual(assigned.status_code,200)
        body={'meeting_id':self.jid,'track_id':'speaker_0','revision':assigned.json['revision'],
              'segment_ids':[c['id'] for c in choices]}
        self.assertEqual(client.post(f'/api/people/{person}/voice',headers=headers,json=body).status_code,400)
        self.assertFalse(backend.calls)
        response=client.post(f'/api/people/{person}/voice',headers=headers,json={**body,'consent':True})
        self.assertEqual(response.status_code,200,response.json)
        self.assertEqual([path for path,_ in backend.calls],[(dest/'microphone_clean.wav').resolve()]*2)
        self.assertTrue(all(clip.capture_source==binding(self.catalog,'microphone_clean') for _,clip in backend.calls))
        current=manager.get(self.jid);forged=copy.deepcopy(current['document'])
        next(r for r in forged['segments'] if r['id']==near['id'])['capture_source']=binding(self.catalog,'system')
        saved=client.put(f'/api/jobs/{self.jid}/transcript',headers=headers,
            json={'revision':current['revision'],'document':forged})
        self.assertEqual(saved.status_code,200,saved.json)
        restored=next(r for r in saved.json['document']['segments'] if r['id']==near['id'])
        self.assertEqual(restored['capture_source'],near['capture_source'])


if __name__=='__main__':unittest.main()
