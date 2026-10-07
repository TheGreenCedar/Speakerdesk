"""Fabricated CPU contract fixtures, not measured speaker/recognition accuracy."""
import copy
import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from reading_turns import (alignment_words, bind_words, project_turns, validated_turns,
                           CALIBRATION, MODEL)
from transcript import export
from review import review_reason
from app import create_app


def fixture(text='  go go café 👩🏽‍💻  next end  '):
    row={'id':'one','canonical_utterance_id':'one','machine_revision':2,
         'canonical_machine_revision':2,'audio_revision':3,'language_epoch':0,
         'start_sample':0,'end_sample':160000,'start':0,'end':10,
         'text':text,'text_audio_anchor':{'start_sample':0,'end_sample':160000},
         'protected_fields':[], 'speaker':'multiple_speakers','speaker_candidates':['speaker_0','speaker_1']}
    row['speaker_activity']={'audio_revision':3,'regions':[
        {'start_sample':0,'end_sample':80000,'speakers':['speaker_0']},
        {'start_sample':80000,'end_sample':160000,'speakers':['speaker_1']}]}
    words=[{'text':m.group(),'start_char':m.start(),'end_char':m.end(),
            'start_sample':(i+1)*16000,'end_sample':(i+1)*16000+3200}
           for i,m in enumerate(re.finditer(r'\S+',text))]
    bind_words(row,words)
    return row


def save_projection(row):
    row=copy.deepcopy(row);row['reading_turns']=project_turns(row)
    row['reading_turn_provenance']={
        'utterance_id':row['id'],'machine_revision':row['machine_revision'],
        'canonical_machine_revision':row['canonical_machine_revision'],
        'audio_revision':row['audio_revision'],'language_epoch':row['language_epoch'],
        'text_sha256':hashlib.sha256(row['text'].encode()).hexdigest(),
        'audio_anchor':row['text_audio_anchor'],
        'activity_sha256':hashlib.sha256(json.dumps(row['speaker_activity'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'turns_sha256':hashlib.sha256(json.dumps(row['reading_turns'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),
        'calibration_id':CALIBRATION,'model_sha256':MODEL}
    return row


class ReadingTurnTests(unittest.TestCase):
    def test_sequential_owners_never_become_overlap_and_raw_unicode_is_lossless(self):
        row=fixture();turns=project_turns(row)
        self.assertEqual(''.join(t['text'] for t in turns),row['text'])
        self.assertEqual([t['attribution'] for t in turns],['single','unknown','single'])
        self.assertEqual([t['speaker'] for t in turns],['speaker_0','unassigned','speaker_1'])
        self.assertIn('go go café 👩🏽‍💻',turns[0]['text'])
        self.assertEqual(row['text'],'  go go café 👩🏽‍💻  next end  ')

    def test_actual_concurrent_region_is_localized_with_exact_owner_set(self):
        row=fixture();row['speaker_activity']['regions'][0]['speakers']=['speaker_0','speaker_1']
        row['speaker_activity']['regions'][1]['speakers']=['speaker_1','speaker_2']
        turns=project_turns(row)
        self.assertEqual([t['attribution'] for t in turns],['overlap','unknown','overlap'])
        self.assertEqual(turns[0]['speaker'],'overlap_speaker_0_speaker_1')
        self.assertEqual(turns[2]['speaker'],'overlap_speaker_1_speaker_2')

    def test_transition_gap_and_missing_word_timing_remain_unknown(self):
        row=fixture();row['speaker_activity']['regions'][1]['speakers']=[]
        words=row['reading_word_evidence']['words'];words[0].update(start_sample=None,end_sample=None)
        turns=project_turns(row)
        self.assertEqual(turns[0]['attribution'],'unknown')
        self.assertEqual(turns[-1]['attribution'],'unknown')
        self.assertIsNone(turns[-1]['start'])

    def test_feed_cells_merge_before_transition_guard(self):
        row=fixture();row['speaker_activity']['regions']=[
            {'start_sample':i*16000,'end_sample':(i+1)*16000,'speakers':['speaker_0']} for i in range(10)]
        turns=project_turns(row)
        self.assertEqual(len(turns),1);self.assertEqual(turns[0]['attribution'],'single')

    def test_revision_text_anchor_protected_and_activity_invalidation(self):
        row=fixture()
        for key,value in [('machine_revision',4),('audio_revision',4),('text',row['text']+'x'),
                          ('text_audio_anchor',{'start_sample':0,'end_sample':1}),('protected_fields',['speaker'])]:
            with self.subTest(key=key):
                changed=copy.deepcopy(row);changed[key]=value
                self.assertEqual(project_turns(changed),[])
        changed=copy.deepcopy(row);changed['speaker_activity']['audio_revision']=4
        self.assertEqual(project_turns(changed),[])
        changed=copy.deepcopy(row);changed['speaker_activity']['regions'][0]['speakers']=['speaker_2']
        self.assertEqual(project_turns(changed)[0]['speaker'],'speaker_2')

    def test_unqualified_or_stale_alignment_and_core_seam_word_stays_null(self):
        row=fixture();text=row['text'];words=row['reading_word_evidence']['words']
        result={'raw_text':text,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
                'audio_anchor':row['text_audio_anchor'],'model_sha256':MODEL,
                'timing_kind':'ctc_emission_cell_envelope','frame_calibration_id':CALIBRATION,
                'score_calibration_id':CALIBRATION,'words':[dict(w,status='aligned') for w in words]}
        result['words'][0].update(start_sample=320,end_sample=640)
        valid=alignment_words(result,text,0,160000)
        self.assertIsNone(valid[0]['start_sample']);self.assertIsNotNone(valid[1]['start_sample'])
        for key in ('text_sha256','model_sha256','frame_calibration_id','score_calibration_id','audio_anchor'):
            broken=copy.deepcopy(result);broken[key]='stale'
            self.assertEqual(alignment_words(broken,text,0,160000),[])

    def test_export_binding_and_forged_slice_fail_closed(self):
        row=save_projection(fixture());self.assertTrue(validated_turns(row))
        for mutate in (lambda s:s['reading_turns'][0].update(text='replacement'),
                       lambda s:s['reading_turns'][0].update(speaker='speaker_1'),
                       lambda s:s.update(machine_revision=3),
                       lambda s:s['speaker_activity']['regions'][0].update(speakers=['speaker_2'])):
            broken=copy.deepcopy(row);mutate(broken);self.assertEqual(validated_turns(broken),[])

    def test_txt_local_labels_and_subtitle_null_fallback_keep_all_words(self):
        row=save_projection(fixture());doc={'speakers':{'multiple_speakers':'Multiple speakers',
            'speaker_0':'A','speaker_1':'B'},'segments':[row]}
        txt,_=export(doc,'txt');self.assertIn('] A:',txt);self.assertIn('] B:',txt)
        self.assertIn('Timing unknown; parent audio',txt)
        for word in ('go go café','👩🏽‍💻','next','end'):
            self.assertEqual(txt.count(word),1)
        srt,_=export(doc,'srt');self.assertEqual(srt.count('-->'),1)
        self.assertIn('Multiple speakers:',srt)
        self.assertNotIn('Overlapping speakers',srt)
        row['transcription_review']={'reason':'token_limit','partial_text':True}
        txt,_=export(doc,'txt')
        self.assertIn('Transcript may be incomplete; audio retained',txt)

    def test_parent_review_does_not_infer_simultaneous_speech_from_union(self):
        row=fixture();self.assertNotIn('Overlapping',review_reason(row))
        row['speaker']='overlap_speaker_0_speaker_1'
        self.assertIn('Overlapping',review_reason(row))

    def test_server_saved_edit_and_import_cannot_retain_or_forge_turns(self):
        with tempfile.TemporaryDirectory() as folder:
            app=create_app(folder);client=app.test_client();ext=app.extensions['speakerdesk'];manager=ext['meetings']
            token=re.search(r'name="speakerdesk-token" content="([^"]+)"',client.get('/').get_data(as_text=True)).group(1)
            client.environ_base['HTTP_X_SPEAKERDESK_TOKEN']=token
            row=save_projection(fixture());doc={'speakers':{'multiple_speakers':'Multiple speakers',
                'speaker_0':'A','speaker_1':'B'},'segments':[row]}
            jid='e'*32;job={'id':jid,'kind':'meeting','name':'CPU turn contract','created':1,
                'status':'ready','duration':10,'language':'auto','revision':0,'document':doc}
            try:
                manager.put(job)
                forged=copy.deepcopy(doc);forged['segments'][0]['reading_turns'][0]['speaker']='speaker_1'
                saved=client.put('/api/jobs/'+jid+'/transcript',json={'revision':0,'document':forged})
                self.assertEqual(saved.status_code,200)
                current=manager.get(jid);self.assertEqual(current['document']['segments'][0]['reading_turns'],row['reading_turns'])
                edited=client.patch('/api/jobs/'+jid+'/segments/one',json={'segment_revision':2,'changes':{'text':'Human correction'}})
                self.assertEqual(edited.status_code,200,edited.get_json())
                self.assertNotIn('reading_turns',manager.get(jid)['document']['segments'][0])
                imported=copy.deepcopy(doc);imported['segments'][0]['id']='new-import'
                response=client.put('/api/jobs/'+jid+'/transcript',json={'revision':manager.get(jid)['revision'],'document':imported,'imported':True})
                self.assertEqual(response.status_code,200,response.get_json())
                imported_row=manager.get(jid)['document']['segments'][0]
                self.assertNotIn('reading_turns',imported_row);self.assertNotIn('reading_turn_provenance',imported_row)
            finally:
                manager.close();ext['executor'].shutdown(wait=True,cancel_futures=True)

    def test_english_routing_keeps_timing_independent_of_lid_confidence(self):
        from canonical_runtime import CanonicalRuntime
        from types import SimpleNamespace
        calls=[]
        models=SimpleNamespace(align_canonical=lambda request,text,**kw:(calls.append((request,text,kw)) or {'words':[]}))
        runtime=object.__new__(CanonicalRuntime);runtime.engine=SimpleNamespace(models=models)
        for reason,review in [('recent_context',True),('detected',True),('best_effort',True),
                              ('detected',False),('override',False)]:
            result=runtime.align_reading({'start_sample':0,'end_sample':16000},'RAW unchanged',
                {'language':'en','language_review':review,'language_detection':{'reason':reason}})
            self.assertEqual(result['reading_language_review'],review)
        self.assertEqual(len(calls),5)
        self.assertTrue(all(x[1]=='RAW unchanged' and x[2]=={'language':'en'} for x in calls))
        for passage in [{'language':'fr','language_detection':{'reason':'detected'}},
                        {'language':'en','language_detection':{'reason':'unsupported'}},
                        {'language':'en','language_detection':{'reason':'mixed_languages'}},
                        {'language':'en','transcription_review':{'reason':'empty_result'}}]:
            self.assertIsNone(runtime.align_reading({'start_sample':0,'end_sample':16000},'RAW unchanged',passage))
        self.assertEqual(len(calls),5)

    def test_real_routing_review_for_multiple_owners_does_not_block_english_timing(self):
        from test_canonical_runtime import CanonicalTests
        from pcm_peer import SpeechEvidencePeer, varying_pcm
        from language_detection import SpeechTranscriber
        from types import SimpleNamespace
        import numpy as np
        import wave
        case=CanonicalTests();case.setUp()
        try:
            # This route represents admitted speech, so use a varying quiet
            # waveform instead of the helper's constant DC transport sentinel.
            with wave.open(str(case.path),'wb') as wav:
                wav.setparams((1,2,16000,0,'NONE','none'))
                wav.writeframes(varying_pcm(60*16000,1,dtype='<i2').tobytes())
            case.peer.mixed=True;asr_calls=[];timing_calls=[]
            asr=SimpleNamespace(transcribe=lambda audio,**kw:(asr_calls.append(len(audio)) or
                SimpleNamespace(text=case.peer.text,tokens=[1])))
            detector=SimpleNamespace(detect=lambda pcm:{'en':.99,'fr':.01})
            case.peer.routing_start=0
            case.peer.set_language_context=lambda context,start:setattr(case.peer,'routing_start',start)
            def transcribe(audio,language,names,overlap=False):
                return SpeechTranscriber(asr,'auto',detector=detector,speech_evidence=SpeechEvidencePeer()).transcribe(
                    np.asarray(audio),16000,tuple(names),max_asr_seconds=24.5,allow_overlap=True,
                    start_sample=case.peer.routing_start)
            def align(request,text,**kw):
                timing_calls.append((request,text,kw))
                words=[{'text':m.group(),'start_char':m.start(),'end_char':m.end(),'status':'aligned',
                        'start_sample':request['start_sample']+(i+1)*1600+4800,
                        'end_sample':request['start_sample']+(i+1)*1600+5120}
                    for i,m in enumerate(re.finditer(r'\S+',text))]
                return {'raw_text':text,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
                    'audio_anchor':{'start_sample':request['start_sample'],'end_sample':request['end_sample']},
                    'model_sha256':MODEL,'timing_kind':'ctc_emission_cell_envelope',
                    'frame_calibration_id':CALIBRATION,'score_calibration_id':CALIBRATION,'words':words}
            case.peer.transcribe=transcribe;case.peer.alignment_supported=lambda language:language=='en'
            case.peer.align_canonical=align
            case.engine.config['language']='auto';case.engine.timeline[0]['language']='auto'
            case.feed(0,7,mode='auto');case.engine.handle({'type':'stop'})
            row=case.rows()[-1]
            self.assertEqual(row['text'],case.peer.text)
            self.assertEqual(len(timing_calls),2)  # Open revision and sealed revision.
            self.assertEqual(sum(n>24.5*16000 for n in asr_calls),0)
            self.assertIn('reading_turns',row)
            self.assertEqual(''.join(t['text'] for t in row['reading_turns']),row['text'])
            self.assertTrue(all(t['attribution']=='unknown' for t in row['reading_turns']))
            self.assertEqual(row['speaker'],'unassigned')
            self.assertTrue(row['review'])
        finally:case.tearDown()

    def test_weak_english_probes_do_not_destroy_independent_acoustic_timing(self):
        from canonical_runtime import CanonicalRuntime
        from types import SimpleNamespace
        text='clear weak clear';words=[{'text':m.group(),'start_char':m.start(),'end_char':m.end(),
            'status':'aligned','start_sample':a,'end_sample':a+320} for m,a in
            zip(re.finditer(r'\S+',text),(16000,64000,112000))]
        result={'words':words};calls=[]
        runtime=object.__new__(CanonicalRuntime)
        runtime.engine=SimpleNamespace(models=SimpleNamespace(align_canonical=lambda *args,**kw:calls.append(args) or result))
        passage={'language':'en','review':True,'language_review':True,'language_detection':{'reason':'recent_context','probes':[
            {'start_sample':0,'end_sample':48000,'language':'en','review':False,'decision':{'reason':'detected'}},
            {'start_sample':48000,'end_sample':96000,'language':'en','review':True,'decision':{'reason':'recent_context'}},
            {'start_sample':96000,'end_sample':144000,'language':'en','review':False,'decision':{'reason':'detected'}}]}}
        projected=runtime.align_reading({'start_sample':0,'end_sample':144000},text,passage)
        self.assertEqual(len(calls),1)
        self.assertEqual([w['start_sample'] for w in projected['words']],[16000,64000,112000])
        self.assertEqual(result['words'][1]['start_sample'],64000)
        self.assertTrue(projected['reading_language_review'])
        self.assertEqual(projected['reading_timing_qualification'],'english_decoder_and_calibrated_acoustic_evidence')
        # The same actual acoustic weakness stays null regardless of LID reason.
        result['words'][1].update(status='unresolved',start_sample=None,end_sample=None)
        projected=runtime.align_reading({'start_sample':0,'end_sample':144000},text,passage)
        self.assertIsNone(projected['words'][1]['start_sample'])
        # Mixed/non-English, stale probe anchors and incomplete partitions do
        # not borrow the English calibration or execute the provider.
        for modify in (lambda p:p['language_detection']['probes'][1].update(language='fr'),
                       lambda p:p['language_detection']['probes'][1].update(start_sample=48001),
                       lambda p:p['language_detection']['probes'].pop(),
                       lambda p:p['language_detection'].update(probes=[]),
                       lambda p:p['language_detection'].update(probes='invalid'),
                       lambda p:p['language_detection']['probes'].__setitem__(1,None),
                       lambda p:p['language_detection']['probes'][1].update(decision='invalid')):
            broken=copy.deepcopy(passage);modify(broken)
            self.assertIsNone(runtime.align_reading({'start_sample':0,'end_sample':144000},text,broken))
        self.assertEqual(len(calls),2)

    def test_lid_confidence_cannot_change_clean_ownership_or_acoustic_unknown(self):
        from canonical_runtime import CanonicalRuntime
        from types import SimpleNamespace
        row=fixture();text=row['text'];words=copy.deepcopy(row['reading_word_evidence']['words'])
        words[1].update(status='unresolved',start_sample=None,end_sample=None)
        result={'raw_text':text,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
                'audio_anchor':row['text_audio_anchor'],'model_sha256':MODEL,
                'timing_kind':'ctc_emission_cell_envelope','frame_calibration_id':CALIBRATION,
                'score_calibration_id':CALIBRATION,'words':[dict(w,status=w.get('status','aligned')) for w in words]}
        runtime=object.__new__(CanonicalRuntime)
        runtime.engine=SimpleNamespace(models=SimpleNamespace(align_canonical=lambda *args,**kw:result))
        projections=[]
        for reason,review in [('detected',False),('recent_context',True),('best_effort',True)]:
            current=copy.deepcopy(row)
            passage={'language':'en','language_review':review,'language_detection':{'reason':reason,'probes':[
                {'start_sample':0,'end_sample':80000,'language':'en','review':review,'decision':{'reason':reason}},
                {'start_sample':80000,'end_sample':160000,'language':'en','review':review,'decision':{'reason':reason}}]}}
            aligned=runtime.align_reading({'start_sample':0,'end_sample':160000},text,passage)
            bind_words(current,alignment_words(aligned,text,0,160000))
            projections.append(project_turns(current))
        self.assertEqual(projections[0],projections[1]);self.assertEqual(projections[1],projections[2])
        self.assertEqual(projections[0][0]['speaker'],'speaker_0')
        self.assertTrue(any(t['attribution']=='unknown' for t in projections[0]))
        self.assertIsNone(result['words'][1]['start_sample'])


if __name__=='__main__':unittest.main()
