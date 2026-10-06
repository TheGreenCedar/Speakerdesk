"""CPU routing/ownership checks; injected text is not acoustic model evidence."""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from pipeline import decode_regions
from live_refinement import context_regions,align_tracks,Engine,coalesce_blanks
from meeting_refinement import activity_references,preceding_language_context
from rolling_refinement import reconcile_window,segment_version,RATE
from language_detection import SpeechTranscriber
from unittest.mock import Mock
from types import SimpleNamespace
import numpy as np
from pcm_peer import varying_pcm, SpeechEvidencePeer


class TurnEnvelopeTests(unittest.TestCase):
    def turns(self,gap=.05,second='speaker_0'):
        return [{'start':0,'end':1,'speaker':'speaker_0'},
                {'start':1+gap,'end':2+gap,'speaker':second},
                {'start':2+2*gap,'end':3+2*gap,'speaker':second}]
    def test_short_internal_holes_share_audio_without_inventing_activity(self):
        for gap in (.01,.05,.2,.4,.65):
            with self.subTest(gap=gap):
                end=3+2*gap;regions=decode_regions(self.turns(gap),coverage=(0,end))
                self.assertEqual(len(regions),1)
                self.assertEqual((regions[0]['start'],regions[0]['end']),(0,end))
                self.assertEqual(regions[0]['speakers'],['speaker_0'])
                self.assertEqual([p['speakers'] for p in regions[0]['activity_regions']],
                                 [['speaker_0'],[],['speaker_0'],[],['speaker_0']])
                self.assertEqual(context_regions(self.turns(gap),coverage=(0,end)),regions)
    def test_real_switch_long_pause_and_outer_unassigned_ranges_are_barriers(self):
        turns=[{'start':1,'end':3,'speaker':'speaker_0'},
               {'start':3.1,'end':5.1,'speaker':'speaker_1'},
               {'start':6.1,'end':8.1,'speaker':'speaker_1'}]
        regions=decode_regions(turns,coverage=(0,9))
        self.assertEqual([(r['start'],r['end'],r['speakers']) for r in regions],
            [(0,1,[]),(1,3,['speaker_0']),(3,3.1,[]),(3.1,5.1,['speaker_1']),
             (5.1,6.1,[]),(6.1,8.1,['speaker_1']),(8.1,9,[])])
        switched=decode_regions([{'start':0,'end':1,'speaker':'speaker_0'},
                                 {'start':1,'end':2,'speaker':'speaker_1'}])
        self.assertEqual([r['speakers'] for r in switched],[['speaker_0'],['speaker_1']])
    def test_duration_caps_and_ownership_clip_activity_ledger(self):
        turns=[{'start':0,'end':17.8,'speaker':'speaker_0'},
               {'start':18,'end':23,'speaker':'speaker_0'}]
        rows=decode_regions(turns,max_seconds=18,coverage=(1,22))
        self.assertEqual([(r['start'],r['end']) for r in rows],[(1,19),(19,22)])
        self.assertTrue(all(r['start']<=p['start']<p['end']<=r['end']
                            for r in rows for p in r['activity_regions']))
    def test_merge_reanchors_playback_and_retains_previous_revision_and_edits(self):
        old=[dict(id='a',start=0,end=1,text='So tell me',speaker='speaker_0',machine_revision=1,
                  audio_anchor={'start_sample':0,'end_sample':RATE}),
             dict(id='b',start=1,end=3.1,text='how will you fix transcription?',speaker='speaker_0',machine_revision=1)]
        doc={'speakers':{'speaker_0':'Albert'},'segments':copy.deepcopy(old)}
        expected={r['id']:segment_version(r) for r in old}
        candidate=dict(start=0,end=3.1,text='So tell me, how will you fix transcription?',speaker='speaker_0')
        win={'id':'turn','start_sample':0,'end_sample':round(3.1*RATE)}
        merged=reconcile_window(doc,win,expected,[candidate])
        self.assertEqual(len(merged['document']['segments']),1)
        self.assertEqual(merged['document']['segments'][0]['audio_anchor'],{'start_sample':0,'end_sample':49600})
        self.assertEqual(merged['previous_revision']['segments'],old)
        doc['segments'][0]['protected_fields']=['text'];expected['a']=segment_version(doc['segments'][0])
        self.assertEqual(reconcile_window(doc,win,expected,[candidate])['document'],doc)
    def test_weak_same_language_probes_decode_whole_sentence_keep_uncertainty(self):
        detector=Mock();detector.detect.side_effect=[{'en':.6,'fr':.4}]*3
        asr=Mock();asr.transcribe.return_value=SimpleNamespace(text='So tell me, how are you going to be fixing transcription?',tokens=[1,2])
        rows=SpeechTranscriber(asr,'auto',detector=detector,context={'language':'en','end_sample':0},speech_evidence=SpeechEvidencePeer()).transcribe(
            varying_pcm(9*RATE,.1),RATE,('speaker_0',),max_asr_seconds=18)
        self.assertEqual(len(rows),1);asr.transcribe.assert_called_once()
        self.assertEqual(len(asr.transcribe.call_args.args[0]),9*RATE)
        self.assertTrue(rows[0]['review']);self.assertEqual(len(rows[0]['language_detection']['probes']),3)
    def test_raw_gap_is_not_reference_evidence_for_an_offline_speaker(self):
        region=decode_regions([{'start':0,'end':1,'speaker':'speaker_0'},
                               {'start':1.6,'end':2.6,'speaker':'speaker_0'}],coverage=(0,2.6))[0]
        source={**region,'speaker':'speaker_0','speaker_candidates':['speaker_0']}
        refs=activity_references([source],0,round(2.6*RATE))
        self.assertEqual(align_tracks([{'start':1.1,'end':1.5,'speaker':'speaker_7'}],refs),{})
        clipped=activity_references([{**source,'start':.8,'end':1.8}],0,round(2.6*RATE))
        self.assertTrue(all(.8<=r['start']<r['end']<=1.8 for r in clipped))
    def test_brief_overlap_does_not_absorb_later_real_speaker_switches(self):
        turns=[{'start':0,'end':3,'speaker':'speaker_0'},
               {'start':2.99,'end':9,'speaker':'speaker_1'},
               {'start':9,'end':12,'speaker':'speaker_2'}]
        regions=decode_regions(turns)
        self.assertEqual([(r['start'],r['end'],r['speakers']) for r in regions],
            [(0,2.99,['speaker_0']),(2.99,3,['speaker_0','speaker_1']),
             (3,9,['speaker_1']),(9,12,['speaker_2'])])
    def test_blank_coalescence_cannot_discard_activity_or_probe_chronology(self):
        first={'start':0,'end':1,'text':'','speaker':'speaker_0',
               'activity_regions':[{'start':0,'end':1,'speakers':['speaker_0']}]}
        second={**first,'start':1,'end':2,
                'activity_regions':[{'start':1,'end':2,'speakers':['speaker_0']}]}
        self.assertEqual(coalesce_blanks([first,second]),[first,second])
        first.pop('activity_regions');second.pop('activity_regions')
        first['language_detection']={'reason':'best_effort','probes':[{'start_sample':0,'end_sample':RATE}]}
        second['language_detection']={'reason':'best_effort','probes':[{'start_sample':RATE,'end_sample':2*RATE}]}
        self.assertEqual(coalesce_blanks([first,second]),[first,second])
    def test_merged_warning_preserves_successful_probe_context_and_failed_switch_barrier(self):
        detector=Mock();detector.detect.side_effect=[{'en':.99,'fr':.01},{'en':.6,'fr':.4},{'fr':.6,'en':.4}]
        asr=Mock();asr.transcribe.return_value=SimpleNamespace(text='English words.',tokens=[1,2])
        transcriber=SpeechTranscriber(asr,'auto',detector=detector,speech_evidence=SpeechEvidencePeer())
        pcm=varying_pcm(9*RATE,.1)
        rows=transcriber.transcribe(pcm,RATE,('speaker_0',),max_asr_seconds=18)
        self.assertEqual([c.kwargs['language'] for c in asr.transcribe.call_args_list],['en','en'])
        self.assertEqual(transcriber.context,{'language':'en','end_sample':3*RATE})
        epoch_rows=[{**r,'language_epoch':0} for r in rows]
        self.assertEqual(preceding_language_context(epoch_rows,9*RATE,0),transcriber.context)
        engine=Engine.__new__(Engine);engine.language_observations=[];engine.models=Mock()
        engine.models.transcribe.return_value=rows
        engine.decode(pcm,'auto',['speaker_0'],0,0)
        self.assertEqual(engine.language_observations,[{'epoch':0,'language':'en','end_sample':3*RATE}])
        failed={**epoch_rows[0],'text':'','transcription_review':{'reason':'empty_result'},
                'language_detection':{'reason':'best_effort','probes':[{
                    'start_sample':0,'end_sample':3*RATE,'language':'fr','decision':{'reason':'detected'}}]}}
        prior={'start':-3,'end':0,'language_epoch':0,'language':'en','text':'English.',
               'language_detection':{'reason':'detected'}}
        self.assertIsNone(preceding_language_context([prior,failed],9*RATE,0))


if __name__=='__main__':unittest.main()
