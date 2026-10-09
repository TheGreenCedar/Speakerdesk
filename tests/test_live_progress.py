"""Scheduling/publication contracts; model peers are not acoustic evidence."""
import json
import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(os.getenv('LIVE_PROGRESS_SOURCE',Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(ROOT/'speakerdesk'))
from live_refinement import Inbox,Engine
from speech_admission import SpeechFrames
RATE=16000

class Peer:
    def __init__(self):
        self.received=0;self.calls=[];self.speech_live=SimpleNamespace(evidence=SpeechFrames())
    def feed(self,audio,final=False):
        start=self.received;self.received+=len(audio)
        end=self.received if final else self.received//512*512
        while self.speech_live.evidence.end_sample<end:
            a=self.speech_live.evidence.end_sample;self.speech_live.evidence.append(a,min(a+512,end),.9)
        return ([{'start':start/RATE,'end':self.received/RATE,'speaker':'speaker_0'}] if len(audio) else []),end/RATE
    def transcribe(self,audio,language,names,overlap=False):
        self.calls.append(len(audio))
        return [{'start':0,'end':len(audio)/RATE,'text':' café repeat repeat ','cohere_raw_text':' café repeat repeat ','language':language}]
    def metrics(self):return {}

class LiveProgressTests(unittest.TestCase):
    def audio(self,inbox,seconds):
        for n in range(seconds):inbox.push({'type':'audio','start_sample':n*RATE,'end_sample':(n+1)*RATE,'language':'en','language_epoch':0})
    def request(self,name):return {'type':'refine','operation_id':name,'language_epoch':0,'canonical':{'canonical_state':'sealed'}}
    def test_pending_sealed_work_gets_a_turn_between_bounded_audio_drains(self):
        inbox=Inbox(audio_batch_seconds=6);self.audio(inbox,60);inbox.push(self.request('sealed'))
        self.assertEqual(inbox.take()['type'],'audio')
        self.assertEqual(inbox.take()['operation_id'],'sealed')
        inbox.finish_refinement('sealed');inbox.push(self.request('next'))
        self.assertEqual(inbox.take()['type'],'audio')
        self.assertEqual(inbox.take()['operation_id'],'next')
    def test_ordered_control_and_cancel_still_win(self):
        inbox=Inbox(audio_batch_seconds=6);self.audio(inbox,1)
        self.assertEqual(inbox.take()['type'],'audio')
        inbox.push({'type':'flush','request_id':'pause','through_sample':RATE});inbox.push(self.request('sealed'))
        self.assertEqual(inbox.take()['type'],'flush')
        inbox.push({'type':'cancel_refinement','operation_id':'sealed'})
        self.assertIsNone(inbox.refinement)
        inbox.push({'type':'shutdown'});self.assertEqual(inbox.take()['type'],'shutdown')
    def test_refinement_does_not_starve_audio_when_next_request_is_already_pending(self):
        inbox=Inbox(audio_batch_seconds=6);self.audio(inbox,20);inbox.push(self.request('first'))
        inbox.take();self.assertEqual(inbox.take()['operation_id'],'first');inbox.finish_refinement('first')
        inbox.push(self.request('second'));self.assertEqual(inbox.take()['type'],'audio')
    def test_long_open_speech_keeps_advancing_without_optional_alignment(self):
        for language in ('en','fr'):
            with self.subTest(language=language),tempfile.TemporaryDirectory() as t:
                path=Path(t)/'audio.wav'
                with wave.open(str(path),'wb') as w:w.setparams((1,2,RATE,0,'NONE','none'));w.writeframes(b'\x01\x00'*60*RATE)
                peer=Peer();events=[];engine=Engine({'audio_path':str(path),'language':language,'job_id':'b'*32,'canonical_utterances':True},peer,events.append,Inbox())
                for n in range(37):engine.handle({'type':'audio','start_sample':n*RATE,'end_sample':(n+1)*RATE,'language':language,'language_epoch':0})
                revisions=[e['candidate'] for e in events if e['type']=='canonical_revision']
                selected=revisions[-1]
                self.assertEqual(selected['canonical_state'],'open')
                self.assertGreater(selected['text_audio_anchor']['end_sample'],30*RATE)
                self.assertGreater(selected['canonical_machine_revision'],revisions[25]['canonical_machine_revision'])
                self.assertEqual(len({r['id'] for r in revisions}),1)
                self.assertFalse(selected.get('canonical_unresolved'))
                self.assertNotIn('reading_turns',selected)
                self.assertTrue(all(r['text'].strip() for r in revisions if r['canonical_machine_revision']))
                self.assertLessEqual(max(peer.calls),24.5*RATE)
                self.assertIsNone(selected['bounded_decode_provenance']['word_timing'])
                journal=[json.loads(s) for s in engine.canonical.archive.path.read_text().splitlines()]
                versions=[r for r in journal if r['type']=='disjoint_core_machine_version']
                self.assertGreaterEqual(len(versions),2)
                for version in versions:
                    self.assertTrue(all(p['text']==' café repeat repeat ' for p in version['parts']))

if __name__=='__main__':unittest.main()
