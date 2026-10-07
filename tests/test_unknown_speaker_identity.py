"""Real Engine→saved API identity controls with fabricated CPU model evidence.

These distinguish acoustic slots, meeting correspondence and word attribution;
they do not measure diarization/recognition accuracy.
"""
import copy
from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile
import unittest
import wave

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'speakerdesk'),
               str(Path(__file__).resolve().parent)]
from app import create_app
from live_refinement import Engine, Inbox
from meeting_refinement import activity_references
from test_sequential_product_path import SequentialPeer, RATE, TEXT


@contextmanager
def source_job(*, timing=False, single=False):
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder);audio = root/'synthetic.wav'
        with wave.open(str(audio),'wb') as wav:
            wav.setparams((1,2,RATE,0,'NONE','none'))
            wav.writeframes(b'\x01\x00\xff\xff'*(10*RATE//2))
        peer = SequentialPeer()
        if not timing:peer.align_canonical = lambda *args, **kwargs:None
        if single:
            original = peer.feed
            def feed(pcm, final=False):
                turns, observed = original(pcm, final)
                return [dict(turn,speaker='speaker_0') for turn in turns], observed
            peer.feed = feed
        jid = 'a'*32;events = []
        engine = Engine(dict(audio_path=str(audio),language='en',job_id=jid,
            canonical_utterances=True),peer,events.append,Inbox())
        for second in range(10):
            engine.handle(dict(type='audio',start_sample=second*RATE,
                end_sample=(second+1)*RATE,language='en',language_epoch=0))
        engine.handle(dict(type='stop'))
        app = create_app(root/'home');manager = app.extensions['speakerdesk']['meetings']
        job = dict(id=jid,created=1,status='ready',kind='meeting',name='CPU identity control',
            language='en',duration=10,revision=0,canonical_utterances=True,
            document=dict(speakers={},segments=[],provenance={},warnings=[]))
        manager.duration = 10;manager.refinement.initialize(job);manager.put(job)
        try:
            for event in events:
                if event['type']=='canonical_revision':manager.refinement.canonical(jid,event)
            yield dict(engine=engine,peer=peer,app=app,manager=manager,jid=jid,
                       client=app.test_client())
        finally:
            manager.close()
            app.extensions['speakerdesk']['executor'].shutdown(wait=True,cancel_futures=True)


def saved(context):
    response = context['client'].get('/api/jobs/'+context['jid'])
    if response.status_code!=200:raise AssertionError(response.status_code)
    return response.json


def refine(context, turns):
    source = saved(context)['document']['segments'][-1]
    context['peer'].batch_turns = lambda pcm:copy.deepcopy(turns)
    request = dict(type='refine',operation_id='identity-control',language='en',language_epoch=0,
        window=dict(id='identity-window',start_sample=0,end_sample=10*RATE,
                    context_start_sample=0,context_end_sample=10*RATE),
        references=activity_references([source],0,10*RATE),canonical=source)
    result = context['engine'].refine(request)
    context['manager'].refinement.canonical(context['jid'],dict(
        fast_sequence=result['fast_sequence'],candidate=result['canonical_candidate']))
    return saved(context)


PAIR = [dict(start=0,end=5,speaker='speaker_7'),dict(start=5,end=10,speaker='speaker_8')]


class UnknownSpeakerIdentityTests(unittest.TestCase):
    def test_distinct_tracks_and_qualified_timing_retain_two_unnamed_identities(self):
        with source_job(timing=True) as context:
            document = saved(context)['document'];row = document['segments'][-1]
            self.assertEqual(row['speaker_candidates'], ['speaker_0','speaker_1'])
            self.assertEqual({turn['speaker'] for turn in row['reading_turns']},
                             {'speaker_0','speaker_1'})
            self.assertEqual({key:document['speakers'][key] for key in ('speaker_0','speaker_1')},
                             dict(speaker_0='Speaker 1',speaker_1='Speaker 2'))
            self.assertEqual(row['text'], TEXT)

    def test_missing_timing_keeps_distinct_activity_without_inventing_word_identity(self):
        with source_job() as context:
            row = saved(context)['document']['segments'][-1]
            self.assertEqual(row['speaker'], 'unassigned')
            self.assertEqual(row['speaker_candidates'], ['speaker_0','speaker_1'])
            self.assertEqual({name for region in row['speaker_activity']['regions']
                              for name in region['speakers']}, {'speaker_0','speaker_1'})
            self.assertNotIn('reading_turns',row)
            self.assertEqual(row['text'], TEXT)

    def test_batch_separation_preserves_unpaired_local_slots_in_saved_source_evidence(self):
        with source_job(single=True) as context:
            row = refine(context,PAIR)['document']['segments'][-1]
            self.assertEqual(row['speaker'], 'unassigned')
            self.assertNotIn('reading_turns',row)
            self.assertEqual(row['text'], TEXT)
            # Refusing an ambiguous meeting match is correct. Discarding the
            # batch model's distinct local slots is not required by that refusal.
            evidence = row.get('speaker_track_mapping')
            self.assertIsInstance(evidence,dict,'distinct raw batch slots disappeared into unknown_mixed')
            self.assertEqual(evidence['mapping'], {})
            self.assertEqual(evidence['raw_batch_turns'], PAIR)
            self.assertEqual({score['local_speaker'] for score in evidence['scores']},
                             {'speaker_7','speaker_8'})
            self.assertTrue(all(score['reason']=='ambiguous_claim' for score in evidence['scores']))
            self.assertEqual(evidence['audio_anchor'],dict(start_sample=0,end_sample=10*RATE))
            self.assertEqual(evidence['audio_revision'],row['audio_revision'])
            self.assertEqual(evidence['utterance_id'],row['canonical_utterance_id'])
            self.assertEqual(evidence['operation_id'],'identity-control')
            self.assertEqual([region['speakers'] for region in evidence['local_activity']],
                             [['speaker_7'],['speaker_8']])

    def test_successful_batch_correspondence_preserves_existing_named_tracks(self):
        with source_job(timing=True) as context:
            job = saved(context);job['document']['speakers'].update(speaker_0='Alice',speaker_1='Bob')
            context['manager'].put(job)
            document = refine(context,PAIR)['document'];row = document['segments'][-1]
            self.assertEqual(row['speaker_candidates'], ['speaker_0','speaker_1'])
            self.assertEqual((document['speakers']['speaker_0'],document['speakers']['speaker_1']),
                             ('Alice','Bob'))
            self.assertEqual({turn['speaker'] for turn in row['reading_turns']},
                             {'speaker_0','speaker_1'})
            evidence = row.get('speaker_track_mapping')
            self.assertIsInstance(evidence,dict)
            self.assertEqual(evidence['mapping'],dict(speaker_7='speaker_0',speaker_8='speaker_1'))

    def test_same_acoustic_slot_does_not_fabricate_a_second_person(self):
        with source_job(single=True) as context:
            row = refine(context,[dict(start=0,end=10,speaker='speaker_7')])['document']['segments'][-1]
            self.assertEqual(row['speaker'], 'speaker_0')
            self.assertEqual(row['speaker_candidates'], ['speaker_0'])
            self.assertNotIn('reading_turns',row)
            self.assertEqual(row['text'], TEXT)

    def test_failed_refinement_keeps_previous_mapping_and_saved_identity(self):
        with source_job(single=True) as context:
            before = refine(context,PAIR)
            old = copy.deepcopy(before['document']['segments'][-1]['speaker_track_mapping'])
            def failed(*args, **kwargs):raise RuntimeError('CPU decoder failure')
            context['peer'].transcribe = failed
            with self.assertRaisesRegex(RuntimeError,'CPU decoder failure'):
                refine(context,[dict(start=0,end=5,speaker='speaker_9'),
                                dict(start=5,end=10,speaker='speaker_10')])
            self.assertEqual(saved(context),before)
            self.assertEqual(context['engine'].canonical.book.snapshot()[-1]['speaker_track_mapping'],old)

    def test_unmatched_person_names_do_not_merge_two_acoustic_track_ids(self):
        from test_voice_recognition import RecognitionTests, segment
        case = RecognitionTests();case.setUp()
        try:
            case.person();job = case.seed(segments=[segment('a1',0,3),segment('a2',4,7),
                dict(segment('b1',8,11),speaker='speaker_1'),
                dict(segment('b2',12,15),speaker='speaker_1')])
            job['document']['speakers']['speaker_1'] = 'Speaker 2'
            before = copy.deepcopy(job['document']);case.recognizer.put(job)
            case.backend.vector = (0.,1.)  # Synthetic mismatch to saved person.
            case.recognizer.observe(case.jid);case.drain()
            response = case.client.get('/api/jobs/'+case.jid)
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.json['document'],before)
            self.assertEqual({key:check['status'] for key,check in response.json['voice_checks'].items()},
                             dict(speaker_0='unknown',speaker_1='unknown'))
            self.assertNotIn('speaker_assignments',response.json)
        finally:case.tearDown()


if __name__=='__main__':unittest.main()
