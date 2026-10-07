"""Recorded model-output assembly regressions; CPU peers are not acoustic proof."""
import copy,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from canonical_runtime import routed_short_revision
from test_canonical_runtime import CanonicalTests

FIXTURE=json.loads((Path(__file__).parent/'fixtures/routed-fallback-passages.json').read_text())
RECORD=FIXTURE['record']
RECORDED_BY_LENGTH={r['audio_anchor']['end_sample']-r['audio_anchor']['start_sample']:r['passages'] for r in FIXTURE['records']}

class RoutedRevisionTests(unittest.TestCase):
 def test_recorded_complete_english_routes_preserve_every_raw_string(self):
  before=copy.deepcopy(RECORD);result=routed_short_revision(RECORD['passages'],0,99968)
  self.assertEqual(result['text'],' '.join(p['cohere_raw_text'] for p in RECORD['passages']))
  self.assertEqual([x['passage'] for x in result['provenance']['parts']],RECORD['passages'])
  self.assertEqual(result['provenance']['audio_anchor'],RECORD['audio_anchor']);self.assertIsNone(result['provenance']['word_timing']);self.assertEqual(RECORD,before)
 def test_gaps_overlaps_incomplete_or_mixed_language_cannot_be_complete(self):
  variants=[('gap',lambda p:p[1].update(start=p[1]['start']+1/16000)),('overlap',lambda p:p[1].update(start=p[1]['start']-1/16000)),('tail',lambda p:p[1].update(end=p[1]['end']-1/16000)),('mixed',lambda p:p[1].update(language='fr')),('unsupported',lambda p:p[1].update(language='xx')),('empty',lambda p:p[1].update(cohere_raw_text='')),('error',lambda p:p[1].update(transcription_review={'reason':'transcription_failed'})),('uncertain',lambda p:p[1]['acoustic_evidence'].update(uncertain_regions=[{'start_sample':66645,'end_sample':67000}])),('wrong-anchor',lambda p:p[1]['acoustic_evidence'].update(start_sample=66646)),('pending',lambda p:p[1]['acoustic_evidence'].update(complete=False))]
  for name,mutate in variants:
   with self.subTest(name=name):
    passages=copy.deepcopy(RECORD['passages']);mutate(passages);self.assertIsNone(routed_short_revision(passages,0,99968))
 def test_repeated_boundary_tokens_and_unicode_are_not_deduplicated(self):
  p=copy.deepcopy(RECORD['passages']);p[0]['cohere_raw_text']='  go go café 👩🏽‍💻  ';p[1]['cohere_raw_text']='go go 12.5 '
  self.assertEqual(routed_short_revision(p,0,99968)['text'],'  go go café 👩🏽‍💻   go go 12.5 ')
 def test_nonzero_origin_retains_physical_slice_and_probe_coordinates(self):
  p=copy.deepcopy(RECORD['passages']);origin=16000
  for row in p:
   e=row['acoustic_evidence']
   for key in ('start_sample','end_sample','observed_until_sample'):e[key]+=origin
   for key in ('speech_regions','model_negative_regions'):
    for span in e[key]:span['start_sample']+=origin;span['end_sample']+=origin
   for probe in row['language_detection'].get('probes',[]):probe['start_sample']+=origin;probe['end_sample']+=origin
  result=routed_short_revision(p,origin,origin+99968)
  self.assertEqual(result['provenance']['parts'][0]['start_sample'],origin);self.assertEqual(result['provenance']['parts'][-1]['end_sample'],origin+99968)

class RuntimeDiscardRegression(unittest.TestCase):
 setUp=CanonicalTests.setUp
 tearDown=CanonicalTests.tearDown
 def decode_recorded(self):
  original=self.peer.transcribe
  self.peer.transcribe=lambda audio,language,names,overlap=False:copy.deepcopy(RECORDED_BY_LENGTH[len(audio)]) if len(audio) in RECORDED_BY_LENGTH else original(audio,language,names,overlap)
  self.engine.timeline[0]['language']='auto';self.peer.mixed=True
  self.engine.handle({'type':'audio','start_sample':0,'end_sample':99968,'language':'auto','language_epoch':0});self.engine.handle({'type':'stop'})
  return self.engine.canonical.book.snapshot()[-1]
 def test_actual_archived_pair_no_longer_disappears_from_canonical_primary(self):
  row=self.decode_recorded();expected=' '.join(p['cohere_raw_text'] for p in RECORD['passages'])
  self.assertEqual(row['text'],expected);self.assertNotIn('canonical_unresolved',row);self.assertEqual(row['language'],'en');self.assertTrue(row['review']);self.assertTrue(row['language_review'])
  self.assertTrue(any(p['decision']['reason']=='recent_context' for p in row['language_detection']['probes']))
  self.assertFalse(row['voice_eligible']);self.assertIsNone(row['bounded_decode_provenance']['word_timing'])
 def test_human_edit_and_stale_cas_still_protect_primary(self):
  row=self.decode_recorded();book=self.engine.canonical.book;identity=row['id'];revision=row['machine_revision']
  book.edit(identity,revision,'My exact human correction')
  self.assertNotIn('bounded_decode_provenance',book.rows[identity])
  current=book.rows[identity];self.engine.canonical.decode(current,'refined')
  self.assertNotIn('bounded_decode_provenance',book.rows[identity])
  self.assertEqual(book.rows[identity]['text'],'My exact human correction');self.assertIn('text',book.rows[identity]['protected_fields'])
  with self.assertRaises(ValueError):book.edit(identity,revision,'stale correction')
 def test_routed_text_does_not_use_first_piece_as_whole_alignment_evidence(self):
  self.peer.alignment_supported=lambda language:True
  self.peer.align_canonical=lambda *args,**kwargs:self.fail('Joined routes require separate timing qualification')
  row=self.decode_recorded()
  self.assertEqual(row['text'],' '.join(p['cohere_raw_text'] for p in RECORD['passages']))
  self.assertIsNone(row['bounded_decode_provenance']['word_timing'])
if __name__=='__main__':unittest.main()
