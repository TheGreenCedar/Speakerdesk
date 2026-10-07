"""Versioned verifier policy tests; no models or acoustic correctness claims."""
import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from acceptance_contract_v2 import quiet_cardinal_equivalent,document_slice,SPEECH_REVISION,SPEECH_POLICY
class CardinalTests(unittest.TestCase):
 def test_explicit_cardinal_equivalence_preserves_full_sentence(self):
  reference='The green notebook has eleven pages. Keep it beside the window.'
  self.assertTrue(quiet_cardinal_equivalent(reference,reference.replace('eleven','11')))
  for token in ('12','110','-11','+11','11.0','11,000','11th','−11','- 11','+ 11','11%','11 %','11‰','$11','$ 11','11€','eleven%','€eleven','', 'Tankyou'):
   with self.subTest(token=token):self.assertFalse(quiet_cardinal_equivalent(reference,reference.replace('eleven',token)))
  self.assertFalse(quiet_cardinal_equivalent(reference,reference.replace('notebook','book')))
class ContextTests(unittest.TestCase):
 def setUp(self):
  self.row={'id':'u','canonical_utterance_id':'u','audio_revision':2,'text':'raw French text','start':0.,'end':.01,'text_audio_anchor':{'start_sample':0,'end_sample':160},'speech_regions':[{'start_sample':100,'end_sample':150}]}
  e={'source':'silero_v6','model_revision':SPEECH_REVISION,'input_policy':SPEECH_POLICY,'complete':True,'decision':'speech','start_sample':0,'end_sample':160,'observed_until_sample':160,'uncertain_regions':[],'speech_regions':[{'start_sample':100,'end_sample':150}],'model_negative_regions':[{'start_sample':0,'end_sample':100},{'start_sample':150,'end_sample':160}]}
  self.row['acoustic_evidence']=e;self.provisional={'id':'job','document':{'segments':[copy.deepcopy(self.row)]}};self.final=copy.deepcopy(self.provisional)
  e=self.final['document']['segments'][0]['acoustic_evidence'];e['speech_regions'][0]['start_sample']=70;e['model_negative_regions'][0]['end_sample']=70
  self.pcm=[0]*100+[1]*50+[0]*10
 def test_independent_bound_receipts_accept_zero_prefix_without_trimming(self):
  before=copy.deepcopy(self.final);got=document_slice(self.final,80,self.pcm,self.provisional)
  self.assertEqual(got['document']['segments'],before['document']['segments']);self.assertEqual(self.final,before)
 def test_nonzero_or_stale_or_uncertain_receipts_reject(self):
  variants=[('revision',lambda j:j['document']['segments'][0].update(audio_revision=3)),('anchor',lambda j:j['document']['segments'][0]['text_audio_anchor'].update(start_sample=1)),('uncertain',lambda j:j['document']['segments'][0]['acoustic_evidence'].update(uncertain_regions=[{'start_sample':70,'end_sample':80}])),('incomplete',lambda j:j['document']['segments'][0]['acoustic_evidence'].update(complete=False)),('model',lambda j:j['document']['segments'][0]['acoustic_evidence'].update(model_revision='other')),('gap',lambda j:j['document']['segments'][0]['acoustic_evidence']['model_negative_regions'][0].update(end_sample=69))]
  for name,mutate in variants:
   with self.subTest(name=name):
    bad=copy.deepcopy(self.final);mutate(bad)
    with self.assertRaises(ValueError):document_slice(bad,80,self.pcm,self.provisional)
  self.pcm[70]=1
  with self.assertRaises(ValueError):document_slice(self.final,80,self.pcm,self.provisional)
if __name__=='__main__':unittest.main()
