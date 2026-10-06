"""Physical crop planning contracts; synthetic evidence, no ASR accuracy claim."""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from core_plan import plan,validate,CAP,MIN_CORE
from speech_admission import INPUT_POLICY,SILERO_SPEC
from utterances import UtteranceBook

RATE=16000

def row(seconds=60,gaps=()):
    end=round(seconds*RATE);regions=[];cursor=0
    for a,b in gaps:
        regions.append({'start_sample':cursor,'end_sample':round(a*RATE)})
        cursor=round(b*RATE)
    regions.append({'start_sample':cursor,'end_sample':end})
    return {'id':'u','machine_revision':0,'audio_revision':2,'language_epoch':1,
            'start_sample':0,'end_sample':end,'speech_regions':regions}

def negative(a,b):
    return {'start_sample':a,'end_sample':b,'complete':True,'decision':'no_speech',
            'input_policy':INPUT_POLICY,'model_revision':SILERO_SPEC['revision'],
            'speech_regions':[],'uncertain_regions':[]}

class CorePlanTests(unittest.TestCase):
    def test_all_original_samples_covered_with_bounded_non_tiny_crops(self):
        for samples in (392001,24*RATE+1,48*RATE+1,54*RATE+319,339*RATE,7200*RATE):
            r=row();r['end_sample']=samples;r['speech_regions']=[{'start_sample':0,'end_sample':samples}]
            p=plan(r,lambda a,b:self.fail('No gap should be read'))
            edges=validate(p,r);sizes=[b-a for a,b in zip(edges,edges[1:])]
            self.assertEqual(sum(sizes),samples);self.assertLessEqual(max(sizes),CAP)
            self.assertGreaterEqual(min(sizes),MIN_CORE)
            self.assertLessEqual(max(sizes)-min(sizes),1)
    def test_nearest_current_model_negative_gap_moves_boundary(self):
        r=row(30,[(14,14.4),(16,16.2)]);p=plan(r,negative)
        self.assertEqual(p['edges'],[0,round(14.2*RATE),30*RATE])
        self.assertEqual(p['cuts'][0]['method'],'verified_model_negative_gap')
        self.assertEqual(validate(p,r),p['edges'])
    def test_weak_pending_uncertain_old_receipt_cannot_move_boundary(self):
        changes=[{'complete':False},{'complete':1},{'decision':'speech'},
                 {'input_policy':'old'},{'model_revision':'old'},
                 {'uncertain_regions':[{'start_sample':0,'end_sample':1}]},
                 {'speech_regions':[{'start_sample':0,'end_sample':1}]},
                 {'start_sample':1}]
        for change in changes:
            with self.subTest(change=change):
                p=plan(row(30,[(14,14.4)]),lambda a,b:dict(negative(a,b),**change))
                self.assertEqual(p['edges'],[0,15*RATE,30*RATE])
                self.assertIsNone(p['cuts'][0]['negative_receipt'])
    def test_at_most_three_reads_for_each_boundary_and_deterministic_tie(self):
        calls=[];r=row(30,[(13,13.2),(14,14.2),(16,16.2),(17,17.2)])
        def pending(a,b):calls.append((a,b));return dict(negative(a,b),complete=False)
        p=plan(r,pending);self.assertEqual(len(calls),3)
        self.assertEqual(p['edges'][1],15*RATE)
        tie=row(30,[(14.4,14.6),(15.4,15.6)])
        self.assertEqual(plan(tie,negative)['edges'][1],round(14.5*RATE))
    def test_revision_epoch_anchor_and_boolean_forgery_rejected(self):
        r=row(30,[(14,14.4)]);p=plan(r,negative)
        for key,value in [('machine_revision',1),('audio_revision',3),('language_epoch',2),
                          ('utterance_id','other'),('machine_revision',False)]:
            q=copy.deepcopy(p);q[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):validate(q,r)
        changed=copy.deepcopy(r);changed['end_sample']+=1
        with self.assertRaises(ValueError):validate(p,changed)
        q=copy.deepcopy(p);q['cuts'][0]['negative_receipt']['end_sample']+=1
        with self.assertRaises(ValueError):validate(q,r)
    def test_negative_receipt_must_match_an_actual_internal_speech_gap(self):
        r=row(30,[(14,14.4)]);p=plan(r,negative)
        r['speech_regions']=[{'start_sample':0,'end_sample':30*RATE}]
        with self.assertRaises(ValueError):validate(p,r)
    def test_exact_plan_raw_versions_and_human_protection(self):
        book=UtteranceBook('physical-plan')
        book.observe({'start_sample':0,'end_sample':30*RATE,'complete':True,'decision':'speech',
            'speech_regions':row(30,[(14,14.4)])['speech_regions']})
        identity=book.finish()[0]['id'];r=book.rows[identity]
        r['decode_core_plan']=plan(r,negative);frozen=copy.deepcopy(r['decode_core_plan'])
        requests=book.core_decode_requests(identity)
        parts=[{'request':request,'text':text,'complete':True} for request,text in zip(requests,['Trump Trump',' café 👩🏽‍💻 '])]
        result=book.apply_core_parts(identity,parts,stage='refined')
        self.assertEqual(result['text'],'Trump Trump  café 👩🏽‍💻 ')
        self.assertEqual(result['bounded_decode_provenance']['core_plan'],frozen)
        self.assertNotIn('decode_core_plan',result)
        edited=book.edit(identity,result['machine_revision'],'My words')
        r=book.rows[identity];r['decode_core_plan']=plan(r,negative)
        parts=[dict(p,request=req) for p,req in zip(parts,book.core_decode_requests(identity))]
        result=book.apply_core_parts(identity,parts,stage='refined')
        self.assertEqual(result['text'],'My words')
        self.assertNotIn('bounded_decode_provenance',result)
        self.assertEqual(result['machine_versions'][-1]['text'],'Trump Trump  café 👩🏽‍💻 ')
