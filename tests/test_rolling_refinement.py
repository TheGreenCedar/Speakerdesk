"""CPU scheduling/reconciliation contracts; no trained model or capture is executed."""
import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from rolling_refinement import RollingPlan, RATE, reconcile_window, segment_version


def row(sid,start,end,text='original words',speaker='speaker_0',**extra):
    return dict(id=sid,start=start,end=end,text=text,speaker=speaker,machine_revision=0,**extra)


def window(start=0,end=18):
    return {'id':'owned-window','start_sample':int(start*RATE),'end_sample':int(end*RATE)}


class SchedulingTests(unittest.TestCase):
    def test_large_backlog_is_two_requests_and_disjoint_owned_audio(self):
        plan=RollingPlan();segments=[row(str(n),n,n+1) for n in range(120)]
        jobs=plan.schedule(120,segments,100)
        self.assertEqual(len(jobs),2)
        self.assertEqual(jobs[0]['end_sample'],jobs[1]['start_sample'])
        for job in jobs:
            self.assertLessEqual(job['context_end_sample']-job['context_start_sample'],24*RATE)
        self.assertEqual(plan.schedule(10000,segments,200),[])
        self.assertEqual(len(plan.snapshot()['pending']),2)
        self.assertIsNone(plan.dispatch(live_backlog_seconds=3))
    def test_delay_or_three_sections_and_real_boundary(self):
        plan=RollingPlan();segments=[row('a',0,3),row('b',3,6),row('c',6,8)]
        self.assertEqual(plan.schedule(11,segments,7),[])
        jobs=plan.schedule(11,segments,8)
        self.assertEqual(jobs[0]['end_sample'],8*RATE)
        self.assertEqual(jobs[0]['context_end_sample'],11*RATE)
    def test_pause_final_flush_and_resume_have_no_audio_gap(self):
        plan=RollingPlan();a=plan.schedule(7,[row('a',0,7)],10,force=True)[0]
        dispatched=plan.dispatch();self.assertTrue(plan.acknowledge(a['id'],dispatched['operation_id']))
        b=plan.schedule(10,[row('b',7,10)],11,force=True)[0]
        self.assertEqual(b['start_sample'],7*RATE);self.assertEqual(b['end_sample'],10*RATE)
    def test_crash_requeues_same_audio_with_new_operation(self):
        plan=RollingPlan();plan.schedule(10,[row('a',0,10)],10,force=True);first=plan.dispatch()
        restored=RollingPlan(plan.snapshot(),recover=True)
        second=restored.dispatch()
        self.assertEqual(first['id'],second['id']);self.assertNotEqual(first['operation_id'],second['operation_id'])
        self.assertFalse(restored.acknowledge(first['id'],first['operation_id']))
        self.assertTrue(restored.acknowledge(second['id'],second['operation_id']))
    def test_cancelled_late_result_cannot_commit_after_resume(self):
        plan=RollingPlan();segments=[row('a',0,10)]
        plan.schedule(10,segments,10,force=True);old=plan.dispatch();plan.cancel()
        self.assertFalse(plan.acknowledge(old['id'],old['operation_id']))
        plan.resume();plan.schedule(10,segments,11,force=True);new=plan.dispatch()
        self.assertFalse(plan.acknowledge(old['id'],old['operation_id']))
        self.assertTrue(plan.acknowledge(new['id'],new['operation_id']))
    def test_failure_retry_is_bounded_and_retains_durable_backlog_cursor(self):
        plan=RollingPlan();plan.schedule(10,[row('a',0,10)],10,force=True)
        first=plan.dispatch();self.assertEqual(plan.failed(first['id'],first['operation_id']),'retry')
        second=plan.dispatch();self.assertEqual(plan.failed(second['id'],second['operation_id']),'unresolved')
        self.assertEqual(plan.snapshot()['completed_sample'],10*RATE)
    def test_clock_tick_does_not_split_a_crossing_source_row(self):
        plan=RollingPlan();segments=[row('a',0,12),row('b',12,20)]
        jobs=plan.schedule(25,segments,30)
        self.assertEqual(jobs[0]['end_sample'],12*RATE)
        self.assertEqual(len(jobs),1)
        paused=plan.schedule(25,segments,31,force=True)
        self.assertEqual(paused[0]['start_sample'],12*RATE)
    def test_nonfinite_and_invalid_recovered_plan_rejected(self):
        with self.assertRaises(ValueError):RollingPlan().schedule(float('nan'),[],1)
        state=RollingPlan().snapshot();state['completed_sample']=-1
        with self.assertRaises(ValueError):RollingPlan(state)


class ReconciliationTests(unittest.TestCase):
    def setUp(self):
        self.old=[row('a',0,6,'first words'),row('b',6,12,'second words'),row('c',12,18,'third words')]
        self.document={'speakers':{'speaker_0':'Confirmed Albert'},'segments':copy.deepcopy(self.old)}
        self.expected={s['id']:segment_version(s) for s in self.old}
    def apply(self,candidates,document=None,win=None):
        return reconcile_window(document or self.document,win or window(),self.expected,candidates,
                                {'speaker_0':'New guessed name'})
    def test_merge_split_keeps_one_owner_stable_ids_names_and_previous_revision(self):
        result=self.apply([row('ignored',0,12,'larger context words'),row('also ignored',12,18,'tail')])
        rows=result['document']['segments']
        self.assertEqual(len(rows),2);self.assertEqual([s['id'] for s in rows],['a','c'])
        self.assertEqual(result['document']['speakers']['speaker_0'],'Confirmed Albert')
        self.assertEqual(result['previous_revision']['segments'],self.old)
        self.assertEqual(self.document['segments'],self.old)
        self.assertEqual([s['refinement_state'] for s in rows],['refined','refined'])
    def test_late_machine_result_cannot_replace_newer_words(self):
        updated=copy.deepcopy(self.document);updated['segments'][0]['text']='saved correction'
        result=self.apply([row('x',0,6,'late words'),row('y',6,12,'new second'),row('z',12,18,'new third')],updated)
        self.assertEqual(result['document']['segments'][0]['text'],'saved correction')
        self.assertEqual(result['document']['segments'][1]['text'],'new second')
    def test_explicit_user_edit_is_protected_even_if_snapshot_matches(self):
        self.document['segments'][1]['protected_fields']=['text']
        self.expected['b']=segment_version(self.document['segments'][1])
        result=self.apply([row('x',0,6,'new first'),row('y',6,12,'replacement'),row('z',12,18,'new third')])
        self.assertEqual(result['document']['segments'][1]['text'],'second words')
        self.assertEqual(result['protected_ids'],['b'])
    def test_blank_or_partial_output_preserves_every_prior_legible_row(self):
        result=self.apply([row('blank',0,12,''),row('tail',12,18,'new tail')])
        self.assertEqual([s['text'] for s in result['document']['segments']],['first words','second words','new tail'])
        partial=self.apply([row('partial',0,3,'only a fragment')])
        self.assertEqual(partial['document'],self.document)
    def test_shared_candidate_protection_closes_until_no_neighbor_loses_audio(self):
        doc={'speakers':self.document['speakers'],'segments':[row('a',0,9,'protected',protected_fields=['text']),row('b',9,18,'all neighboring words')]}
        expected={s['id']:segment_version(s) for s in doc['segments']}
        result=reconcile_window(doc,window(),expected,[row('x',0,12,'crosses edit'),row('y',12,18,'partial neighbor')])
        self.assertEqual(result['document'],doc)
    def test_context_text_cannot_leak_into_neighbor_and_duplicate_words(self):
        with self.assertRaises(ValueError):self.apply([row('outside',0,21,'includes right context')])
        with self.assertRaises(ValueError):self.apply([row('x',0,12,'one'),row('y',6,18,'duplicated ownership')])
    def test_old_result_is_idempotent_and_cannot_duplicate_rows(self):
        first=self.apply([row('x',0,6,'new first'),row('y',6,12,'new second'),row('z',12,18,'new third')])
        second=self.apply([row('x',0,6,'new first'),row('y',6,12,'new second'),row('z',12,18,'new third')],first['document'])
        self.assertEqual(second['document'],first['document'])
        self.assertEqual(len({s['id'] for s in second['document']['segments']}),3)
    def test_partial_window_cannot_erase_a_straddling_utterance(self):
        result=self.apply([row('x',0,6,'new words')],win=window(0,9))
        self.assertEqual(result['document']['segments'][1],self.old[1])
        self.assertEqual(result['document']['segments'][2],self.old[2])


if __name__=='__main__':unittest.main()
