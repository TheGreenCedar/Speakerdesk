"""CPU scheduling/retained-audio contracts; peers are not acoustic proof."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
import test_canonical_runtime as fixtures
from live_refinement import Inbox
RATE=16000

class LiveSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.case=fixtures.CanonicalTests();self.case.setUp()
        self.engine=self.case.engine;self.peer=self.case.peer
        self.inbox=Inbox(audio_batch_seconds=6,refinement_interval_seconds=30)
        self.engine.inbox=self.inbox
    def tearDown(self):self.case.tearDown()
    def audio(self,a,b,language='en',epoch=0):
        for n in range(a,b):self.inbox.push({'type':'audio','start_sample':n*RATE,
            'end_sample':(n+1)*RATE,'language':language,'language_epoch':epoch})
    def take(self):
        command=self.inbox.take();self.engine.handle(command);return command
    def test_stale_interims_coalesce_but_first_text_and_every_sample_are_retained(self):
        fed=[];original=self.peer.feed
        def feed(pcm,**kw):fed.append(len(pcm));return original(pcm,**kw)
        self.peer.feed=feed;self.audio(0,18)
        self.take();self.assertEqual(self.peer.calls,[])
        self.take();self.assertEqual(len(self.peer.calls),1)
        self.assertGreaterEqual(self.peer.calls[0],6*RATE);self.assertLess(self.peer.calls[0],7*RATE)
        self.take();self.assertEqual(len(self.peer.calls),2)
        self.assertEqual(fed,[RATE]*18)
        self.engine.handle({'type':'stop'});self.assertEqual(self.engine.canonical.book.cursor,18*RATE)
        calls=list(self.peer.calls);self.engine.commit(final=True);self.assertEqual(self.peer.calls,calls)
        row=self.engine.canonical.book.snapshot()[0]
        self.assertEqual(row['state'],'sealed');self.assertEqual(row['text_audio_anchor'],{'start_sample':0,'end_sample':18*RATE})
    def test_current_notifications_keep_existing_six_then_three_second_cadence(self):
        for n in range(12):self.audio(n,n+1);self.take()
        baseline=fixtures.CanonicalTests();baseline.setUp()
        try:
            baseline.feed(0,12);self.assertEqual(self.peer.calls,baseline.peer.calls)
        finally:baseline.tearDown()
    def test_sealed_speech_is_recognized_even_with_newer_audio_queued(self):
        self.peer.silent_after=6*RATE;self.audio(0,20);self.take();self.take()
        self.assertEqual(len(self.peer.calls),1);self.assertTrue(self.inbox.messages)
        row=self.engine.canonical.book.snapshot()[0]
        self.assertEqual(row['state'],'sealed');self.assertGreater(row['end_sample'],6*RATE)
        self.assertLess(row['end_sample'],7*RATE);self.assertTrue(row['text'].strip())
    def test_language_control_is_an_ordered_barrier_not_a_superseding_tail(self):
        self.audio(0,4);self.inbox.push({'type':'language','generation':1,'language':'fr','start_sample':4*RATE})
        self.audio(4,8,'fr',1)
        self.assertEqual(self.take()['end_sample'],4*RATE)
        self.assertFalse(self.inbox.newer_audio_queued(4*RATE,0))
        self.take();self.take();self.engine.handle({'type':'stop'})
        rows=self.engine.canonical.book.snapshot()
        self.assertEqual([(r['start_sample'],r['end_sample'],r['language_epoch']) for r in rows],[(0,4*RATE,0),(4*RATE,8*RATE,1)])
        self.assertEqual([self.engine.canonical.project(r)['language'] for r in rows],['en','fr'])
    def test_pause_barrier_keeps_exact_anchor_and_protected_words_through_stop(self):
        self.audio(0,8);self.inbox.push({'type':'flush','request_id':'pause','through_sample':8*RATE});self.audio(8,10)
        self.take();self.take();self.take()
        ack=self.case.events[-1];self.assertEqual(ack['type'],'flush_ack');self.assertEqual(ack['received_sample'],8*RATE)
        row=self.engine.canonical.book.snapshot()[0]
        self.engine.canonical.book.edit(row['id'],row['machine_revision'],'Human correction retained')
        self.take();self.engine.handle({'type':'stop'})
        latest=self.engine.canonical.book.rows[row['id']]
        self.assertEqual(latest['text'],'Human correction retained');self.assertIn('text',latest['protected_fields'])
        self.assertEqual(latest['end_sample'],10*RATE);self.assertEqual(latest['state'],'sealed')
    def test_background_refinement_runs_periodically_when_current_and_drains_at_stop(self):
        from unittest.mock import patch
        request={'type':'refine','operation_id':'saved','language_epoch':0}
        with patch('live_refinement.time.monotonic',return_value=100):
            self.inbox.push(request);self.audio(0,6)
            self.assertEqual(self.take()['type'],'audio')
            self.assertEqual(self.inbox.take(),request)
        second={**request,'operation_id':'next'};self.inbox.push(second)
        with patch('live_refinement.time.monotonic',return_value=129):self.assertIsNone(self.inbox.take())
        with patch('live_refinement.time.monotonic',return_value=130):self.assertEqual(self.inbox.take(),second)
        self.inbox.push(request);self.inbox.push({'type':'stop'});self.take()
        self.assertEqual(self.inbox.take(),request)
        historical=Inbox(refinement_interval_seconds=30);historical.historical=True
        historical.last_refinement_at=100000000000;historical.push(request)
        self.assertEqual(historical.take(),request)
    def test_cancellation_and_shutdown_still_preempt_pending_refinement(self):
        request={'type':'refine','operation_id':'saved','language_epoch':0}
        self.inbox.push(request);self.inbox.push({'type':'cancel_refinement','operation_id':'saved'})
        self.assertIsNone(self.inbox.refinement);self.assertTrue(self.inbox.cancelled_request(request))
        self.audio(0,10);self.inbox.push({'type':'shutdown'})
        self.assertEqual(self.inbox.take(),{'type':'shutdown'})
    def test_canonical_background_keeps_original_epochs_and_explicit_cancellation(self):
        request={'type':'refine','operation_id':'old-epoch','language_epoch':0,'canonical':{'canonical_state':'sealed'}}
        self.inbox.latest_epoch=1
        self.assertFalse(self.inbox.cancelled_request(request))
        rolling={k:v for k,v in request.items() if k!='canonical'}
        self.assertTrue(self.inbox.cancelled_request(rolling))
        future={**request,'language_epoch':2};self.assertTrue(self.inbox.cancelled_request(future))
        self.inbox.capture_finished=True
        self.assertFalse(self.inbox.cancelled_request(request))
        self.inbox.push({'type':'cancel_refinement','operation_id':'old-epoch'})
        self.assertTrue(self.inbox.cancelled_request(request))
        self.inbox.cancelled.clear();self.inbox.push({'type':'shutdown'})
        self.assertTrue(self.inbox.cancelled_request(request))
    def test_host_stop_refinement_completes_both_original_language_epochs(self):
        import queue
        from app import create_app
        self.audio(0,4);self.inbox.push({'type':'language','generation':1,'language':'fr','start_sample':4*RATE})
        self.audio(4,8,'fr',1)
        self.take();self.take();self.take();self.engine.handle({'type':'stop'})
        app=create_app(self.case.root/'host');manager=app.extensions['speakerdesk']['meetings']
        manager.duration=8
        job={'id':self.case.jid,'created':1,'status':'finishing','kind':'meeting','name':'CPU epochs',
             'language':'fr','language_epoch':1,'duration':8,'revision':0,'canonical_utterances':True,
             'language_history':[{'epoch':0,'generation':0,'language':'en','start_sample':0},
                                 {'epoch':1,'generation':1,'language':'fr','start_sample':4*RATE}],
             'document':{'speakers':{},'segments':[],'provenance':{},'warnings':[]}}
        try:
            manager.refinement.initialize(job);manager.put(job)
            for event in self.case.events:
                if event['type']=='canonical_revision':manager.refinement.canonical(self.case.jid,event)
            manager.refinement.ready(self.case.jid,True)
            manager.refinement.capture_done(self.case.jid,observed_sample=8*RATE,uncertain_samples=0)
            epochs=[]
            for _ in range(8):
                while True:
                    try:self.inbox.push(manager.worker_controls.get_nowait())
                    except queue.Empty:break
                command=self.inbox.take()
                if command['type']=='shutdown':break
                self.assertEqual(command['type'],'refine');epochs.append(command['language_epoch'])
                result=self.engine.refine(command);self.assertNotIn('cancelled',result)
                manager.refinement.result(self.case.jid,result)
            self.assertEqual(epochs,[0,1])
            final=manager.get(self.case.jid);self.assertEqual(final['refinement_status'],'complete')
            self.assertNotIn('refinement_error',final)
            self.assertEqual([(r['start_sample'],r['end_sample'],r['language_epoch'])
                for r in final['document']['segments']],[(0,4*RATE,0),(4*RATE,8*RATE,1)])
        finally:
            manager.close();app.extensions['speakerdesk']['executor'].shutdown(wait=True,cancel_futures=True)
    def test_real_host_background_refines_prior_sealed_epoch_before_stop(self):
        import queue
        from app import create_app
        self.peer.silent_after=6*RATE
        self.audio(0,4);self.inbox.push({'type':'language','generation':1,'language':'fr','start_sample':4*RATE})
        self.audio(4,8,'fr',1);self.take();self.take();self.take()
        app=create_app(self.case.root/'background-host');manager=app.extensions['speakerdesk']['meetings']
        manager.duration=8
        job={'id':self.case.jid,'created':1,'status':'recording','kind':'meeting','name':'CPU background',
             'language':'fr','language_epoch':1,'duration':8,'revision':0,'canonical_utterances':True,
             'language_history':[{'epoch':0,'generation':0,'language':'en','start_sample':0},
                                 {'epoch':1,'generation':1,'language':'fr','start_sample':4*RATE}],
             'document':{'speakers':{},'segments':[],'provenance':{},'warnings':[]}}
        try:
            manager.refinement.initialize(job);manager.put(job)
            for event in self.case.events:
                if event['type']=='canonical_revision':manager.refinement.canonical(self.case.jid,event)
            manager.refinement.ready(self.case.jid,True);manager.refinement.schedule(self.case.jid)
            while True:
                try:self.inbox.push(manager.worker_controls.get_nowait())
                except queue.Empty:break
            request=self.inbox.take();self.assertEqual(request['type'],'refine');self.assertEqual(request['language_epoch'],0)
            self.assertFalse(self.engine.capture_finished)
            result=self.engine.refine(request);self.assertNotIn('cancelled',result)
            manager.refinement.result(self.case.jid,result)
            final=manager.get(self.case.jid)
            self.assertEqual(final['canonical_refined'][request['canonical']['id']],request['canonical']['audio_revision'])
            self.assertNotIn('refinement_error',final)
        finally:
            manager.close();app.extensions['speakerdesk']['executor'].shutdown(wait=True,cancel_futures=True)
    def test_language_change_during_refinement_rolls_back_then_fresh_prior_epoch_runs(self):
        import copy
        self.peer.silent_after=6*RATE;self.audio(0,8);self.take();self.take()
        row=copy.deepcopy(self.case.rows()[-1]);before=copy.deepcopy(self.engine.canonical.book.rows[row['id']])
        request={'type':'refine','canonical':row,'operation_id':'old-active','language_epoch':0,'language':'en',
                 'window':{'id':row['id'],'start_sample':row['start_sample'],'end_sample':row['end_sample']},
                 'references':[{'start':0,'end':8,'speaker_candidates':['speaker_0']}]}
        original=self.peer.transcribe
        def during(*args,**kwargs):
            self.inbox.push({'type':'language','generation':1,'language':'fr','start_sample':8*RATE})
            return original(*args,**kwargs)
        self.peer.transcribe=during;self.inbox.push(request);self.engine.handle(self.inbox.take())
        self.assertTrue(self.case.events[-1]['cancelled']);self.assertEqual(self.engine.canonical.book.rows[row['id']],before)
        self.assertIsNone(self.inbox.active_refinement)
        self.peer.transcribe=original;self.take()
        fresh={**request,'operation_id':'fresh-after-boundary'};self.inbox.push(fresh)
        self.inbox.last_refinement_at=None;self.engine.handle(self.inbox.take())
        self.assertNotIn('cancelled',self.case.events[-1]);self.assertIn('canonical_candidate',self.case.events[-1])
        self.assertIsNone(self.inbox.active_refinement)
    def test_pending_cancellation_flood_cannot_revive_active_cancelled_operation(self):
        request={'type':'refine','operation_id':'active','language_epoch':0,
                 'canonical':{'canonical_state':'sealed'}}
        self.inbox.push(request);self.assertEqual(self.inbox.take(),request)
        self.inbox.push({'type':'language','generation':1,'language':'fr','start_sample':0})
        for n in range(40):
            self.inbox.push({'type':'cancel_refinement','operation_id':f'pending-{n}'})
            self.assertTrue(self.inbox.cancelled_request(request));self.assertLessEqual(len(self.inbox.cancelled),16)
        self.inbox.finish_refinement('active');self.assertIsNone(self.inbox.active_refinement)

if __name__=='__main__':unittest.main()
