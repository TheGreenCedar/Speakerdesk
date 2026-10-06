"""Atomic host reconciliation and resumable refinement of the existing local WAV."""
import copy
import time
from rolling_refinement import RATE, RollingPlan, anchor_id, bounds, reconcile_window, segment_version, split_same_origin
from transcript import validate
from live_language import validate_language_segment


class RefinementController:
    def __init__(self,manager):
        self.manager=manager;self.enabled=False;self.force=False;self.final=False;self.shutdown_sent=False
    def initialize(self,job):
        job.setdefault('language_epoch',0)
        job.setdefault('language_revision',job['language_epoch'])
        job.setdefault('language_acknowledged_revision',0)
        job.setdefault('language_history',[{'epoch':0,'generation':0,'language':job['language'],'start_sample':0}])
        if 'rolling_refinement' not in job:
            plan=RollingPlan();plan.state['last_scheduled_at']=time.monotonic()
            job['rolling_refinement']=plan.snapshot()
        job.setdefault('refinement_status','waiting')
        job.setdefault('refinement_history',{})
        job.setdefault('fast_history',{})
        job.setdefault('refinement_unresolved',[])
        job.setdefault('rolling_sources',{})
    def ready(self,jid,two_pass):
        self.enabled=bool(two_pass);self.force=self.final=self.shutdown_sent=False
        if not self.enabled:return
        with self.manager.lock:
            job=self.manager.get(jid);self.initialize(job);self.manager.put(job)
    def send(self,message):return self.manager.queue_worker(message)
    def store(self,job,plan):
        job['rolling_refinement']=plan.snapshot();self.manager.put(job)
    def schedule(self,jid,force=False):
        if not self.enabled:return
        with self.manager.lock:
            job=self.manager.get(jid);self.initialize(job);plan=RollingPlan(job['rolling_refinement'])
            available=job.get('duration',0) if self.force or force else min(job.get('duration',0),self.manager.processed)
            sources=list(job['rolling_sources'].values())
            settled=[s for s in sources if s.get('finalized') and s['text'].strip()]
            created=plan.schedule(available,settled,time.monotonic(),force=self.force or force)
            # A core cannot cross a captured language/config boundary.
            for window in created:
                boundaries=[x['start_sample'] for x in job['language_history'] if window['start_sample']<x['start_sample']<window['end_sample']]
                if boundaries:
                    actual=next(w for w in plan.state['pending'] if w['id']==window['id'])
                    actual['end_sample']=min(boundaries);actual['context_end_sample']=min(round(available*RATE),actual['end_sample']+3*RATE)
                    # Later windows were planned against the previous endpoint.
                    plan.state['pending']=plan.state['pending'][:plan.state['pending'].index(actual)+1]
                    break
            backlog=max(0,job.get('duration',0)-self.manager.processed)
            dispatched=plan.dispatch(0 if self.force or force else backlog)
            while dispatched:
                stamp=next(x for x in reversed(job['language_history']) if x['start_sample']<=dispatched['start_sample'])
                if stamp['epoch']==job['language_epoch'] or self.manager.refining_saved:break
                plan.acknowledge(dispatched['id'],dispatched['operation_id'])
                dispatched=plan.dispatch(0 if self.force or force else backlog)
            if dispatched:
                # Blank audio placeholders have no words to align. Split only these
                # at ownership boundaries so recovery can fill a long missed region.
                split_blank_placeholders(job['document'],dispatched)
                expected={s['id']:segment_version(s) for s in job['document']['segments']
                          if s['start']<dispatched['end_sample']/RATE and s['end']>dispatched['start_sample']/RATE}
                references=[{'start':s.get('source_start',s['start']),'end':s.get('source_end',s['end']),
                             'speaker_candidates':s.get('source_speaker_candidates',s.get('speaker_candidates',[s['speaker']]))}
                            for s in sources if s['start']<dispatched['context_end_sample']/RATE and s['end']>dispatched['context_start_sample']/RATE]
                request={'type':'refine','window':dispatched,'operation_id':dispatched['operation_id'],
                         'language_epoch':stamp['epoch'],'language':stamp['language'],
                         'expected':expected,'references':references}
                job['rolling_inflight']=copy.deepcopy(request)
                if self.send(request):job['refinement_status']='refining'
                else:plan.failed(dispatched['id'],dispatched['operation_id']);job['refinement_status']='waiting'
            elif plan.state['cancelled']:
                job['refinement_status']='paused'
                if self.final and not self.shutdown_sent:self.shutdown_sent=self.send({'type':'shutdown'})
            elif self.final and not plan.state['pending'] and plan.state['completed_sample']>=round(available*RATE):
                job['refinement_status']='unresolved' if job['refinement_unresolved'] else 'complete'
                if not self.shutdown_sent:self.shutdown_sent=self.send({'type':'shutdown'})
            self.store(job,plan)
    def provisional(self,jid,result):
        with self.manager.lock:
            job=self.manager.get(jid);self.initialize(job)
            sequence=result.get('fast_sequence',0)
            if sequence and sequence<=job.get('last_fast_sequence',0):return
            if result['language_epoch']!=job['language_epoch']:
                # Historical captured epochs may append, but cannot replace newer rows.
                if result.get('expected'):return
            for row in result['candidates']:validate_language_segment(row,job['language_history'])
            candidates,expected=prepare_fast_candidates(job,result)
            merged=reconcile_window(job['document'],result['window'],expected,candidates,result.get('speakers'))
            job['duration']=max(job.get('duration',0),self.manager.duration)
            document=validate(merged['document'],job['duration'])
            if merged['changed']:
                job['document']=document;job['document']['provenance']['kind']='local_inference';job['revision']+=1
            for s in document['segments']:
                if result['window']['start_sample']/RATE<=s['start']<s['end']<=result['window']['end_sample']/RATE and not s.get('protected_fields'):
                    job['rolling_sources'][s['id']]=copy.deepcopy(s)
            if merged['previous_revision']['segments']:
                job['fast_previous_revision']=merged['previous_revision']
                save_history(job['fast_history'],str(result['window']['start_sample']),
                    {'created':time.time(),**merged['previous_revision'],'candidates':copy.deepcopy(result['candidates'])})
            if sequence:job['last_fast_sequence']=sequence
            if merged['protected_ids']:
                job['fast_retained_candidate']=copy.deepcopy(result)
            self.manager.put(job)
        self.schedule(jid)
    def result(self,jid,result):
        with self.manager.lock:
            job=self.manager.get(jid);self.initialize(job);plan=RollingPlan(job['rolling_refinement'])
            operation=result['operation_id'];window=result['window'];request=job.get('rolling_inflight') or {}
            if (request.get('operation_id')!=operation or not plan.state['pending']
                    or plan.state['pending'][0].get('operation_id')!=operation):return
            if window!=request['window']:raise ValueError('Refinement output does not match its dispatched audio window.')
            if (result['language_epoch']!=request['language_epoch']
                    or (result['language_epoch']!=job['language_epoch'] and not self.manager.refining_saved)
                    or result.get('cancelled')):
                plan.failed(window['id'],operation);job['refinement_status']='waiting'
            elif result.get('error'):
                status=plan.failed(window['id'],operation);job['refinement_status']='waiting' if status=='retry' else 'unresolved'
                job['refinement_error']=result['error']
                if status=='unresolved':
                    job['refinement_unresolved'].append(copy.deepcopy(window))
                    previous=[copy.deepcopy(s) for s in job['document']['segments'] if s['start']<window['end_sample']/RATE and s['end']>window['start_sample']/RATE]
                    save_history(job['refinement_history'],window['id'],{'window_id':window['id'],'segments':previous,'candidates':[],
                        'error':result['error'],'language_epoch':result['language_epoch'],'created':time.time()})
                    for row in job['document']['segments']:
                        if row['id'] in request['expected'] and segment_version(row)==request['expected'][row['id']] and not row.get('protected_fields'):
                            row.update(refinement_state='unresolved',refinement_window=window['id'],
                                       transcription_review={'reason':'refinement_incomplete','partial_text':bool(row['text'].strip())})
                    job['revision']+=1
            else:
                for row in result.get('candidates',[]):validate_language_segment(row,job['language_history'])
                merged=reconcile_window(job['document'],window,request['expected'],result.get('candidates',[]),result.get('speakers'))
                document=validate(merged['document'],job['duration'])
                for row in document['segments']:
                    if (row['id'] in merged['protected_ids'] and not row.get('protected_fields')
                            and request['expected'].get(row['id'])==segment_version(row)
                            and window['start_sample']/RATE<=row['start']<row['end']<=window['end_sample']/RATE):
                        row.update(refinement_state='unresolved',finalized=True,
                                   transcription_review={'reason':'refinement_incomplete','partial_text':bool(row['text'].strip())})
                save_history(job['refinement_history'],window['id'],{'language_epoch':result['language_epoch'],'created':time.time(),
                    **merged['previous_revision'],'candidates':result.get('candidates',[])})
                job.update(document=document,revision=job['revision']+1)
                plan.acknowledge(window['id'],operation);job['refinement_status']='waiting'
                job['refinement_unresolved']=[w for w in job['refinement_unresolved'] if w['id']!=window['id']]
                job.pop('refinement_error',None)
            job.pop('rolling_inflight',None);self.store(job,plan)
            if self.manager.recognizer:self.manager.recognizer.observe(jid)
        self.schedule(jid)
    def capture_done(self,jid):
        self.force=self.final=True;self.schedule(jid,force=True)
    def pause(self,jid,final=False):
        with self.manager.lock:
            job=self.manager.get(jid);self.initialize(job);plan=RollingPlan(job['rolling_refinement'])
            request=job.pop('rolling_inflight',None)
            plan.cancel();job['refinement_status']='paused';self.store(job,plan)
            if request:self.send({'type':'cancel_refinement','operation_id':request['operation_id']})
            if final and not self.shutdown_sent:self.shutdown_sent=self.send({'type':'shutdown'})
    def resume(self,jid):
        with self.manager.lock:
            job=self.manager.get(jid);self.initialize(job);plan=RollingPlan(job['rolling_refinement']);plan.resume()
            if job['refinement_unresolved']:
                plan.state['completed_sample']=min(plan.state['completed_sample'],min(w['start_sample'] for w in job['refinement_unresolved']))
                plan.state['pending']=[]
            job['refinement_status']='waiting';self.store(job,plan)
        self.schedule(jid,force=self.force)


def split_blank_placeholders(document,window):
    edges=(window['start_sample']/RATE,window['end_sample']/RATE)
    rows=[]
    for row in document['segments']:
        cuts=[x for x in edges if row['start']<x<row['end']]
        if not cuts or row['text'].strip() or row.get('protected_fields'):
            rows.append(row);continue
        points=[row['start'],*cuts,row['end']]
        for index,(start,end) in enumerate(zip(points,points[1:])):
            piece=copy.deepcopy(row);piece.update(start=start,end=end,source_start=start,source_end=end,
                machine_revision=row.get('machine_revision',0)+1,voice_eligible=False)
            if index:piece['id']=anchor_id(round(start*RATE),round(end*RATE),'placeholder')
            piece['audio_anchor']={'start_sample':round(start*RATE),'end_sample':round(end*RATE)}
            rows.append(piece)
    document['segments']=rows


def save_history(history,key,entry):
    old=history.get(key);versions=list(old.get('versions',[])) if old else []
    if old:versions.append({k:copy.deepcopy(value) for k,value in old.items() if k!='versions'})
    history[key]={**entry,'versions':versions}


def prepare_fast_candidates(job,result):
    """Keep a saved prefix while admitting a provably same-origin lexical tail.

    Monotonic resident sequence numbers allow rebasing unedited provisional rows
    against the host mirror. They never authorize replacing refined/user rows.
    Native source text, rather than edited words, is used for lexical continuity.
    """
    candidates=copy.deepcopy(result['candidates']);expected=dict(result.get('expected',{}))
    if not result.get('fast_sequence'):return candidates,expected
    origin=result['window']['start_sample'];epoch=result['language_epoch']
    for row in job['document']['segments']:
        if (not row.get('protected_fields') and row.get('refinement_state')=='provisional'
                and row.get('fast_origin_sample')==origin and row.get('language_epoch')==epoch):
            expected[row['id']]=segment_version(row)
        if not row.get('protected_fields'):continue
        native=job['rolling_sources'].get(row['id'])
        if not native or native['start']!=row['start'] or native['end']!=row['end']:continue
        for index,candidate in enumerate(candidates):
            if (candidate['start']!=row['start'] or candidate['end']<=row['end']
                    or candidate.get('language_epoch')!=epoch):continue
            split=split_same_origin(native['text'],candidate['text'])
            if split is None:continue
            prefix=copy.deepcopy(candidate);prefix.update(end=row['end'],source_start=row['start'],source_end=row['end'],text=split[0])
            tail=copy.deepcopy(candidate);tail.update(start=row['end'],source_start=row['end'],source_end=candidate['end'],text=split[1])
            candidates[index:index+1]=[prefix,tail];break
    return candidates,expected
