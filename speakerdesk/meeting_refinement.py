"""Atomic host reconciliation and resumable refinement of the existing local WAV."""
import copy
import time
from rolling_refinement import RATE, CORE_SAMPLES, RollingPlan, anchor_id, bounds, reconcile_window, segment_version, split_same_origin
from transcript import validate
from live_language import validate_language_segment
from language_detection import language_probe_events
from utterances import complete_non_speech


def activity_references(sources, start_sample, end_sample):
    """Raw activity only; an ASR envelope does not claim its internal gaps."""
    references=[]
    for row in sources:
        if row.get('canonical_utterance_id'):
            activity=row.get('speaker_activity') or {}
            parts=activity.get('regions',[])
            cursor=row.get('start_sample');valid=(type(cursor) is int
                and type(activity.get('audio_revision')) is int
                and activity['audio_revision']==row.get('audio_revision') and isinstance(parts,list))
            for part in parts if isinstance(parts,list) else []:
                a,b=part.get('start_sample'),part.get('end_sample');names=part.get('speakers')
                if (type(a) is not int or type(b) is not int or a!=cursor or not a<b
                        or not isinstance(names,list) or any(not isinstance(name,str) or not name for name in names)
                        or len(names)!=len(set(names))):valid=False;break
                cursor=b
            if not valid or cursor!=row.get('end_sample'):continue
            for part in parts:
                a=max(part['start_sample'],row['start_sample'],start_sample)
                b=min(part['end_sample'],row['end_sample'],end_sample)
                if a<b:references.append({'start':a/RATE,'end':b/RATE,'speaker_candidates':list(part['speakers'])})
            continue
        parts=row.get('activity_regions')
        if parts:
            for part in parts:
                a=max(part['start'],row.get('source_start',row['start']),start_sample/RATE)
                b=min(part['end'],row.get('source_end',row['end']),end_sample/RATE)
                if a<b:references.append({'start':a,'end':b,'speaker_candidates':list(part['speakers'])})
        elif row['start']<end_sample/RATE and row['end']>start_sample/RATE:
            references.append({'start':row.get('source_start',row['start']),'end':row.get('source_end',row['end']),
                'speaker_candidates':row.get('source_speaker_candidates',row.get('speaker_candidates',[row['speaker']]))})
    return references


def preceding_language_context(sources,start_sample,epoch):
    """Replay preceding observations without extending failed-ASR context."""
    context=None
    preceding=sorted((s for s in sources if s.get('language_epoch')==epoch
        and round(s['end']*RATE)<=start_sample),key=lambda s:s['end'])
    for row in preceding:
        for event in language_probe_events(row):
            reason=event['reason']
            if reason=='unsupported':context=None
            elif reason=='detected':
                if event['language'] and row.get('text','').strip() and not row.get('transcription_review'):
                    context={'language':event['language'],'end_sample':event['end_sample']}
                elif context and context['language']!=event['language']:context=None
    return context if context and 0<=start_sample-context['end_sample']<=60*RATE else None


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
            if job.get('canonical_utterances'):
                # Canonical objects cannot be dispatched through disjoint18s
                # row-containment windows. Their whole anchor is the CAS unit.
                return self.schedule_canonical(jid,job,plan)
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
                remember_unresolved(job,dispatched['start_sample'],dispatched['end_sample'],'language_changed')
                plan.acknowledge(dispatched['id'],dispatched['operation_id'])
                dispatched=plan.dispatch(0 if self.force or force else backlog)
            if dispatched:
                # Blank audio placeholders have no words to align. Split only these
                # at ownership boundaries so recovery can fill a long missed region.
                split_blank_placeholders(job['document'],dispatched)
                expected={s['id']:segment_version(s) for s in job['document']['segments']
                          if s['start']<dispatched['end_sample']/RATE and s['end']>dispatched['start_sample']/RATE}
                references=activity_references(sources,dispatched['context_start_sample'],dispatched['context_end_sample'])
                # Speaker mapping intentionally retains raw removed-row history.
                # Language routing follows current machine rows instead, so an
                # obsolete split cannot contradict a newer merged/refined row.
                language_sources=[job['rolling_sources'].get(s['id'],s) if s.get('protected_fields') else s
                    for s in job['document']['segments']]
                request={'type':'refine','window':dispatched,'operation_id':dispatched['operation_id'],
                         'language_epoch':stamp['epoch'],'language':stamp['language'],
                         'language_start_sample':stamp['start_sample'],
                         'language_context':preceding_language_context(language_sources,dispatched['start_sample'],stamp['epoch']),
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
    def canonical(self,jid,result):
        import hashlib
        with self.manager.lock:
            job=self.manager.get(jid);self.initialize(job)
            sequence=result.get('fast_sequence')
            if type(sequence) is not int or sequence<=0:raise ValueError('Invalid canonical sequence.')
            if sequence<=job.get('last_fast_sequence',0):return
            row=copy.deepcopy(result['candidate']);sid=row['id']
            for key in ('start_sample','end_sample','audio_revision','canonical_machine_revision','language_epoch'):
                if type(row.get(key)) is not int or row[key]<0:raise ValueError('Invalid canonical revision.')
            expected='utterance-'+hashlib.sha256(f"{jid}:{row['language_epoch']}:{row['start_sample']}".encode()).hexdigest()[:24]
            if (sid!=expected or row.get('canonical_utterance_id')!=sid or
                    row.get('canonical_state') not in ('open','sealed') or
                    (round(row['start']*RATE),round(row['end']*RATE))!=(row['start_sample'],row['end_sample'])):
                raise ValueError('Invalid canonical identity or audio anchor.')
            validate_language_segment(row,job['language_history'])
            source=job['rolling_sources'].get(sid)
            if source and (row['audio_revision']<source['audio_revision']
                    or row['canonical_machine_revision']<source['canonical_machine_revision']
                    or (row['canonical_machine_revision']==source['canonical_machine_revision'] and row['text']!=source['text'])
                    or (row['audio_revision']==source['audio_revision'] and
                        row['end_sample']!=source['end_sample'])):
                raise ValueError('Stale canonical output.')
            prior=next((s for s in job['document']['segments'] if s['id']==sid),None)
            job['rolling_sources'][sid]=copy.deepcopy(row)
            job['last_fast_sequence']=sequence
            if source and prior is None:
                # A removed saved passage stays removed; raw machine revisions
                # remain in the worker's durable journal and source ledger.
                self.manager.put(job);self.schedule(jid);return
            if prior:
                protected=set(prior.get('protected_fields',[]))
                row['protected_fields']=sorted(protected)
                row['machine_revision']=prior.get('machine_revision',0)+(row!=source)
                for key in protected & {'text','speaker','start','end'}:row[key]=prior[key]
                if 'text' in protected:
                    row.pop('text_audio_anchor',None)
                    if prior.get('text_audio_anchor'):row['text_audio_anchor']=copy.deepcopy(prior['text_audio_anchor'])
                if protected:
                    row['refinement_state']='edited';row['voice_eligible']=False
                    row.pop('alignment',None)
                    row.pop('assembly_provenance',None)
                    row.pop('bounded_decode_provenance',None);row.pop('reading_turns',None);row.pop('reading_turn_provenance',None)
                    if protected & {'start','end'}:row.pop('text_audio_anchor',None)
                negative=(source is not None and
                    not protected & {'text','start','end'} and
                    row['canonical_machine_revision']>source['canonical_machine_revision'] and
                    row['audio_revision']==source['audio_revision'] and
                    complete_non_speech(row,row['start_sample'],row['end_sample']))
                if negative and prior['text'].strip():
                    save_history(job['refinement_history'],sid,{'window_id':sid,
                        'segments':[copy.deepcopy(prior)],'candidates':[copy.deepcopy(row)],
                        'reason':'complete_canonical_model_non_speech','language_epoch':row['language_epoch'],
                        'created':time.time()})
                if not row['text'].strip() and prior['text'].strip() and not negative:
                    row['text']=prior['text'];row['review']=True;row.pop('alignment',None);row.pop('assembly_provenance',None)
                    row.pop('bounded_decode_provenance',None);row.pop('reading_turns',None);row.pop('reading_turn_provenance',None)
                    row.pop('text_audio_anchor',None)
                    if prior.get('text_audio_anchor') and not protected & {'start','end'}:
                        row['text_audio_anchor']=copy.deepcopy(prior['text_audio_anchor'])
                job['document']['segments']=[row if s['id']==sid else s for s in job['document']['segments']]
            else:
                row['machine_revision']=0;job['document']['segments'].append(row)
            if row.get('reading_turn_provenance'):
                row['reading_turn_provenance']['machine_revision']=row['machine_revision']
            names=result.get('speakers') or {}
            for name in {row['speaker'],*row.get('speaker_candidates',[])}:
                job['document']['speakers'].setdefault(name,names.get(name,
                    'Speaker '+str(int(name.split('_')[-1])+1) if name.startswith('speaker_') else
                    'Unknown speaker' if name=='unassigned' else 'Multiple speakers' if name=='multiple_speakers' else 'Mixed audio'))
            job['duration']=max(job.get('duration',0),self.manager.duration)
            job['document']=validate(job['document'],job['duration'])
            job['document']['provenance'].update(kind='local_inference',mode='canonical_vad_utterances',
                timing='Original VAD audio anchors; NVIDIA activity is independent. No word timestamps without calibrated alignment.')
            job['revision']+=1;self.manager.put(job)
        self.schedule(jid)

    def schedule_canonical(self,jid,job,plan):
        if job.get('rolling_inflight'):return
        if plan.state['cancelled']:
            job['refinement_status']='paused'
            if self.final and not self.shutdown_sent:self.shutdown_sent=self.send({'type':'shutdown'})
            self.manager.put(job);return
        done=job.setdefault('canonical_refined',{})
        sources=sorted(job['rolling_sources'].values(),key=lambda row:row['start_sample'])
        row=next((row for row in sources if row.get('canonical_state')=='sealed'
            and done.get(row['id'])!=row['audio_revision']),None)
        if row is not None:
            import uuid
            current=next((s for s in job['document']['segments'] if s['id']==row['id']),None)
            if current is None:
                done[row['id']]=row['audio_revision'];self.manager.put(job);return self.schedule(jid)
            request={'type':'refine','canonical':copy.deepcopy(row),'operation_id':uuid.uuid4().hex,
                'language_epoch':row['language_epoch'],'language':row['language_mode'],
                'window':{'id':row['id'],'start_sample':row['start_sample'],'end_sample':row['end_sample'],
                    'context_start_sample':row['start_sample'],'context_end_sample':row['end_sample']},
                'expected':{row['id']:segment_version(current)},'references':activity_references(
                    sources,row['start_sample'],row['end_sample'])}
            if self.send(request):job['rolling_inflight']=request;job['refinement_status']='refining'
        elif self.final:
            unresolved=(any(row.get('canonical_unresolved') or row.get('canonical_state')!='sealed' for row in sources)
                or job.get('canonical_observed_sample')!=round(job.get('duration',0)*RATE)
                or bool(job.get('canonical_uncertain_samples',0)))
            job['refinement_status']='unresolved' if unresolved else 'complete'
            if not unresolved:
                plan.state['completed_sample']=round(job.get('duration',0)*RATE)
                job['rolling_refinement']=plan.snapshot()
            if not self.shutdown_sent:self.shutdown_sent=self.send({'type':'shutdown'})
        self.manager.put(job)
    def provisional(self,jid,result):
        with self.manager.lock:
            job=self.manager.get(jid);self.initialize(job)
            sequence=result.get('fast_sequence',0)
            if sequence and sequence<=job.get('last_fast_sequence',0):return
            epoch=next((x for x in job['language_history'] if x['epoch']==result['language_epoch']),None)
            if epoch is None:raise ValueError('Fast output has an unknown language epoch.')
            index=job['language_history'].index(epoch);window=result['window']
            if (window['start_sample']<epoch['start_sample'] or
                    (index+1<len(job['language_history']) and window['end_sample']>job['language_history'][index+1]['start_sample'])):
                raise ValueError('Fast output crosses its captured language boundary.')
            for row in result['candidates']:
                if row.get('language_generation')!=epoch['generation'] or row.get('language_epoch')!=epoch['epoch']:
                    raise ValueError('Fast output does not match its captured language epoch.')
                validate_language_segment(row,job['language_history'])
            candidates,expected=prepare_fast_candidates(job,result)
            # An old epoch can finish its same-origin phrase after the setting
            # changes. It cannot authorize replacing rows from another epoch.
            expected={sid:version for sid,version in expected.items()
                      if any(s['id']==sid and s.get('language_epoch')==epoch['epoch']
                             for s in job['document']['segments'])}
            merged=reconcile_window(job['document'],result['window'],expected,candidates,result.get('speakers'))
            job['duration']=max(job.get('duration',0),self.manager.duration)
            document=validate(merged['document'],job['duration'])
            if merged['changed']:
                job['document']=document;job['document']['provenance']['kind']='local_inference';job['revision']+=1
            for s in document['segments']:
                if result['window']['start_sample']<=bounds(s)[0]<bounds(s)[1]<=result['window']['end_sample'] and not s.get('protected_fields'):
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
            if job.get('canonical_utterances'):return self.canonical_result(jid,job,result)
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
                incomplete=False
                for row in document['segments']:
                    if row.get('refinement_window')==window['id'] and row.get('refinement_state')=='unresolved' and not row.get('protected_fields'):
                        incomplete=True;row['review']=True
                    if (row['id'] in merged['protected_ids'] and not row.get('protected_fields')
                            and request['expected'].get(row['id'])==segment_version(row)
                            and window['start_sample']<=bounds(row)[0]<bounds(row)[1]<=window['end_sample']):
                        incomplete=True
                        gaps=merged['coverage_gaps'].get(row['id'],[])
                        row.update(refinement_state='unresolved',finalized=True,review=True,
                                   transcription_review={'reason':'refinement_incomplete' if gaps else 'refinement_conflict',
                                       'partial_text':bool(row['text'].strip()),'uncovered_audio':gaps})
                save_history(job['refinement_history'],window['id'],{'language_epoch':result['language_epoch'],'created':time.time(),
                    **merged['previous_revision'],'candidates':result.get('candidates',[])})
                job.update(document=document,revision=job['revision']+1)
                plan.acknowledge(window['id'],operation);job['refinement_status']='waiting'
                resolve_unresolved(job,window['start_sample'],window['end_sample'])
                if incomplete:remember_unresolved(job,window['start_sample'],window['end_sample'],'candidate_incomplete')
                job.pop('refinement_error',None)
            job.pop('rolling_inflight',None);self.store(job,plan)
            if self.manager.recognizer:self.manager.recognizer.observe(jid)
        self.schedule(jid)
    def canonical_result(self,jid,job,result):
        request=job.get('rolling_inflight') or {}
        if result.get('operation_id')!=request.get('operation_id') or not request:return
        if result.get('window')!=request['window'] or result.get('language_epoch')!=request['language_epoch']:
            raise ValueError('Canonical refinement does not match its dispatch.')
        row=request['canonical'];candidate=result.get('canonical_candidate')
        job.pop('rolling_inflight',None)
        # One failed attempt remains explicit instead of an infinite retry loop.
        job.setdefault('canonical_refined',{})[row['id']]=row['audio_revision']
        if result.get('error') or result.get('cancelled') or candidate is None:
            source=job['rolling_sources'].get(row['id'])
            if source:source['canonical_unresolved']='refinement_incomplete'
            job['refinement_error']=result.get('error','Refinement incomplete; previous words and audio retained.')
            job['refinement_status']='unresolved';self.manager.put(job);self.schedule(jid);return
        source=job['rolling_sources'].get(row['id'])
        if not source or source['audio_revision']!=row['audio_revision']:
            self.manager.put(job);self.schedule(jid);return
        if (candidate.get('id')!=row['id'] or candidate.get('audio_revision')!=row['audio_revision']
                or candidate.get('start_sample')!=row['start_sample'] or candidate.get('end_sample')!=row['end_sample']):
            raise ValueError('Canonical refinement changed its original anchor.')
        self.manager.put(job)
        self.canonical(jid,{'candidate':candidate,'fast_sequence':result['fast_sequence']})
    def capture_done(self,jid,*,observed_sample=None,uncertain_samples=None,admission=None):
        if observed_sample is not None:
            with self.manager.lock:
                job=self.manager.get(jid)
                if job.get('admission_execution') and not self.manager.refining_saved:
                    from admission_receipt import validate_receipt,retained_pcm_digest
                    request=job.get('capture_inspection_request') or {}
                    received=request.get('through_sample')
                    if received!=round(job.get('duration',0)*RATE):
                        raise ValueError('Unbound canonical Stop inspection endpoint.')
                    job['capture_admission']=validate_receipt(admission,job['admission_execution'],
                        phase='stop',request_id=request.get('request_id'),received_sample=received,
                        observed_sample=observed_sample,uncertain_samples=uncertain_samples,
                        pcm_sha256=retained_pcm_digest(self.manager.folder(jid)/'audio.wav',observed_sample))
                if type(observed_sample) is not int or not 0<=observed_sample<=round(job.get('duration',0)*RATE):
                    raise ValueError('Invalid canonical capture horizon.')
                if uncertain_samples is not None:
                    if type(uncertain_samples) is not int or not 0<=uncertain_samples<=observed_sample:
                        raise ValueError('Invalid canonical uncertain-audio count.')
                    job['canonical_uncertain_samples']=uncertain_samples
                job['canonical_observed_sample']=observed_sample;self.manager.put(job)
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
            if job.get('canonical_utterances'):
                for row in job['rolling_sources'].values():
                    if row.get('canonical_unresolved'):
                        job.setdefault('canonical_refined',{}).pop(row['id'],None)
            if job['refinement_unresolved']:
                plan.state['completed_sample']=min(plan.state['completed_sample'],min(w['start_sample'] for w in job['refinement_unresolved']))
                plan.state['pending']=[]
            job['refinement_status']='waiting';self.store(job,plan)
        self.schedule(jid,force=self.force)


def remember_unresolved(job,start,end,reason):
    """Retain skipped/incomplete audio ranges for an explicit saved-audio retry."""
    if start>=end:return
    # A retry may choose different utterance boundaries. Track coverage rather
    # than assuming its window ID will be identical to the original request.
    resolve_unresolved(job,start,end)
    while start<end:
        stop=min(end,start+CORE_SAMPLES)
        job['refinement_unresolved'].append({'id':anchor_id(start,stop,'refine'),
            'start_sample':start,'end_sample':stop,'reason':reason})
        start=stop
    job['refinement_unresolved'].sort(key=lambda w:w['start_sample'])


def resolve_unresolved(job,start,end):
    retained=[]
    for window in job['refinement_unresolved']:
        a,b=window['start_sample'],window['end_sample']
        if b<=start or a>=end:retained.append(window);continue
        for first,last in ((a,min(b,start)),(max(a,end),b)):
            if first<last:
                piece=copy.deepcopy(window);piece.update(id=anchor_id(first,last,'refine'),start_sample=first,end_sample=last)
                retained.append(piece)
    job['refinement_unresolved']=retained


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
