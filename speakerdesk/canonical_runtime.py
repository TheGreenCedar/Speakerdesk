"""Resident canonical utterances, independent of NVIDIA's activity boundaries.

The short utterance path replaces one complete same-anchor Cohere revision.
Long utterances retain bounded context decodes as unresolved candidates until
the configured alignment provider supplies calibrated ownership evidence.
"""
import copy
from pathlib import Path
import uuid

from utterances import UtteranceBook, RevisionArchive, MAX_DECODE_SAMPLES, RATE


def activity_regions(turns, start, end, *, observed=None):
    points={start,end}
    if observed is not None and start<observed<end:points.add(observed)
    for turn in turns:
        a,b=max(start,round(turn['start']*RATE)),min(end,round(turn['end']*RATE))
        if a<b:points.update((a,b))
    points=sorted(points)
    return [{'start_sample':a,'end_sample':b,'speakers':sorted({turn['speaker'] for turn in turns
             if round(turn['start']*RATE)<b and round(turn['end']*RATE)>a
             and (observed is None or b<=observed)})}
            for a,b in zip(points,points[1:])]


class CanonicalRuntime:
    def __init__(self,engine):
        self.engine=engine
        self.archive=RevisionArchive(Path(engine.path).parent/f'utterance-versions-{uuid.uuid4().hex}.jsonl')
        self.book=UtteranceBook(engine.config['job_id'],history_limit=2,archive=self.archive)
        self.last_decoded={};self.published={}

    def observe(self,available):
        ledger=self.engine.models.speech_live.evidence
        while self.book.cursor<available:
            boundary=next((x for x in self.engine.timeline if x['epoch']>self.book.epoch),None)
            if boundary and boundary['start_sample']==self.book.cursor:
                self.book.boundary(self.book.cursor,boundary['epoch'])
                continue
            end=min(available,boundary['start_sample']) if boundary else available
            self.book.observe(ledger.admission(self.book.cursor,end))
        # A boundary with no following samples still seals the earlier object.
        for boundary in self.engine.timeline:
            if boundary['epoch']==self.book.epoch+1 and boundary['start_sample']==self.book.cursor:
                self.book.boundary(self.book.cursor,boundary['epoch'])

    def decode(self,row,stage):
        from live_refinement import read_audio
        e=self.engine;stamp=e.language_at(row['start_sample'])
        a,b=row['start_sample'],row['end_sample']
        names=row.get('speaker_candidates',[])
        if b-a<=MAX_DECODE_SAMPLES:
            passages=e.decode(read_audio(e.path,a,b),stamp['language'],names,a/RATE,row['language_epoch'],overlap=True)
            # Multiple language-routed pieces have no safe shared word ownership.
            complete=(len(passages)==1 and not passages[0].get('transcription_review')
                      and round(passages[0]['start']*RATE)==0
                      and round(passages[0]['end']*RATE)==b-a)
            text=passages[0].get('cohere_raw_text',passages[0]['text']) if len(passages)==1 else ''
            self.archive.append({'type':'cohere_passages','utterance_id':row['id'],
                'audio_revision':row['audio_revision'],'machine_revision':row['machine_revision'],
                'audio_anchor':{'start_sample':a,'end_sample':b},'passages':passages})
            self.book.apply_model(row['id'],row['machine_revision'],text,
                start_sample=a,end_sample=b,stage=stage,complete=complete)
            updated=self.book.rows[row['id']]
            if len(passages)==1:
                for key in ('language','language_detection','acoustic_evidence','transcription_review','audio_state'):
                    if key in passages[0]:updated[key]=copy.deepcopy(passages[0][key])
                    else:updated.pop(key,None)
            if not complete or not text.strip():updated['canonical_unresolved']='incomplete_cohere_revision'
            else:updated.pop('canonical_unresolved',None)
        else:
            parts=[]
            for request in self.book.decode_requests(row['id']):
                if stage=='refined' and e.inbox.cancelled_request(self.refinement_request):return False
                x,y=request['start_sample'],request['end_sample']
                pcm=read_audio(e.path,x,y)
                try:
                    if stage=='refined' and hasattr(e.models,'begin_refinement'):e.models.begin_refinement(pcm,x)
                    passages=e.decode(pcm,stamp['language'],names,x/RATE,row['language_epoch'],overlap=True)
                finally:
                    if stage=='refined' and hasattr(e.models,'end_refinement'):e.models.end_refinement()
                text=passages[0].get('cohere_raw_text',passages[0]['text']) if len(passages)==1 else ''
                complete=len(passages)==1 and not passages[0].get('transcription_review')
                alignment=None
                if complete and text.strip() and hasattr(e.models,'align_canonical'):
                    # Provider must independently enforce its approved clock/score
                    # policy and return the exact raw-text/revision-bound result.
                    alignment=e.models.align_canonical(request,text,language=passages[0].get('language'))
                parts.append({'request':request,'text':text,'complete':complete,'alignment':alignment,
                              'passages':passages})
            self.book.apply_decode_parts(row['id'],parts,stage=stage)
            updated=self.book.rows[row['id']]
            last=updated['machine_versions'][-1]
            if not last['complete']:updated['canonical_unresolved']=last['reason']
            else:updated.pop('canonical_unresolved',None)
        self.last_decoded[row['id']]=(b,row['audio_revision'],stage)
        return True

    def refine(self,request):
        from live_refinement import read_audio,align_tracks
        e=self.engine
        result={'type':'refinement_result',**{key:request[key] for key in ('operation_id','language_epoch','window')}}
        if e.inbox.cancelled_request(request):return {**result,'cancelled':True}
        supplied=request['canonical'];identity=supplied['id']
        current=self.book.rows.get(identity)
        if current and (current['audio_revision']!=supplied['audio_revision']
                or current['machine_revision']!=supplied['canonical_machine_revision']):
            return {**result,'cancelled':True}
        previous=copy.deepcopy(current)
        previous_decode=self.last_decoded.get(identity)
        def restore():
            if previous is None:
                self.book.rows.pop(identity,None)
                self.book.order=[key for key in self.book.order if key!=identity]
            else:self.book.rows[identity]=previous
            if previous_decode is None:self.last_decoded.pop(identity,None)
            else:self.last_decoded[identity]=previous_decode
        if current is None:
            current=copy.deepcopy(supplied)
            current.update(machine_revision=supplied['canonical_machine_revision'],state='sealed',
                machine_versions=[],protected_fields=[],last_speech_sample=supplied['end_sample'])
            self.book.rows[identity]=current;self.book.order.append(identity)
        self.refinement_request=request
        a,b=current['start_sample'],current['end_sample']
        try:
            if b-a<=MAX_DECODE_SAMPLES:
                pcm=read_audio(e.path,a,b)
                if hasattr(e.models,'begin_refinement'):e.models.begin_refinement(pcm,a)
                absolute=[dict(turn,start=turn['start']+a/RATE,end=turn['end']+a/RATE)
                          for turn in e.models.batch_turns(pcm)]
                mapping=align_tracks(absolute,request['references'])
                mapped=[dict(turn,speaker=mapping.get(turn['speaker'],'unknown_mixed')) for turn in absolute]
                self.book.attach_activity(identity,current['audio_revision'],activity_regions(mapped,a,b))
            if not self.decode(current,'refined') or e.inbox.cancelled_request(request):
                restore();return {**result,'cancelled':True}
            candidate=self.project(self.book.rows[identity])
            self.archive.append({'type':'canonical_refinement_publication','candidate':candidate})
            self.published[identity]=copy.deepcopy(candidate)
            e.fast_sequence+=1
            return {**result,'canonical_candidate':candidate,'fast_sequence':e.fast_sequence}
        except Exception:
            restore();raise
        finally:
            if hasattr(e.models,'end_refinement'):e.models.end_refinement()

    def project(self,row):
        e=self.engine;stamp=e.language_at(row['start_sample']);names=row.get('speaker_candidates',[])
        speaker=names[0] if len(names)==1 else 'overlap_'+'_'.join(names) if names else 'unassigned'
        result={key:copy.deepcopy(value) for key,value in row.items()
                if key not in ('machine_versions','protected_fields','last_speech_sample','state','machine_revision')}
        result.update(canonical_utterance_id=row['id'],canonical_machine_revision=row['machine_revision'],
            canonical_state=row['state'],start=row['start_sample']/RATE,end=row['end_sample']/RATE,
            speaker=speaker,source_speaker_candidates=names,language_generation=row['language_epoch'],
            language_mode=stamp['language'],finalized=row['state']=='sealed',
            review=len(names)!=1 or bool(row.get('canonical_unresolved')) or bool(row.get('transcription_review')),
            timing='canonical_vad_audio_anchor',refinement_state=row.get('refinement_state','provisional'))
        if stamp['language']!='auto':result['language']=stamp['language']
        if row.get('canonical_unresolved'):
            result.update(refinement_state='unresolved',voice_eligible=False,
                transcription_review={'reason':'canonical_ownership_unresolved','partial_text':bool(row['text'].strip())})
        return result

    def publish(self,row):
        candidate=self.project(row)
        if self.published.get(row['id'])==candidate:return
        self.archive.append({'type':'canonical_publication','candidate':candidate})
        self.published[row['id']]=copy.deepcopy(candidate)
        e=self.engine;e.fast_sequence+=1
        e.emit({'type':'canonical_revision','candidate':candidate,'fast_sequence':e.fast_sequence})

    def commit(self,final=False):
        e=self.engine;available=min(e.received,e.models.speech_live.evidence.end_sample)
        self.observe(available)
        if e.capture_finished and available==e.received and not self.book.closed:self.book.finish()
        for identity in self.book.order:
            row=self.book.rows[identity]
            previous=self.published.get(identity)
            if (row['state']=='sealed' and previous and previous['canonical_state']=='sealed'
                    and previous['audio_revision']==row['audio_revision']):continue
            self.book.attach_activity(identity,row['audio_revision'],activity_regions(e.turns,row['start_sample'],row['end_sample'],
                observed=min(e.received,getattr(e.models,'nvidia_observed_sample',round(e.processed*RATE)))))
            last=self.last_decoded.get(identity)
            changed=last is None or last[1]!=row['audio_revision']
            # Keep the existing6s first/3s subsequent cadence for growing audio.
            due=final or row['state']=='sealed' or (row['end_sample']-row['start_sample']>=6*RATE
                and (last is None or row['end_sample']-last[0]>=3*RATE))
            if row['end_sample']-row['start_sample']>MAX_DECODE_SAMPLES:
                row['canonical_unresolved']='unresolved_alignment'
            # Uncalibrated long open audio cannot be repeatedly decoded/stitched.
            if changed and due and (row['end_sample']-row['start_sample']<=MAX_DECODE_SAMPLES or row['state']=='sealed' or final):
                self.decode(row,'live')
            self.publish(row)
        e.cursor=self.book.cursor/RATE
        # Published sealed objects remain durable in the journal and host store.
        # Keep the latest object for left-context ownership when speech resumes.
        latest=self.book.order[-1] if self.book.order else None
        retire=[identity for identity in self.book.order if identity!=latest
                and self.book.rows[identity]['state']=='sealed'
                and self.book.rows[identity]['end_sample']<available-60*RATE]
        for identity in retire:
            self.book.rows.pop(identity);self.published.pop(identity,None);self.last_decoded.pop(identity,None)
        if retire:self.book.order=[identity for identity in self.book.order if identity not in set(retire)]
        e.turns=[turn for turn in e.turns if round(turn['end']*RATE)>max(0,self.book.cursor-60*RATE)
                 or (self.book.active and round(turn['end']*RATE)>self.book.rows[self.book.active]['start_sample'])]
