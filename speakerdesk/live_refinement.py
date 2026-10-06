"""Resident serial model executor with retained-audio ranges and a provisional tail.

The injectable engine is exercised with CPU peers. Models is the real offline
MLX implementation; it is never replaced in a normal app run.
"""
import contextlib
import copy
from collections import deque
import json
from pathlib import Path
import resource
import threading
import time
import wave

from pipeline import decode_regions, activity_slice, TURN_GAP_SECONDS
from language_detection import language_probe_events
from rolling_refinement import RATE, anchor_id, bounds, reconcile_window, segment_version, split_same_origin


def read_audio(path, start, end):
    import numpy as np
    if not 0 <= start < end or end-start > int(24.5*RATE):raise ValueError('Oversized audio read.')
    with wave.open(str(path),'rb') as source:
        if (source.getnchannels(),source.getsampwidth(),source.getframerate()) != (1,2,RATE):raise ValueError('Invalid saved meeting audio.')
        if end > source.getnframes():raise ValueError('Requested audio is not yet saved.')
        source.setpos(start);data=source.readframes(end-start)
    if len(data)!=(end-start)*2:raise ValueError('Incomplete saved audio.')
    return np.frombuffer(data,dtype='<i2').astype('float32')/32767


class Inbox:
    """Coalesce audio notifications into ranges; PCM stays in the saved WAV."""
    def __init__(self):
        self.condition=threading.Condition();self.messages=deque();self.refinement=None
        self.closed=False;self.shutdown=False;self.error=None;self.historical=False;self.latest_epoch=0;self.cancelled=set();self.received=0
    def push(self,message):
        with self.condition:
            kind=message['type']
            if kind=='audio':
                a,b=message['start_sample'],message['end_sample']
                if type(a) is not int or type(b) is not int or a!=self.received or not a<b or b-a>RATE:raise ValueError('Noncontiguous audio notification.')
                self.received=b
                if (self.messages and self.messages[-1]['type']=='audio'
                        and self.messages[-1]['language_epoch']==message['language_epoch']):
                    self.messages[-1]['end_sample']=b
                else:self.messages.append(copy.deepcopy(message))
            elif kind=='refine':
                if self.refinement is not None:raise ValueError('Only one dispatched refinement is allowed.')
                self.refinement=copy.deepcopy(message)
            elif kind=='cancel_refinement':
                if message.get('operation_id'):self.cancelled.add(message['operation_id'])
                if self.refinement and self.refinement['operation_id']==message.get('operation_id'):
                    self.cancelled.add(self.refinement['operation_id']);self.refinement=None
                self.cancelled=set(list(self.cancelled)[-16:])
            elif kind=='language':
                self.latest_epoch=int(message.get('language_epoch',message.get('generation',0)))
                if self.refinement and self.refinement['language_epoch']!=self.latest_epoch:self.refinement=None
                self.messages.append(copy.deepcopy(message))
            elif kind=='shutdown':self.shutdown=True
            elif kind in ('flush','stop'):self.messages.append(copy.deepcopy(message))
            else:raise ValueError('Unexpected resident worker command.')
            if len(self.messages)>128:raise ValueError('Too many pending language/control boundaries.')
            self.condition.notify_all()
    def cancelled_request(self,request):
        with self.condition:
            return self.shutdown or request['operation_id'] in self.cancelled or (not self.historical and request['language_epoch']!=self.latest_epoch)
    def take(self):
        with self.condition:
            if self.error:raise ValueError(self.error)
            if self.shutdown:return {'type':'shutdown'}
            if self.messages:
                result=copy.deepcopy(self.messages[0])
                if result['type']=='audio':
                    result['end_sample']=min(result['end_sample'],result['start_sample']+RATE)
                    self.messages[0]['start_sample']=result['end_sample']
                    if result['end_sample']==self.messages[0]['end_sample']:self.messages.popleft()
                else:self.messages.popleft()
                return result
            if self.refinement:
                result=self.refinement;self.refinement=None;return result
            if self.closed:return {'type':'shutdown'}
            self.condition.wait(.1);return None


def align_tracks(turns, references, *, diagnostics=None):
    """Temporal correspondence only; ambiguous local slots remain mixed/unknown."""
    local=sorted({t['speaker'] for t in turns});known=sorted({s for r in references for s in r.get('speaker_candidates',[])})
    mapping={};claims={};details={}
    for name in local:
        activity=[t for t in turns if t['speaker']==name];duration=interval_duration([(t['start'],t['end']) for t in activity])
        score={k:interval_duration([(max(t['start'],r['start']),min(t['end'],r['end'])) for t in activity for r in references if k in r.get('speaker_candidates',[])]) for k in known}
        ranked=sorted(score,key=score.get,reverse=True)
        details[name]={'local_speaker':name,'active_seconds':duration,'reference_overlap_seconds':score,
                       'best_speaker':ranked[0] if ranked else None,'best_fraction':0.,'runner_up_fraction':0.,
                       'reason':'no_reference'}
        if ranked and duration>0:
            best=score[ranked[0]]/duration;second=score[ranked[1]]/duration if len(ranked)>1 else 0
            details[name].update(best_fraction=best,runner_up_fraction=second,
                reason='insufficient_overlap' if best<.60 else 'insufficient_margin' if best-second<.20 else 'ambiguous_claim')
            if best>=.60 and best-second>=.20:claims.setdefault(ranked[0],[]).append(name)
    for global_name,names in claims.items():
        if len(names)==1:
            mapping[names[0]]=global_name;details[names[0]]['reason']='mapped'
    if diagnostics is not None:diagnostics.extend(details[name] for name in local)
    return mapping


def interval_duration(intervals):
    total=0.;end=None
    for start,last in sorted((a,b) for a,b in intervals if a<b):
        total+=max(0,last-max(start,end if end is not None else start));end=max(last,end if end is not None else last)
    return total


def coalesce_blanks(rows):
    result=[]
    for row in rows:
        reason=(row.get('language_detection') or {}).get('reason')
        if (result and not row['text'].strip() and not result[-1]['text'].strip()
                and not row.get('activity_regions') and not result[-1].get('activity_regions')
                and not (row.get('language_detection') or {}).get('probes')
                and not (result[-1].get('language_detection') or {}).get('probes')
                and row['speaker']==result[-1]['speaker'] and abs(row['start']-result[-1]['end'])<.02
                and row.get('language_epoch')==result[-1].get('language_epoch')
                and row.get('language')==result[-1].get('language')
                and row.get('transcription_review')==result[-1].get('transcription_review')
                and row.get('audio_state')==result[-1].get('audio_state')
                and reason==(result[-1].get('language_detection') or {}).get('reason')):
            result[-1]['end']=row['end']
        else:result.append(row)
    return result


def context_regions(turns,max_seconds=None,*,coverage=None):
    return decode_regions(turns,max_seconds,coverage=coverage)


class Models:
    def __init__(self,config):
        from inference_worker import check_memory
        check_memory()
        import mlx.core as mx
        from mlx_audio.vad import load
        from mlx_speech.generation.cohere_asr import CohereAsrModel
        self.mx=mx;self.check_memory=check_memory;self.config=config;self.detector=None
        mx.set_memory_limit(5*1024**3);mx.set_cache_limit(256*1024**2)
        self.diar=load(Path(config['diar_path']),strict=True);self.diar.set_streaming_config('low')
        self.state=self.diar.init_streaming_state()
        self.asr=CohereAsrModel.from_path(Path(config['cohere_path']))
        self.language_context=None;self.transcription_start_sample=0
    def set_language_context(self,context,start_sample):
        self.language_context=copy.deepcopy(context);self.transcription_start_sample=start_sample
    def transcribe(self,audio,language,names,overlap=False):
        import sys
        from language_detection import SpeechTranscriber,WhisperLanguageDetector
        self.check_memory()
        if language=='auto' and self.detector is None:self.detector=WhisperLanguageDetector(self.config['lid_path'])
        transcriber=SpeechTranscriber(self.asr,language,self.config.get('lid_path'),detector=self.detector,context=self.language_context)
        with contextlib.redirect_stdout(sys.stderr):
            result=transcriber.transcribe(audio,RATE,tuple(names),max_asr_seconds=24.5,allow_overlap=overlap,
                start_sample=self.transcription_start_sample)
        self.detector=transcriber.detector
        self.mx.clear_cache();return result
    def feed(self,audio,final=False):
        import sys
        with contextlib.redirect_stdout(sys.stderr):
            result,self.state=self.diar.feed(audio,self.state,sample_rate=RATE,final=final,threshold=.5,min_duration=0,merge_gap=0)
        return [{'start':s.start,'end':s.end,'speaker':f'speaker_{s.speaker}'} for s in result.segments],self.state.frames_processed*.01
    def batch_turns(self,audio):
        import sys
        self.check_memory()
        with contextlib.redirect_stdout(sys.stderr):
            # generate creates a separate streaming state. The executor is
            # serial, so temporarily selecting the official offline preset
            # gives this bounded window full context without touching live state.
            self.diar.set_streaming_config('offline')
            try:result=self.diar.generate(audio,sample_rate=RATE,threshold=.5,min_duration=0,merge_gap=0)
            finally:self.diar.set_streaming_config('low')
        return [{'start':s.start,'end':s.end,'speaker':f'speaker_{s.speaker}'} for s in result.segments]
    def metrics(self):
        return {'peak_mlx_bytes':self.mx.get_peak_memory(),'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


class Engine:
    def __init__(self,config,models,emit,inbox):
        self.config=config;self.models=models;self.emit=emit;self.inbox=inbox;self.path=Path(config['audio_path'])
        self.received=0;self.processed=0.;self.turns=[];self.cursor=0.;self.last_decode=0.
        self.timeline=[{'start_sample':0,'language':config['language'],'epoch':config.get('language_epoch',0)}]
        self.document={'speakers':{},'segments':[]};self.capture_finished=False;self.fast_sequence=0
        self.carry=None;self.language_observations=[]
    def language_at(self,sample):
        return next(x for x in reversed(self.timeline) if x['start_sample']<=sample)
    def decode(self,pcm,language,names,start,epoch,*,overlap=False,context_seed=None,observations=None):
        """Seed each decode only with successful preceding speech in this epoch.

        Historical refinement must never inherit the resident worker's future
        language. Failed clear switches also invalidate contradicted context.
        """
        sample=round(start*RATE)
        history=self.language_observations if observations is None else observations
        candidates=[x for x in history if x['epoch']==epoch and x['end_sample']<=sample]
        if context_seed is not None:
            candidates.append({**context_seed,'epoch':epoch})
        context=max(candidates,key=lambda x:x['end_sample']) if candidates else None
        if context and (not context.get('language') or not 0<=sample-context['end_sample']<=60*RATE):context=None
        if hasattr(self.models,'set_language_context'):
            self.models.set_language_context({k:context[k] for k in ('language','end_sample')} if context else None,sample)
        passages=self.models.transcribe(pcm,language,names,overlap=overlap)
        for passage in passages:
            for event in language_probe_events(passage,sample) if language=='auto' else []:
                if event['reason'] not in ('detected','unsupported'):continue
                successful=passage.get('text','').strip() and not passage.get('transcription_review')
                prior=[x for x in history if x['epoch']==epoch
                    and x['end_sample']<=event['start_sample']]
                if context:prior.append(context)
                recent=max(prior,key=lambda x:x['end_sample']) if prior else None
                if (not successful and event['reason']=='detected' and recent
                        and recent.get('language')==event['language']):
                    continue
                observation={'epoch':epoch,'end_sample':event['end_sample'],
                    'language':event['language'] if successful and event['reason']=='detected' else None}
                history[:]=[x for x in history
                    if (x['epoch'],x['end_sample'])!=(epoch,observation['end_sample'])]
                history.append(observation)
        history[:]=sorted(history,key=lambda x:x['end_sample'])[-256:]
        return passages
    def decorate(self,passages,start,end,names,epoch,finalized=False):
        speaker=names[0] if len(names)==1 else 'overlap_'+'_'.join(names) if names else 'unassigned'
        rows=[]
        for p in passages:
            a=max(start,start+p['start']);b=min(end,start+p['end'])
            if b<=a:continue
            rows.append({**p,'start':a,'end':b,'speaker':speaker,'speaker_candidates':list(names),
                         'source_speaker_candidates':list(names),'source_start':a,'source_end':b,
                         'language_epoch':epoch,'confidence':None,'timing':'contextual_audio_regions',
                         'language_generation':epoch,'language_mode':self.language_at(round(a*RATE))['language'],
                         'voice_eligible':finalized and len(names)==1 and not p.get('audio_state'),'finalized':finalized,
                         'review':p.get('review',False) or len(names)!=1,'refinement_state':'provisional'})
        return rows
    def publish(self,start,end,rows,names):
        self.fast_sequence+=1
        for row in rows:
            mode=self.language_at(round(row['start']*RATE))
            if row.get('activity_regions'):
                row['activity_regions']=activity_slice(row['activity_regions'],row['start'],row['end'])
            row.update(source_start=row['start'],source_end=row['end'],fast_origin_sample=round(start*RATE),
                       language_generation=mode['epoch'],language_mode=mode['language'])
            if not row.get('language') and mode['language']!='auto':row['language']=mode['language']
        win={'id':anchor_id(round(start*RATE),round(end*RATE),'fast'), 'kind':'fast_tail',
             'start_sample':round(start*RATE),'end_sample':round(end*RATE)}
        previous=[s for s in self.document['segments'] if s['start']<end and s['end']>start]
        expected={s['id']:segment_version(s) for s in previous}
        speakers={s['speaker']:('Speaker '+str(int(s['speaker'].split('_')[-1])+1) if s['speaker'].startswith('speaker_') else 'Unknown speaker' if s['speaker']=='unassigned' else 'Mixed audio') for s in rows}
        result=reconcile_window(self.document,win,expected,rows,speakers)
        self.document=result['document']
        self.emit({'type':'provisional_revision','window':win,'expected':expected,'candidates':rows,
                   'speakers':speakers,'fast_sequence':self.fast_sequence,
                   'language_epoch':rows[0]['language_epoch'] if rows else self.language_at(win['start_sample'])['epoch']})
        self.document['segments']=[s for s in self.document['segments'] if s['end']>=end-60]
    def transcribe_owned(self,start,end,config,names):
        """Carry left audio across a forced cap; discard only a verified prefix.

        Reference and extended decodes have the same audio origin. Matching is
        lexical, never a moving suffix dedup or an assertion of word timestamps.
        """
        carry=self.carry if self.carry and self.carry['end']==start and self.carry['epoch']==config['epoch'] else None
        if carry is None and end-start<.5:
            # Reuse causal waveform context within this language epoch, with
            # SAME-origin prefix comparison; never moving-window suffix dedup.
            epoch_start=next(x['start_sample']/RATE for x in self.timeline if x['epoch']==config['epoch'])
            context_start=max(epoch_start,start-3)
            if start-context_start>=.5:
                prefix=read_audio(self.path,round(context_start*RATE),round(start*RATE))
                reference=self.decode(prefix,config['language'],names,context_start,config['epoch'],overlap=len(names)>1)
                carry={'start':context_start,'end':start,'epoch':config['epoch'],'passages':reference}
        begin=carry['start'] if carry else start
        pcm=read_audio(self.path,round(begin*RATE),round(end*RATE))
        if carry and not pcm[round((start-begin)*RATE):].any():
            # A proven empty right-hand crop has no lexical prefix to align.
            pcm=pcm[round((start-begin)*RATE):];begin=start;carry=None
        passages=self.decode(pcm,config['language'],names,begin,config['epoch'],overlap=len(names)>1)
        if not carry:return self.decorate(passages,start,end,names,config['epoch'])
        reference=carry['passages']
        split=split_same_origin(reference[0]['text'],passages[0]['text']) if (len(reference)==len(passages)==1
            and reference[0].get('language')==passages[0].get('language')
            and not reference[0].get('transcription_review') and not passages[0].get('transcription_review')) else None
        if split:
            result={**passages[0],'start':0,'end':end-start,'text':split[1]}
            if not split[1]:result.update(audio_state='context_no_new_words',review=False)
            elif end-start<.2:
                # Complete same-origin decodes corroborate the new suffix.
                # Keep these words visible even when the interval is brief.
                result.update(review=True,audio_state='short_acoustic_context',
                              transcription_review={'reason':'short_acoustic_context','partial_text':False})
            return self.decorate([result],start,end,names,config['epoch'])
        self.emit({'type':'boundary_candidate','start':begin,'end':end,'language_epoch':config['epoch'],'candidates':passages})
        return self.decorate([{'start':0,'end':end-start,'text':'','language':None,'review':True,
                              'transcription_review':{'reason':'refinement_incomplete'}}],start,end,names,config['epoch'])
    def commit(self,final=False):
        while True:
            before=self.cursor;self._commit_once(final)
            if self.cursor==before or self.cursor>=min(self.received/RATE,self.processed):break
    def _commit_once(self,final=False):
        available=min(self.received/RATE,self.processed)
        regions=context_regions(self.turns,coverage=(self.cursor,available))
        for region in regions:
            start=max(self.cursor,region['start']);limit=min(region['end'],available)
            if limit<=start:continue
            config=self.language_at(round(start*RATE))
            switches=[x['start_sample']/RATE for x in self.timeline if start<x['start_sample']/RATE<=limit]
            if switches:limit=min(limit,min(switches))
            settled=final or limit<available-TURN_GAP_SECONDS or bool(switches)
            end=min(limit,start+18)
            if round(end*RATE)<=round(start*RATE):
                if settled:self.cursor=end
                continue
            if settled and end==self.last_decode:
                previous=[copy.deepcopy(s) for s in self.document['segments'] if start<=s['start']<s['end']<=end]
                if previous:
                    for row in previous:row.update(finalized=True,voice_eligible=len(region['speakers'])==1 and not row.get('audio_state') and not region.get('activity_regions'))
                    self.publish(start,end,previous,region['speakers']);self.cursor=end;continue
            if not settled and end-start<6:break
            context_ready=end>=start+18 and available>=end+3
            if not settled and end-self.last_decode<3 and not context_ready:break
            if not settled and end>=start+18 and available<end+3:
                if self.last_decode>=end:break
            decode_end=min(limit,end+3) if end>=start+18 and (settled or available>=end+3) else end
            names=region['speakers']
            rows=self.transcribe_owned(start,decode_end,config,names)
            for row in rows:
                row.update(finalized=settled,voice_eligible=settled and len(names)==1 and not row.get('audio_state') and not region.get('activity_regions'))
                if region.get('activity_regions'):row['activity_regions']=copy.deepcopy(region['activity_regions'])
            if decode_end>end:
                if rows and all(s.get('audio_state')=='digital_silence' for s in rows):
                    self.publish(start,decode_end,rows,names);self.cursor=end;self.last_decode=end;self.carry=None
                    continue
                previous=[s for s in self.document['segments'] if s['start']>=start and s['end']<=end]
                if not previous:
                    previous=self.transcribe_owned(start,end,config,names)
                split=split_same_origin(previous[0]['text'],rows[0]['text']) if len(previous)==len(rows)==1 and previous[0].get('language')==rows[0].get('language') else None
                if split:
                    prefix=copy.deepcopy(rows[0]);prefix.update(end=end,text=split[0],finalized=True,voice_eligible=len(names)==1 and not prefix.get('audio_state') and not region.get('activity_regions'))
                    tail=copy.deepcopy(rows[0]);tail.update(start=end,text=split[1],finalized=False,voice_eligible=False)
                    rows=[prefix,tail]
                else:
                    # Keep the full contextual candidate, never suffix-deduplicate repeats.
                    self.emit({'type':'boundary_candidate','start':start,'end':decode_end,'language_epoch':config['epoch'],'candidates':rows})
                    rows=[{**s,'finalized':True,'boundary_unresolved':True,'refinement_state':'unresolved'} for s in previous]
                    rows.append({'start':end,'end':decode_end,'text':'','speaker':names[0] if len(names)==1 else 'overlap',
                                 'speaker_candidates':names,'language':None,'language_epoch':config['epoch'],
                                 'review':True,'voice_eligible':False,'finalized':False,'refinement_state':'provisional'})
                self.publish(start,decode_end,rows,names);self.cursor=end;self.last_decode=end
                carry_start=max(start,end-3)
                reference=self.decode(read_audio(self.path,round(carry_start*RATE),round(end*RATE)),config['language'],names,carry_start,config['epoch'],overlap=len(names)>1)
                self.carry={'start':carry_start,'end':end,'epoch':config['epoch'],'passages':reference}
            else:
                self.publish(start,end,rows,names);self.last_decode=end
                if settled:self.cursor=end;self.carry=None
                else:break
            if self.cursor>=available:break
        self.turns=[dict(t,start=max(t['start'],self.cursor)) for t in self.turns if t['end']>self.cursor]
    def refine(self,request):
        window=request['window'];a=window['context_start_sample'];b=window['context_end_sample']
        if self.inbox.cancelled_request(request):return {'type':'refinement_result',**{k:request[k] for k in ('operation_id','language_epoch','window')},'cancelled':True}
        pcm=read_audio(self.path,a,b);turns=self.models.batch_turns(pcm)
        absolute=[dict(t,start=t['start']+a/RATE,end=t['end']+a/RATE) for t in turns]
        diagnostics=[] if self.config.get('track_mapping_diagnostics') else None
        raw_turns=copy.deepcopy(absolute) if diagnostics is not None else None
        mapping=align_tracks(absolute,request['references'],diagnostics=diagnostics);mapped=[]
        for t in absolute:
            t['speaker']=mapping.get(t['speaker'],'unknown_mixed');mapped.append(t)
        rows=[];speakers={};context_history=[]
        for region in context_regions(mapped,max_seconds=18,coverage=(window['start_sample']/RATE,window['end_sample']/RATE)):
            if self.inbox.cancelled_request(request):return {'type':'refinement_result',**{k:request[k] for k in ('operation_id','language_epoch','window')},'cancelled':True}
            start=max(region['start'],window['start_sample']/RATE);end=min(region['end'],window['end_sample']/RATE)
            if end<=start:continue
            names=region['speakers'];piece=pcm[round(start*RATE)-a:round(end*RATE)-a]
            passages=self.decode(piece,request['language'],names,start,request['language_epoch'],overlap=True,
                context_seed=request.get('language_context'),observations=context_history)
            produced=self.decorate(passages,start,end,names,request['language_epoch'],True)
            for row in produced:
                row['language_mode']=request['language']
                if region.get('activity_regions'):row.update(activity_regions=activity_slice(region['activity_regions'],row['start'],row['end']),voice_eligible=False)
                if 'unknown_mixed' in names:row.update(speaker='overlap_unknown',voice_eligible=False,review=True,speaker_candidates=[])
                speakers[row['speaker']]=('Unknown speaker' if row['speaker']=='unassigned' else 'Mixed audio' if row['speaker'].startswith('overlap') else 'Speaker '+str(int(row['speaker'].split('_')[-1])+1))
            rows.extend(produced)
        result={'type':'refinement_result','operation_id':request['operation_id'],'language_epoch':request['language_epoch'],
                'window':window,'candidates':coalesce_blanks(rows),'speakers':speakers}
        if diagnostics is not None:result['track_mapping']={'raw_batch_turns':raw_turns,'references':copy.deepcopy(request['references']),
            'scores':diagnostics,'mapping':mapping}
        return result
    def handle(self,message):
        import numpy as np
        kind=message['type']
        if kind=='audio':
            if self.capture_finished:raise ValueError('Audio arrived after capture finished.')
            stamp={'start_sample':message['start_sample'],'language':message['language'],'epoch':message['language_epoch']}
            if stamp['epoch']!=self.timeline[-1]['epoch']:self.timeline.append(stamp)
            pcm=read_audio(self.path,message['start_sample'],message['end_sample'])
            output,self.processed=self.models.feed(pcm);self.received=message['end_sample']
            self.turns.extend(output)
            self.commit()
            self.emit({'type':'progress','processed_seconds':self.processed,'received_seconds':self.received/RATE,**self.models.metrics()})
        elif kind in ('flush','stop'):
            if kind=='stop':
                output,self.processed=self.models.feed(np.empty(0,dtype='float32'),final=True);self.turns.extend(output)
            self.commit(final=True)
            self.emit({'type':'progress','processed_seconds':self.processed,'received_seconds':self.received/RATE,'flush':True,**self.models.metrics()})
            if kind=='stop':self.capture_finished=True;self.emit({'type':'capture_finished','duration':self.received/RATE})
        elif kind=='language':
            generation=message.get('generation',message.get('language_epoch'))
            boundary=message.get('start_sample',message.get('apply_from_sample'))
            if generation!=self.timeline[-1]['epoch']+1 or boundary!=self.received:
                raise ValueError('Invalid ordered language boundary.')
            self.timeline.append({'start_sample':boundary,'language':message['language'],'epoch':generation})
            self.commit()
            self.emit({'type':'language_registered','generation':generation,'language':message['language'],'start_sample':boundary})
        elif kind=='refine':
            try:result=self.refine(message)
            except Exception:result={'type':'refinement_result','operation_id':message['operation_id'],'language_epoch':message['language_epoch'],'window':message['window'],'error':'Refinement failed; previous words and audio retained.'}
            self.emit(result)
        elif kind=='shutdown':self.emit({'type':'finished','duration':self.received/RATE});return False
        return True


def run(config,emit):
    import sys
    with contextlib.redirect_stdout(sys.stderr):models=Models(config)
    inbox=Inbox();inbox.latest_epoch=config.get('language_epoch',0);inbox.historical=bool(config.get('refinement_only'))
    engine=Engine(config,models,emit,inbox)
    emit({'type':'ready','two_pass':True,'asr':'growing_phrase_with_rolling_refinement',**models.metrics()})
    def read():
        try:
            for line in sys.stdin:inbox.push(json.loads(line))
        except Exception:
            with inbox.condition:inbox.error='Invalid resident worker command. Saved audio and prior words are retained.'
        finally:
            with inbox.condition:inbox.closed=True;inbox.condition.notify_all()
    threading.Thread(target=read,daemon=True).start()
    if config.get('refinement_only'):
        with wave.open(config['audio_path'],'rb') as audio:engine.received=audio.getnframes()
        engine.capture_finished=True;emit({'type':'capture_finished','duration':engine.received/RATE})
    while True:
        message=inbox.take()
        if message is None:continue
        continuing=engine.handle(message)
        if not continuing:return
