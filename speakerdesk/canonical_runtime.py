"""Resident canonical utterances, independent of NVIDIA's activity boundaries.

The short utterance path replaces one complete same-anchor Cohere revision.
Long English utterances may use calibrated context ownership. Other languages
and missing/failed alignment use disjoint original-audio crops, preserving every
raw core text without pretending to know individual word boundaries.
"""
import copy
from pathlib import Path
import uuid

from utterances import UtteranceBook, RevisionArchive, MAX_DECODE_SAMPLES, RATE
from reading_turns import alignment_words, bind_words, project_turns


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
                for key in ('language','language_detection','language_review','acoustic_evidence','transcription_review','audio_state','review'):
                    if key in passages[0]:updated[key]=copy.deepcopy(passages[0][key])
                    else:updated.pop(key,None)
            if not complete or not text.strip():updated['canonical_unresolved']='incomplete_cohere_revision'
            else:updated.pop('canonical_unresolved',None)
            if complete and text.strip() and updated['text']==text and self.reading_alignment_enabled(row):
                request={'start_sample':a,'end_sample':b}
                alignment=self.align_reading(request,text,passages[0])
                bind_words(updated,alignment_words(alignment,text,a,b))
        else:
            use_alignment=(stamp['language']=='en' and hasattr(e.models,'align_canonical')
                and (not hasattr(e.models,'alignment_supported') or e.models.alignment_supported('en')))
            if use_alignment:
                parts=self.decode_long_parts(row,stage,self.book.decode_requests(row['id']),aligned=True)
                if parts is False:return False
                self.book.apply_decode_parts(row['id'],parts,stage=stage)
                updated=self.book.rows[row['id']]
                # Keep failed aligned candidates in the journal. If ownership
                # failed, decode separate cores; no uncalibrated CTC fallback.
                if not updated['machine_versions'][-1]['complete'] and all(p['complete'] for p in parts):
                    use_alignment=False
            if not use_alignment:
                from core_plan import plan,validate
                row['decode_core_plan']=plan(row,e.models.speech_live.evidence.admission)
                validate(row['decode_core_plan'],row)
                parts=self.decode_long_parts(row,stage,self.book.core_decode_requests(row['id']),aligned=False)
                if parts is False:return False
                self.book.apply_core_parts(row['id'],parts,stage=stage)
                updated=self.book.rows[row['id']]
                if updated['machine_versions'][-1]['complete']:
                    self.update_core_metadata(updated,parts,stamp['language'])
                    if self.reading_alignment_enabled(row):
                        import re
                        words=[];offset=0
                        for part in parts:
                            request=part['request'];raw=part['text']
                            aligned_words=alignment_words(part.get('reading_alignment'),raw,
                                request['start_sample'],request['end_sample'])
                            if not aligned_words:
                                aligned_words=[{'text':m.group(),'start_char':m.start(),'end_char':m.end(),
                                    'start_sample':None,'end_sample':None} for m in re.finditer(r'\S+',raw)]
                            for word in aligned_words:
                                words.append(dict(word,start_char=word['start_char']+offset,end_char=word['end_char']+offset))
                            offset+=len(raw)+1
                        bind_words(updated,words)
            if use_alignment and updated.get('assembly_provenance'):
                bind_words(updated,updated['assembly_provenance']['words'])
            last=updated['machine_versions'][-1]
            if not last['complete']:updated['canonical_unresolved']=last.get('reason') or 'incomplete_cohere_revision'
            else:updated.pop('canonical_unresolved',None)
        self.last_decoded[row['id']]=(b,row['audio_revision'],stage)
        return True

    def reading_alignment_enabled(self,row):
        models=self.engine.models
        return (row['state']=='sealed' and len(row.get('speaker_candidates',[]))>1
            and hasattr(models,'align_canonical') and hasattr(models,'alignment_supported')
            and models.alignment_supported('en'))

    def align_reading(self,request,text,passage):
        # Timing never changes language routing or supplies a transcript. Auto
        # requires the actual current confident decision, not a recent guess.
        detection=passage.get('language_detection') or {}
        if passage.get('language')!='en' or passage.get('transcription_review'):
            return None
        probes=detection.get('probes')
        if probes:
            regions=[];cursor=request['start_sample']
            for probe in probes:
                a,b=probe.get('start_sample'),probe.get('end_sample')
                if type(a) is not int or type(b) is not int or not cursor==a<b<=request['end_sample']:
                    return None
                cursor=b
                if (probe.get('language')=='en' and not probe.get('review')
                        and probe.get('decision',{}).get('reason') in ('detected','override')):
                    if regions and regions[-1][1]==a:regions[-1][1]=b
                    else:regions.append([a,b])
            if cursor!=request['end_sample']:return None
        elif (not passage.get('language_review',passage.get('review'))
              and detection.get('reason') in ('detected','override')):
            regions=[[request['start_sample'],request['end_sample']]]
        else:
            regions=[]
        if not regions:
            return None
        result=self.engine.models.align_canonical(request,text,language='en')
        if not isinstance(result,dict):return None
        result=copy.deepcopy(result);result['reading_english_regions']=regions
        # Align unchanged whole core text, but trust no envelope in its weak or
        # context-routed language slots. Merge confident contiguous probes so
        # a language detector's feed grid cannot create artificial word gaps.
        for word in result.get('words',[]):
            a,b=word.get('start_sample'),word.get('end_sample')
            if not (type(a) is int and type(b) is int and
                    any(x<=a-4000<b+4000<=y for x,y in regions)):
                word.update(status='unresolved_language',start_sample=None,end_sample=None)
        return result

    def decode_long_parts(self,row,stage,requests,*,aligned):
        from live_refinement import read_audio
        from rolling_refinement import successful_non_speech
        e=self.engine;stamp=e.language_at(row['start_sample']);a,b=row['start_sample'],row['end_sample']
        parts=[]
        for request in requests:
            if stage=='refined' and e.inbox.cancelled_request(self.refinement_request):return False
            x,y=request['start_sample'],request['end_sample'];pcm=read_audio(e.path,x,y)
            try:
                if hasattr(e.models,'set_decode_boundary_padding'):
                    e.models.set_decode_boundary_padding(3200 if x>a else 0,3200 if y<b else 0)
                if stage=='refined' and hasattr(e.models,'begin_refinement'):e.models.begin_refinement(pcm,x)
                passages=e.decode(pcm,stamp['language'],row.get('speaker_candidates',[]),x/RATE,row['language_epoch'],overlap=True)
            finally:
                if hasattr(e.models,'set_decode_boundary_padding'):e.models.set_decode_boundary_padding(0,0)
                if stage=='refined' and hasattr(e.models,'end_refinement'):e.models.end_refinement()
            # Language-routed pieces must cover this core in order. Keep their
            # raw text/provenance; missing or failed ASR cannot become a blank.
            cursor=0;complete=bool(passages);texts=[]
            for passage in passages:
                begin,end=round(passage['start']*RATE),round(passage['end']*RATE)
                text=passage.get('cohere_raw_text',passage['text']);texts.append(text)
                valid=(begin==cursor and begin<end<=y-x and not passage.get('transcription_review')
                       and (bool(text.strip()) or successful_non_speech(dict(passage,start=passage['start']+x/RATE,end=passage['end']+x/RATE))))
                complete=complete and valid;cursor=end
            complete=complete and cursor==y-x and (not aligned or len(passages)==1)
            text=' '.join(texts)
            alignment=None
            if aligned and complete and text.strip():
                alignment=e.models.align_canonical(request,text,language=passages[0].get('language'))
            reading_alignment=None
            if not aligned and complete and text.strip() and len(passages)==1 and self.reading_alignment_enabled(row):
                reading_alignment=self.align_reading(request,text,passages[0])
            parts.append({'request':request,'text':text,'complete':complete,'alignment':alignment,
                          'reading_alignment':reading_alignment,
                          'cohere_input_padding':[copy.deepcopy(p.get('cohere_input_padding')) for p in passages],
                          'passages':passages})
        return parts

    @staticmethod
    def update_core_metadata(row,parts,mode):
        """Current disjoint decode decisions replace stale short-phrase labels.

        Preserve every actual language probe at its absolute physical position;
        a heterogeneous utterance has no single inferred language. Weak routing
        remains reviewable without pretending ASR/alignment verified language.
        """
        probes=[];speech=[]
        for part in parts:
            origin=part['request']['start_sample']
            for passage in part['passages']:
                detection=copy.deepcopy(passage.get('language_detection') or {})
                if detection.get('probes'):
                    probes.extend(copy.deepcopy(detection['probes']))
                else:
                    probes.append({'start_sample':origin+round(passage['start']*RATE),
                        'end_sample':origin+round(passage['end']*RATE),
                        'language':passage.get('language'),'decision':detection,
                        'review':bool(passage.get('review'))})
                if passage['text'].strip():speech.append(passage)
        languages={p.get('language') for p in speech}
        row['language']=next(iter(languages)) if len(languages)==1 else None
        weak=next((p for p in speech if p.get('review') or
                   (p.get('language_detection') or {}).get('reason') not in ('detected','override')),None)
        reason=('override' if mode!='auto' else 'mixed_languages' if len(languages)>1
                else ((weak.get('language_detection') or {}).get('reason') or 'uncertain') if weak
                else 'detected')
        row['language_detection']={'mode':'manual' if mode!='auto' else 'auto','reason':reason,
            'probes':probes,'source':'disjoint_core_language_decisions'}
        row['review']=bool(weak) or (mode=='auto' and (len(languages)!=1 or None in languages))
        # Earlier short-crop evidence does not describe the current whole audio.
        for key in ('acoustic_evidence','transcription_review','audio_state'):row.pop(key,None)

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
        speaker=names[0] if len(names)==1 else 'multiple_speakers' if names else 'unassigned'
        result={key:copy.deepcopy(value) for key,value in row.items()
                if key not in ('machine_versions','protected_fields','last_speech_sample','state','machine_revision','reading_word_evidence','reading_turns','reading_turn_provenance','decode_core_plan')}
        result.update(canonical_utterance_id=row['id'],canonical_machine_revision=row['machine_revision'],
            canonical_state=row['state'],start=row['start_sample']/RATE,end=row['end_sample']/RATE,
            speaker=speaker,source_speaker_candidates=names,language_generation=row['language_epoch'],
            language_mode=stamp['language'],finalized=row['state']=='sealed',
            review=len(names)!=1 or bool(row.get('review')) or bool(row.get('canonical_unresolved')) or bool(row.get('transcription_review')),
            timing='canonical_vad_audio_anchor',refinement_state=row.get('refinement_state','provisional'))
        if stamp['language']!='auto':result['language']=stamp['language']
        turns=project_turns(row)
        if turns:
            import hashlib,json
            result['reading_turns']=turns
            result['reading_turn_provenance']={'utterance_id':row['id'],
                'canonical_machine_revision':row['machine_revision'],'audio_revision':row['audio_revision'],
                'language_epoch':row['language_epoch'],'text_sha256':hashlib.sha256(row['text'].encode()).hexdigest(),
                'audio_anchor':copy.deepcopy(row['text_audio_anchor']),
                'activity_sha256':hashlib.sha256(json.dumps(row['speaker_activity'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                'turns_sha256':hashlib.sha256(json.dumps(turns,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),
                'method':'english_coarse_emissions_temporal_nvidia_v1','transition_margin_samples':4000,
                'calibration_id':row['reading_word_evidence']['calibration_id'],
                'model_sha256':row['reading_word_evidence']['model_sha256']}
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
