"""Resident canonical utterances, independent of NVIDIA's activity boundaries.

The short utterance path replaces one complete same-anchor Cohere revision.
Long English utterances may use calibrated context ownership. Other languages
use disjoint original-audio crops, preserving every
raw core text without pretending to know individual word boundaries.
"""
import copy
import hashlib
from pathlib import Path
import re
import uuid

from utterances import UtteranceBook, RevisionArchive, MAX_DECODE_SAMPLES, RATE, complete_non_speech
from reading_turns import alignment_words, bind_words, project_turns


def routed_short_revision(passages, start, end):
    """Preserve successful disjoint same-language raw decodes, without stitching.

    LID routing can establish context only after an earlier ASR succeeds, so
    even one short utterance can return several adjacent routes in one language.
    Their exact original-audio slices authorize ordered raw text, not word times.
    """
    import math
    from transcript import LANGUAGES
    cursor=0;language=None;parts=[]
    if not isinstance(passages,list) or len(passages)<2:return None
    for passage in passages:
        if not isinstance(passage,dict):return None
        first,last=passage.get('start'),passage.get('end')
        if any(type(value) not in (int,float) or not math.isfinite(value) for value in (first,last)):return None
        first,last=round(first*RATE),round(last*RATE)
        raw=passage.get('cohere_raw_text',passage.get('text'))
        selected=passage.get('language');evidence=passage.get('acoustic_evidence') or {}
        if (first!=cursor or not first<last<=end-start
                or not isinstance(raw,str) or not raw.strip()
                or selected not in LANGUAGES or (language is not None and selected!=language)
                or passage.get('transcription_review') or passage.get('audio_state')
                or not isinstance(passage.get('language_detection'),dict)
                or not isinstance(evidence,dict) or evidence.get('source')!='silero_v6'
                or evidence.get('complete') is not True or evidence.get('decision')!='speech'
                or evidence.get('start_sample')!=start+first or evidence.get('end_sample')!=start+last
                or evidence.get('uncertain_regions') or not evidence.get('speech_regions')):
            return None
        language=selected;cursor=last
        parts.append({'start_sample':start+first,'end_sample':start+last,
                      'text':raw,'text_sha256':hashlib.sha256(raw.encode()).hexdigest(),
                      'passage':copy.deepcopy(passage)})
    if cursor!=end-start:return None
    return {'text':' '.join(part['text'] for part in parts),
            'provenance':{'method':'contiguous_same_language_passages_v1','separator':' ',
                'audio_anchor':{'start_sample':start,'end_sample':end},'parts':parts,'word_timing':None}}


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
        from authority_store import AuthorityStore
        self.last_decoded={};self.published={};self.authoritative_commitments={}
        self.authoritative_store=AuthorityStore(Path(engine.path).parent/f'authority-{uuid.uuid4().hex}')

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
        # Historical refinement inspected the same immutable full anchor anew.
        # Its complete negative result can replace prior machine words; empty
        # recognition and contextual seam inference cannot prove non-speech.
        historical=getattr(e.models,'speech_historical',None)
        if historical is not None:
            try:evidence=historical.admission(a,b)
            except (RuntimeError,ValueError,OSError,TypeError,KeyError):evidence=None
            negative=dict(text='',audio_state='model_non_speech',acoustic_evidence=evidence)
            if complete_non_speech(negative,a,b):
                self.book.apply_model(row['id'],row['machine_revision'],'',start_sample=a,end_sample=b,
                    stage=stage,complete=True,non_speech_evidence=negative['acoustic_evidence'])
                updated=self.book.rows[row['id']]
                updated.update(audio_state='model_non_speech',acoustic_evidence=negative['acoustic_evidence'],
                    language=None if stamp['language']=='auto' else stamp['language'],language_review=True,review=True,
                    language_detection={'mode':'auto' if stamp['language']=='auto' else 'manual',
                                        'reason':'insufficient_speech'})
                updated.pop('transcription_review',None);updated.pop('canonical_unresolved',None)
                self.last_decoded[row['id']]=(b,row['audio_revision'],stage)
                return True
        if b-a<=self.decode_limit(row):
            passages=e.decode(read_audio(e.path,a,b),stamp['language'],names,a/RATE,row['language_epoch'],overlap=True)
            # Disjoint same-language routes retain exact raw text and slices;
            # their adjacency does not authorize word stitching or word times.
            routed=routed_short_revision(passages,a,b) if len(passages)>1 else None
            complete=(len(passages)==1 and not passages[0].get('transcription_review')
                      and round(passages[0]['start']*RATE)==0
                      and round(passages[0]['end']*RATE)==b-a)
            complete=complete or routed is not None
            text=passages[0].get('cohere_raw_text',passages[0]['text']) if len(passages)==1 else routed['text'] if routed else ''
            negative=complete and len(passages)==1 and complete_non_speech(passages[0],a,b)
            self.archive.append({'type':'cohere_passages','utterance_id':row['id'],
                'audio_revision':row['audio_revision'],'machine_revision':row['machine_revision'],
                'audio_anchor':{'start_sample':a,'end_sample':b},'passages':passages})
            self.book.apply_model(row['id'],row['machine_revision'],text,
                start_sample=a,end_sample=b,stage=stage,complete=complete,
                non_speech_evidence=passages[0]['acoustic_evidence'] if negative else None)
            updated=self.book.rows[row['id']]
            if len(passages)==1:
                for key in ('language','language_detection','language_review','acoustic_evidence','transcription_review','audio_state','review'):
                    if key in passages[0]:updated[key]=copy.deepcopy(passages[0][key])
                    else:updated.pop(key,None)
            elif routed:
                self.update_core_metadata(updated,[{'request':{'start_sample':a},'passages':passages}],stamp['language'])
                updated['language_detection']['source']='disjoint_short_language_decisions_v1'
                updated['language_review']=any(p.get('language_review',p.get('review',False)) for p in passages)
                if 'text' not in updated['protected_fields'] and updated['text']==routed['text']:
                    updated['bounded_decode_provenance']=routed['provenance']
            if not complete or (not text.strip() and not negative):updated['canonical_unresolved']='incomplete_cohere_revision'
            else:updated.pop('canonical_unresolved',None)
            if complete and text.strip() and updated['text']==text and self.reading_alignment_enabled(row):
                request={'start_sample':a,'end_sample':b}
                # A complete routed aggregate is qualified by every original
                # route's English probe, not the first piece's language label.
                passage=passages[0] if len(passages)==1 else updated
                alignment=self.align_reading(request,text,passage)
                bind_words(updated,alignment_words(alignment,text,a,b))
        else:
            use_alignment=((stamp['language']=='en' or (stamp['language']=='auto' and row.get('language')=='en'))
                and hasattr(e.models,'align_canonical')
                and (not hasattr(e.models,'alignment_supported') or e.models.alignment_supported('en')))
            if use_alignment:
                authoritative=e.config.get('authoritative_tail',True)
                if authoritative:
                    from authoritative_tail import requests
                    wanted=requests(row)
                else:wanted=self.book.decode_requests(row['id'],contextual=True)
                parts=self.decode_long_parts(row,stage,wanted,aligned=True,authoritative=authoritative)
                if parts is False:return False
                if authoritative:
                    vocabulary=getattr(getattr(e.models,'coarse_aligner',None),'vocabulary',None)
                    operation=self.refinement_request['operation_id'] if stage=='refined' else 'live'
                    key=(row['id'],row['language_epoch'],stage)
                    previous=self.authoritative_commitments.get(key,{})
                    locked=previous.get('receipts',{}) if previous.get('operation')==operation else {}
                    current=self.book.apply_authoritative_parts(row['id'],parts,stage=stage,vocabulary=vocabulary,
                        committed_receipts=locked)
                    if current['machine_versions'][-1]['complete'] and 'text' not in current['protected_fields']:
                        self.authoritative_commitments[key]={'operation':operation,'receipts':{
                            receipt['nominal_frontier_sample']:copy.deepcopy(receipt)
                            for receipt in current['assembly_provenance']['rollover_receipts'] if receipt['committed']}}

                else:self.book.apply_decode_parts(row['id'],parts,stage=stage,contextual=True)
                updated=self.book.rows[row['id']]
                # A failed contextual seam remains an explicit candidate. A
                # successful disjoint decode cannot certify this boundary or
                # silently replace protected/prior words with clipped crops.
                if updated['machine_versions'][-1]['complete'] and 'text' not in updated['protected_fields']:
                    self.update_context_metadata(updated,parts,stamp['language'])
                    updated.pop('context_candidate_language',None)
                else:
                    # Candidate routing cannot change the primary revision's
                    # eligibility and make an unchanged retry bypass context.
                    candidate={}
                    self.update_context_metadata(candidate,parts,stamp['language'])
                    updated['context_candidate_language']={key:copy.deepcopy(candidate[key]) for key in
                        ('language','language_review','review','language_detection')}
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

    def decode_limit(self,row):
        stamp=self.engine.language_at(row['start_sample']);models=self.engine.models
        if ((stamp['language']=='en' or (stamp['language']=='auto' and row.get('language')=='en'))
                and hasattr(models,'align_canonical')
                and (not hasattr(models,'alignment_supported') or models.alignment_supported('en'))):
            from context_plan import PHYSICAL_MAX
            return PHYSICAL_MAX
        return MAX_DECODE_SAMPLES

    def reading_alignment_enabled(self,row):
        models=self.engine.models
        # Open revisions already have an exact text/audio anchor. Waiting for
        # a VAD seal hides sequential speaker turns during continuous speech.
        return (len(row.get('speaker_candidates',[]))>1
            and hasattr(models,'align_canonical') and hasattr(models,'alignment_supported')
            and models.alignment_supported('en'))

    def align_reading(self,request,text,passage):
        # Actual English decoder routing establishes this provider's scope.
        # LID uncertainty remains review metadata, independent of acoustic
        # timing and NVIDIA activity. CTC neither certifies language nor repairs
        # the transcript. Unsupported/mixed routes and failed ASR stay excluded.
        detection=passage.get('language_detection') or {}
        if not isinstance(detection,dict):return None
        if (passage.get('language')!='en' or passage.get('transcription_review')
                or detection.get('reason') in ('mixed_languages','unsupported','needs_language','insufficient_speech',
                    'speech_admission_uncertain','speech_evidence_pending')):
            return None
        probes=detection.get('probes')
        if 'probes' in detection:
            if not isinstance(probes,list) or not probes:return None
            cursor=request['start_sample']
            for probe in probes:
                if not isinstance(probe,dict) or not isinstance(probe.get('decision',{}),dict):return None
                a,b=probe.get('start_sample'),probe.get('end_sample')
                if (type(a) is not int or type(b) is not int or not cursor==a<b<=request['end_sample']
                        or probe.get('language')!='en'
                        or probe.get('decision',{}).get('reason') in ('unsupported','needs_language',
                            'insufficient_speech','speech_admission_uncertain','speech_evidence_pending')):
                    return None
                cursor=b
            if cursor!=request['end_sample']:return None
        result=self.engine.models.align_canonical(request,text,language='en')
        if not isinstance(result,dict):return None
        result=copy.deepcopy(result)
        result['reading_timing_qualification']='english_decoder_and_calibrated_acoustic_evidence'
        result['reading_language_review']=bool(passage.get('language_review',passage.get('review')))
        return result

    def decode_long_parts(self,row,stage,requests,*,aligned,authoritative=False):
        from live_refinement import read_audio
        from rolling_refinement import successful_non_speech
        e=self.engine;stamp=e.language_at(row['start_sample']);a,b=row['start_sample'],row['end_sample']
        parts=[]
        for request in requests:
            if stage=='refined' and e.inbox.cancelled_request(self.refinement_request):return False
            x,y=request['start_sample'],request['end_sample'];pcm=read_audio(e.path,x,y)
            digest=hashlib.sha256(pcm.astype('<f4').tobytes()).hexdigest()
            operation=self.refinement_request['operation_id'] if stage=='refined' else 'live'
            key=(row['id'],row['language_epoch'],stamp['language'],stage,operation,x,y,digest)
            saved=self.authoritative_store.get(key) if authoritative else None
            if saved is not None:
                saved['authority_reference']=self.authoritative_store.reference(key)
                saved['request']=copy.deepcopy(request);saved['reuse']='previous_selected_same_pcm_authority'
                # Chosen text is stable; failed optional timing is retryable.
                # Explicit ASR refinement/retry gets a distinct operation key.
                if saved.get('alignment') is None and hasattr(e.models,'align_canonical'):
                    saved['alignment']=e.models.align_canonical(request,saved['text'],language=saved['passages'][0].get('language'))
                parts.append(saved);continue
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
            part={'request':request,'source_inference_request':copy.deepcopy(request),'source_audio_float32_sha256':digest,'text':text,'complete':complete,'alignment':alignment,
                          'reading_alignment':reading_alignment,
                          'cohere_input_padding':[copy.deepcopy(p.get('cohere_input_padding')) for p in passages],
                          'passages':passages}
            parts.append(part)
            if authoritative and complete:
                self.authoritative_store.put(key,part)
                part['authority_reference']=self.authoritative_store.reference(key)
        return parts

    @staticmethod
    def update_context_metadata(row,parts,mode):
        passages=[p for part in parts for p in part['passages'] if p['text'].strip()]
        languages={p.get('language') for p in passages}
        row['language']=next(iter(languages)) if len(languages)==1 else None
        weak=next((p for p in passages if p.get('language_review',p.get('review'))),None)
        reason=('override' if mode!='auto' else 'mixed_languages' if len(languages)>1
                else ((weak.get('language_detection') or {}).get('reason') or 'uncertain') if weak else 'detected')
        row['language_review']=bool(weak)
        row['review']=bool(weak) or (mode=='auto' and (len(languages)!=1 or None in languages))
        row['language_detection']={'mode':'manual' if mode!='auto' else 'auto','reason':reason,
            'source':'contextual_decode_language_decisions','contexts':[
                {'start_sample':part['request']['start_sample'],'end_sample':part['request']['end_sample'],
                 'passages':[{'language':p.get('language'),'language_review':p.get('language_review',p.get('review')),
                              'language_detection':copy.deepcopy(p.get('language_detection'))} for p in part['passages']]}
                for part in parts]}
        for key in ('acoustic_evidence','transcription_review','audio_state'):row.pop(key,None)

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
            if b-a<=self.decode_limit(current):
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
        turns=project_turns(row)
        attributed=any(turn['attribution']!='unknown' for turn in turns)
        # A union over an utterance is activity evidence, not attribution of
        # its unsplit words. Missing timing preserves raw text as unassigned.
        single=len(names)==1 and bool(re.fullmatch(r'speaker_\d+',names[0]))
        speaker=names[0] if single else 'multiple_speakers' if attributed else 'unassigned'
        result={key:copy.deepcopy(value) for key,value in row.items()
                if key not in ('machine_versions','protected_fields','last_speech_sample','state','machine_revision','reading_word_evidence','reading_turns','reading_turn_provenance','decode_core_plan')}
        result.update(canonical_utterance_id=row['id'],canonical_machine_revision=row['machine_revision'],
            canonical_state=row['state'],start=row['start_sample']/RATE,end=row['end_sample']/RATE,
            speaker=speaker,source_speaker_candidates=names,language_generation=row['language_epoch'],
            language_mode=stamp['language'],finalized=row['state']=='sealed',
            review=not single or bool(row.get('review')) or bool(row.get('canonical_unresolved')) or bool(row.get('transcription_review')),
            timing='canonical_vad_audio_anchor',refinement_state=row.get('refinement_state','provisional'))
        if stamp['language']!='auto':result['language']=stamp['language']
        if turns:
            import hashlib,json
            result['reading_turns']=turns
            result['reading_turn_provenance']={'utterance_id':row['id'],
                'canonical_machine_revision':row['machine_revision'],'audio_revision':row['audio_revision'],
                'language_epoch':row['language_epoch'],'text_sha256':hashlib.sha256(row['text'].encode()).hexdigest(),
                'audio_anchor':copy.deepcopy(row['text_audio_anchor']),
                'activity_sha256':hashlib.sha256(json.dumps(row['speaker_activity'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                'turns_sha256':hashlib.sha256(json.dumps(turns,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest(),
                'method':'english_coarse_emissions_temporal_nvidia_v2','transition_margin_samples':4000,
                'uncertainty_policy':'emission_inside_activity_no_competing_owner_in_margin',
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

    def commit(self,final=False,*,decode_open=True):
        e=self.engine;available=min(e.received,e.models.speech_live.evidence.end_sample)
        self.observe(available)
        if e.capture_finished and available==e.received and not self.book.closed:self.book.finish()
        for identity in self.book.order:
            row=self.book.rows[identity]
            previous=self.published.get(identity)
            if (row['state']=='sealed' and previous and previous['canonical_state']=='sealed'
                    and previous['audio_revision']==row['audio_revision']):continue
            observed = max(row['start_sample'], min(row['end_sample'], e.received,
                getattr(e.models,'nvidia_observed_sample',round(e.processed*RATE))))
            self.book.attach_activity(identity,row['audio_revision'],
                activity_regions(e.turns,row['start_sample'],row['end_sample'],observed=observed),
                observed_end_sample=observed)
            last=self.last_decoded.get(identity)
            changed=last is None or last[1]!=row['audio_revision']
            # First text/language routing keeps its six-second deadline. Only
            # later open revisions are superseded by newer queued audio; seals
            # and explicit flush/Stop remain mandatory original-anchor work.
            due=final or row['state']=='sealed' or (row['end_sample']-row['start_sample']>=6*RATE
                and (last is None or (decode_open and row['end_sample']-last[0]>=3*RATE)))
            limit=self.decode_limit(row)
            # A selected tail revises with new audio; stable earlier physical
            # hypotheses are reused, not independently re-recognized each tick.
            stamp=e.language_at(row['start_sample'])
            authoritative=(e.config.get('authoritative_tail',True) and (stamp['language']=='en' or
                (stamp['language']=='auto' and row.get('language')=='en')) and hasattr(e.models,'align_canonical')
                and (not hasattr(e.models,'alignment_supported') or e.models.alignment_supported('en')))
            if row['end_sample']-row['start_sample']>limit and not authoritative:
                row.setdefault('canonical_unresolved','unresolved_alignment')
            if changed and due and (row['end_sample']-row['start_sample']<=limit or row['state']=='sealed' or final or authoritative):
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

        if retire:
            self.authoritative_commitments={key:value for key,value in self.authoritative_commitments.items() if key[0] not in retire}
        if retire:self.book.order=[identity for identity in self.book.order if identity not in set(retire)]
        e.turns=[turn for turn in e.turns if round(turn['end']*RATE)>max(0,self.book.cursor-60*RATE)
                 or (self.book.active and round(turn['end']*RATE)>self.book.rows[self.book.active]['start_sample'])]
