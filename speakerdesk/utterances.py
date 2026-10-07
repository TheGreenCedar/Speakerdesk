"""Canonical speech objects; model snapshots replace text instead of appending it.

VAD defines retained utterance audio. NVIDIA activity remains a separate ledger.
All positions reference the original16kHz recording, never inferred word times.
"""
import copy
import hashlib
import re
import json
import os
import math
from pathlib import Path

RATE = 16000
MAX_DECODE_SAMPLES = 392000
CORE_SAMPLES = 18*RATE
CONTEXT_SAMPLES = 3*RATE


def complete_non_speech(row, start, end):
    """A blank negative classification covering this exact physical anchor.

    Empty recognition, legacy/failed evidence and partial negative coverage do
    not authorize removal of machine words. The current admission policy may
    qualify exact constant PCM independently of neural probabilities; neither
    outcome asserts that quiet nonconstant speech is impossible.
    """
    from speech_admission import INPUT_POLICY, SILERO_SPEC
    if (type(start) is not int or type(end) is not int or not 0<=start<end
            or not isinstance(row,dict) or not isinstance(row.get('text'),str)
            or row['text'].strip() or row.get('transcription_review') or row.get('canonical_unresolved')
            or row.get('audio_state')!='model_non_speech'):
        return False
    evidence=row.get('acoustic_evidence')
    if not isinstance(evidence,dict):return False
    probability=evidence.get('maximum_probability')
    return (evidence.get('source')=='silero_v6' and evidence.get('model_revision')==SILERO_SPEC['revision']
        and evidence.get('input_policy')==INPUT_POLICY and evidence.get('complete') is True
        and evidence.get('decision')=='no_speech' and evidence.get('speech_regions')==[]
        and evidence.get('uncertain_regions')==[] and type(probability) in (int,float)
        and math.isfinite(probability) and 0<=probability<.5
        and type(evidence.get('start_sample')) is int and type(evidence.get('end_sample')) is int
        and (evidence['start_sample'],evidence['end_sample'])==(start,end))


class RevisionArchive:
    """Private append-only raw versions; hot versions can then be bounded."""
    def __init__(self, path):
        self.path=Path(path)
        descriptor=os.open(self.path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        os.close(descriptor)

    def append(self,event):
        data=(json.dumps(event,ensure_ascii=False,separators=(',',':'))+'\n').encode()
        with self.path.open('r+b') as output:
            output.seek(0,os.SEEK_END);offset=output.tell()
            try:output.write(data);output.flush();os.fsync(output.fileno())
            except Exception:
                output.truncate(offset)
                raise


class UtteranceBook:
    def __init__(self, job_id, *, silence_samples=8000, context_samples=3200,
                 history_limit=None, archive=None):
        if not isinstance(job_id, str) or not job_id:
            raise ValueError('Utterance identity requires a recording.')
        if (type(silence_samples) is not int or type(context_samples) is not int
                or not 0 < context_samples < silence_samples):
            raise ValueError('Invalid utterance context policy.')
        self.job_id, self.silence_samples, self.context_samples = job_id, silence_samples, context_samples
        if history_limit is not None and (type(history_limit) is not int or history_limit<1 or archive is None):
            raise ValueError('Bounded utterance history requires durable archival.')
        self.history_limit,self.archive=history_limit,archive
        self.cursor = self.epoch_start = self.epoch = 0
        self.rows = {}
        self.order = []
        self.active = None
        self.closed = False
        self.uncertain_sample_count = 0

    @staticmethod
    def _set_end(row, end):
        if row['end_sample'] != end:
            previous_end, previous_revision = row['end_sample'], row['audio_revision']
            row['end_sample'] = end
            row['audio_revision'] += 1
            row.pop('alignment', None)
            evidence = row.get('reading_word_evidence') or {}
            anchor = row.get('text_audio_anchor') or {}
            if (end > previous_end and evidence.get('audio_anchor') == anchor
                    and anchor.get('start_sample') == row['start_sample']
                    and type(anchor.get('end_sample')) is int and anchor['end_sample'] <= previous_end
                    and evidence.get('retained_audio_revision', evidence.get('audio_revision')) == previous_revision):
                # Appending immutable recording samples leaves the aligned
                # text prefix intact. Keep its original anchor/source revision;
                # current activity will independently requalify each word.
                evidence['retained_audio_revision'] = row['audio_revision']
            else:
                row.pop('reading_word_evidence',None)
            row.pop('decode_core_plan',None)
            row.pop('assembly_provenance',None)
            row.pop('bounded_decode_provenance',None)
            row.pop('speaker_activity', None)
            row['voice_eligible'] = False

    def _seal(self, end):
        if self.active is None:
            return
        row = self.rows[self.active]
        self._set_end(row, min(end, row['last_speech_sample']+self.context_samples))
        row['state'] = 'sealed'
        self.active = None

    def boundary(self, sample, epoch):
        if self.closed or type(sample) is not int or sample != self.cursor or type(epoch) is not int or epoch != self.epoch+1:
            raise ValueError('Language boundaries must follow observed audio.')
        self._seal(sample)
        self.epoch_start, self.epoch = sample, epoch

    def observe(self, evidence):
        start, end = evidence['start_sample'], evidence['end_sample']
        if (self.closed or evidence.get('complete') is not True or evidence.get('decision') not in ('speech', 'no_speech','uncertain')
                or type(start) is not int or type(end) is not int or start != self.cursor or end <= start):
            raise ValueError('Canonical utterances require ordered complete speech evidence.')
        regions = evidence['speech_regions']
        if (evidence['decision'] == 'speech') != bool(regions):
            raise ValueError('Speech decision and observed regions disagree.')
        position = start
        for region in regions:
            a, b = region['start_sample'], region['end_sample']
            if type(a) is not int or type(b) is not int or not position <= a < b <= end:
                raise ValueError('Invalid speech-region ledger.')
            position = b
        uncertain=evidence.get('uncertain_regions',
            [{'start_sample':start,'end_sample':end}] if evidence['decision']=='uncertain' else [])
        position=start
        for region in uncertain:
            a,b=region['start_sample'],region['end_sample']
            if type(a) is not int or type(b) is not int or not position<=a<b<=end:
                raise ValueError('Invalid uncertain speech-region ledger.')
            if any(a<speech['end_sample'] and b>speech['start_sample'] for speech in regions):
                raise ValueError('Uncertain speech overlaps admitted speech.')
            position=b
        if uncertain:
            # FrameArchive retains exact ranges/probabilities. Keep a bounded
            # summary independent of speech objects so empty uncertain input
            # cannot disappear from the final completion receipt.
            if self.archive:self.archive.append({'type':'speech_admission_uncertain',
                'start_sample':start,'end_sample':end,'regions':copy.deepcopy(uncertain),
                'language_epoch':self.epoch})
            self.uncertain_sample_count+=sum(region['end_sample']-region['start_sample'] for region in uncertain)
        for region in regions:
            a, b = region['start_sample'], region['end_sample']
            if self.active and a-self.rows[self.active]['last_speech_sample'] >= self.silence_samples:
                self._seal(a)
            if self.active is None:
                previous_end = self.rows[self.order[-1]]['end_sample'] if self.order else 0
                begin = max(self.epoch_start, previous_end, a-self.context_samples)
                identity = hashlib.sha256(f'{self.job_id}:{self.epoch}:{begin}'.encode()).hexdigest()[:24]
                self.active = 'utterance-'+identity
                self.rows[self.active] = {'id': self.active, 'start_sample': begin, 'end_sample': b,
                    'last_speech_sample': b, 'language_epoch': self.epoch, 'state': 'open',
                    'text': '', 'machine_revision': 0, 'audio_revision': 0,
                    'speech_regions': [], 'voice_eligible': False,
                    'protected_fields': [], 'machine_versions': []}
                self.order.append(self.active)
            row = self.rows[self.active]
            if row['speech_regions'] and row['speech_regions'][-1]['end_sample'] == a:
                row['speech_regions'][-1]['end_sample'] = b
            else:
                row['speech_regions'].append({'start_sample': a, 'end_sample': b})
            row['last_speech_sample'] = b
            self._set_end(row, b)
        if self.active:
            row = self.rows[self.active]
            self._set_end(row, min(end, row['last_speech_sample']+self.context_samples))
            if end-row['last_speech_sample'] >= self.silence_samples:
                self._seal(end)
        self.cursor = end
        return self.snapshot()

    def finish(self):
        self._seal(self.cursor)
        self.closed = True
        return self.snapshot()

    def snapshot(self):
        return copy.deepcopy([self.rows[identity] for identity in self.order])

    def edit(self, identity, revision, text):
        row = self.rows[identity]
        if type(revision) is not int or revision != row['machine_revision'] or not isinstance(text, str):
            raise ValueError('Utterance changed; preserve the draft for comparison.')
        if self.archive:
            self.archive.append({'type':'human_edit','utterance_id':identity,
                'base_revision':revision,'previous_text':row['text'],'text':text,
                'audio_revision':row['audio_revision'],
                'audio_anchor':{'start_sample':row['start_sample'],'end_sample':row['end_sample']}})
        row['text'] = text
        row['text_audio_anchor'] = {'start_sample': row['start_sample'], 'end_sample': row['end_sample']}
        row['protected_fields'] = sorted(set(row['protected_fields']) | {'text'})
        row['machine_revision'] += 1
        row.pop('alignment', None)
        row.pop('reading_word_evidence',None)
        row.pop('decode_core_plan',None)
        row.pop('assembly_provenance',None)
        row.pop('bounded_decode_provenance',None)
        return copy.deepcopy(row)

    def apply_model(self, identity, revision, text, *, start_sample, end_sample, stage, complete,
                    non_speech_evidence=None):
        row = self.rows[identity]
        if (type(revision) is not int or revision != row['machine_revision'] or not isinstance(text, str)
                or type(start_sample) is not int or type(end_sample) is not int
                or (start_sample, end_sample) != (row['start_sample'], row['end_sample'])
                or stage not in ('live', 'refined') or type(complete) is not bool):
            raise ValueError('Stale or mismatched utterance result.')
        negative=complete and complete_non_speech(dict(text=text,audio_state='model_non_speech',
            acoustic_evidence=non_speech_evidence),start_sample,end_sample)
        if complete and not text.strip() and not negative:
            complete = False
        version = {'text': text, 'stage': stage, 'complete': complete,
                   'audio_anchor': {'start_sample': start_sample, 'end_sample': end_sample},
                   'base_revision': revision, 'audio_revision': row['audio_revision']}
        if negative:
            version['non_speech_evidence']=copy.deepcopy(non_speech_evidence)
            version['previous_text']=row['text']
        if self.archive:
            self.archive.append({'type':'machine_version','utterance_id':identity,'version':version})
        row['machine_versions'].append(version)
        if self.history_limit is not None:row['machine_versions']=row['machine_versions'][-self.history_limit:]
        # Incomplete decoding is retained as a candidate, never a replacement
        # that silently clears earlier words. Human corrections stay primary.
        if complete and 'text' not in row['protected_fields']:
            row['text'] = text
            row['text_audio_anchor'] = copy.deepcopy(version['audio_anchor'])
            row['machine_revision'] += 1
            row['refinement_state'] = 'refined' if stage == 'refined' and row['state'] == 'sealed' else 'provisional'
            row.pop('alignment', None)
            row.pop('reading_word_evidence',None)
            row.pop('decode_core_plan',None)
            row.pop('assembly_provenance',None)
            row.pop('bounded_decode_provenance',None)
            if negative:
                row.pop('reading_turns',None);row.pop('reading_turn_provenance',None)
                row['voice_eligible']=False
        return copy.deepcopy(row)

    def decode_requests(self, identity, *, contextual=False):
        """Bounded core/context requests, without assigning words to cores."""
        if contextual:
            from context_plan import requests
            return requests(self.rows[identity])
        row=self.rows[identity];start,end=row['start_sample'],row['end_sample']
        requests=[]
        for core_start in range(start,end,CORE_SAMPLES):
            core_end=min(end,core_start+CORE_SAMPLES)
            a,b=max(start,core_start-CONTEXT_SAMPLES),min(end,core_end+CONTEXT_SAMPLES)
            if b-a>MAX_DECODE_SAMPLES:raise ValueError('Oversized canonical decode request.')
            requests.append({'utterance_id':identity,'machine_revision':row['machine_revision'],
                'audio_revision':row['audio_revision'],'language_epoch':row['language_epoch'],
                'core_start_sample':core_start,'core_end_sample':core_end,
                'start_sample':a,'end_sample':b})
        return requests

    def core_decode_requests(self, identity):
        """Balanced disjoint crops avoid an unusably short final remainder."""
        row=self.rows[identity];a,b=row['start_sample'],row['end_sample']
        if 'decode_core_plan' in row:
            from core_plan import validate
            edges=validate(row['decode_core_plan'],row)
        else:
            count=max(1,(b-a+CORE_SAMPLES-1)//CORE_SAMPLES)
            edges=[a+(b-a)*i//count for i in range(count+1)]
        return [{'utterance_id':identity,'machine_revision':row['machine_revision'],
            'audio_revision':row['audio_revision'],'language_epoch':row['language_epoch'],
            'core_start_sample':x,'core_end_sample':y,'start_sample':x,'end_sample':y}
            for x,y in zip(edges,edges[1:])]

    def apply_core_parts(self, identity, parts, *, stage):
        """Retain every whole raw core decode; no shared context or word timing.

        Cores partition physical audio, not word ownership. A word crossing a
        crop edge can be misrecognized; no duplicate-removal or timing claim
        conceals that recognition limit. Failed/empty ASR is never non-speech.
        """
        if stage not in ('live','refined'):raise ValueError('Invalid canonical decode stage.')
        row=self.rows[identity];requests=self.core_decode_requests(identity)
        if not isinstance(parts,list) or len(parts)!=len(requests):
            raise ValueError('Missing disjoint core decodes.')
        for request,part in zip(requests,parts):
            if (not isinstance(part,dict) or part.get('request')!=request
                    or not isinstance(part.get('text'),str) or type(part.get('complete')) is not bool):
                raise ValueError('Disjoint core belongs to another revision or audio anchor.')
        complete=all(part['complete'] for part in parts)
        text=' '.join(part['text'] for part in parts)
        event={'type':'disjoint_core_machine_version','utterance_id':identity,
            'audio_revision':row['audio_revision'],'base_revision':row['machine_revision'],
            'stage':stage,'complete':complete,'parts':copy.deepcopy(parts),
            'core_plan':copy.deepcopy(row.get('decode_core_plan'))}
        if self.archive:self.archive.append(event)
        self.apply_model(identity,row['machine_revision'],text,start_sample=row['start_sample'],
            end_sample=row['end_sample'],stage=stage,complete=complete)
        if complete and text.strip() and 'text' not in row['protected_fields']:
            row['bounded_decode_provenance']={'method':'disjoint_original_audio_cores',
                'core_plan':copy.deepcopy(event['core_plan']),
                'separator_policy':'one_space_between_whole_raw_core_texts','word_timing':None,
                'calibration_id':None,'audio_revision':row['audio_revision'],
                'machine_revision':row['machine_revision'],
                'audio_anchor':copy.deepcopy(row['text_audio_anchor']),
                'parts':[{'audio_anchor':{'start_sample':p['request']['start_sample'],'end_sample':p['request']['end_sample']},
                          'raw_text_sha256':hashlib.sha256(p['text'].encode()).hexdigest(),
                          'cohere_input_padding':copy.deepcopy(p.get('cohere_input_padding'))} for p in parts]}
            row.pop('alignment',None)
            row.pop('reading_word_evidence',None);row.pop('assembly_provenance',None)
            row['voice_eligible']=False
        return copy.deepcopy(row)

    def apply_decode_parts(self, identity, parts, *, stage, contextual=False):
        """CAS-bound bounded decoding; no lexical stitching of context windows."""
        from canonical_assembly import assemble_parts
        if stage not in ('live','refined'):raise ValueError('Invalid canonical decode stage.')
        row=self.rows[identity]
        assembled=assemble_parts(row,self.decode_requests(identity,contextual=contextual),parts)
        version={'text':assembled['text'],'stage':stage,'complete':assembled['complete'],
            'reason':assembled['reason'],'base_revision':row['machine_revision'],
            'audio_revision':row['audio_revision'],
            'audio_anchor':{'start_sample':row['start_sample'],'end_sample':row['end_sample']}}
        if contextual:
            from context_plan import POLICY
            version['context_policy']=POLICY
        # Keep every raw full-context text/alignment in the durable journal.
        # Hot history contains the assembly result, not duplicate raw contexts.
        if self.archive:self.archive.append({'type':'bounded_machine_version','utterance_id':identity,
            'version':version,'parts':parts})
        else:version['parts']=copy.deepcopy(parts)
        row['machine_versions'].append(copy.deepcopy(version))
        if self.history_limit is not None:row['machine_versions']=row['machine_versions'][-self.history_limit:]
        if assembled['complete'] and 'text' not in row['protected_fields']:
            row['text']=assembled['text'];row['machine_revision']+=1
            row['text_audio_anchor']=copy.deepcopy(version['audio_anchor'])
            row['refinement_state']='refined' if stage=='refined' and row['state']=='sealed' else 'provisional'
            row.pop('alignment',None)
            row.pop('reading_word_evidence',None)
            row.pop('decode_core_plan',None)
            row.pop('bounded_decode_provenance',None)
            row['assembly_provenance']={key:copy.deepcopy(assembled[key]) for key in
                ('text_sha256','words','alignment_complete','model_sha256','timing_kind',
                 'frame_calibration_id','score_calibration_id','separator_policy')}
            row['assembly_provenance'].update(machine_revision=row['machine_revision'],
                audio_revision=row['audio_revision'],audio_anchor=copy.deepcopy(version['audio_anchor']))
            if contextual:row['assembly_provenance']['context_policy']=version['context_policy']
            if assembled['alignment_complete']:
                self.attach_alignment(identity,row['machine_revision'],assembled['text_sha256'],
                    assembled['words'],audio_revision=row['audio_revision'])
                row['alignment'].update({key:assembled[key] for key in
                    ('model_sha256','timing_kind','frame_calibration_id','score_calibration_id','separator_policy')})
        return copy.deepcopy(row)

    def apply_authoritative_parts(self, identity, parts, *, stage, vocabulary=None,committed_receipts=None):
        from authoritative_tail import assemble,requests,POLICY
        if stage not in ('live','refined'):raise ValueError('Invalid canonical decode stage.')
        row=self.rows[identity]
        result=assemble(row,requests(row),parts,vocabulary=vocabulary,committed_receipts=committed_receipts)
        # Engine authorities are immutable once-only disk records. Bind the
        # current CAS separately; don't duplicate old acoustic arrays every tick.
        references=all(isinstance(part.get('authority_reference'),dict) for part in parts)
        archived_parts=([{'authority_reference':copy.deepcopy(part['authority_reference']),
            'request':copy.deepcopy(part['request']),
            'source_inference_request':copy.deepcopy(part.get('source_inference_request',part['request']))}
            for part in parts] if references else copy.deepcopy(parts))
        archived_result=copy.deepcopy(result)
        if references:
            archived_result.pop('words',None)
            archived_result['word_count']=len(result['words'])
            archived_result['rollover_receipts']=[{key:receipt[key] for key in
                ('policy','nominal_frontier_sample','left_raw_end_char','right_raw_start_char',
                 'left_text_sha256','right_text_sha256','model_sha256','calibration_id',
                 'uncertainty_samples','bidirectional_unique','intersecting_cell_count',
                 'committed','anchor_selection','later_unmapped_raw_anchors',
                 'left_source_request','right_source_request','left_source_audio_sha256','right_source_audio_sha256')}
                for receipt in result['rollover_receipts']]
        event={'type':'authoritative_machine_version','utterance_id':identity,
            'base_revision':row['machine_revision'],'audio_revision':row['audio_revision'],
            'stage':stage,'policy':POLICY,'parts':archived_parts,'result':archived_result}
        if self.archive:self.archive.append(event)
        version={'text':result['text'],'stage':stage,'complete':result['complete'],
            'recognition_complete':result['recognition_complete'],'reason':result['reason'],
            'base_revision':row['machine_revision'],'audio_revision':row['audio_revision'],
            'audio_anchor':{'start_sample':row['start_sample'],'end_sample':row['end_sample']},'context_policy':POLICY}
        if not self.archive:version['parts']=copy.deepcopy(parts)
        row['machine_versions'].append(version)
        if self.history_limit is not None:row['machine_versions']=row['machine_versions'][-self.history_limit:]
        row['recognition_complete']=result['recognition_complete']
        row['local_seam_review']=copy.deepcopy(result['local_seams'])
        if result['complete'] and 'text' not in row['protected_fields']:
            row['text']=result['text'];row['machine_revision']+=1
            row['text_audio_anchor']=copy.deepcopy(version['audio_anchor'])
            row['refinement_state']='refined' if stage=='refined' and row['state']=='sealed' else 'provisional'
            for key in ('alignment','reading_word_evidence','decode_core_plan','bounded_decode_provenance'):row.pop(key,None)
            row['assembly_provenance']={key:copy.deepcopy(result[key]) for key in
                ('text_sha256','words','alignment_complete','model_sha256','timing_kind','frame_calibration_id','score_calibration_id','rollover_receipts')}
            row['assembly_provenance'].update(context_policy=POLICY,machine_revision=row['machine_revision'],
                audio_revision=row['audio_revision'],audio_anchor=copy.deepcopy(version['audio_anchor']))
        return copy.deepcopy(row)

    def attach_alignment(self, identity, revision, text_sha256, words, *, audio_revision):
        row = self.rows[identity]
        if (type(revision) is not int or revision != row['machine_revision']
                or type(audio_revision) is not int or audio_revision != row['audio_revision']
                or text_sha256 != hashlib.sha256(row['text'].encode()).hexdigest()):
            raise ValueError('Alignment belongs to a different transcript revision.')
        if not isinstance(words, list) or not words:
            raise ValueError('Alignment did not provide word positions.')
        if row.get('text_audio_anchor')!={'start_sample':row['start_sample'],'end_sample':row['end_sample']}:
            raise ValueError('Alignment requires the exact current text audio anchor.')
        raw_units=list(re.finditer(r'\S+',row['text']))
        if len(words)!=len(raw_units):raise ValueError('Alignment does not cover the complete raw text.')
        previous = row['start_sample']
        for word,raw in zip(words,raw_units):
            a, b = word['start_sample'], word['end_sample']
            if (type(a) is not int or type(b) is not int or not previous <= a < b <= row['end_sample']
                    or word.get('text')!=raw.group()
                    or type(word.get('start_char')) is not int or type(word.get('end_char')) is not int
                    or (word.get('start_char'),word.get('end_char'))!=raw.span()):
                raise ValueError('Alignment is outside original utterance audio.')
            previous = b
        row['alignment'] = {'machine_revision': revision, 'audio_revision': audio_revision,
                            'text_sha256': text_sha256, 'words': copy.deepcopy(words)}
        return copy.deepcopy(row)

    def attach_activity(self, identity, audio_revision, regions, *, observed_end_sample=None):
        """Retain NVIDIA activity independently; it cannot split Cohere words."""
        row = self.rows[identity]
        if type(audio_revision) is not int or audio_revision != row['audio_revision']:
            raise ValueError('Speaker activity belongs to different utterance audio.')
        observed_end_sample = row['end_sample'] if observed_end_sample is None else observed_end_sample
        if (type(observed_end_sample) is not int
                or not row['start_sample'] <= observed_end_sample <= row['end_sample']):
            raise ValueError('Invalid speaker-activity observation horizon.')
        cursor = row['start_sample']
        owners = set()
        clean = True
        for region in regions:
            a, b, speakers = region['start_sample'], region['end_sample'], region['speakers']
            if (type(a) is not int or type(b) is not int or not cursor == a < b <= row['end_sample']
                    or not isinstance(speakers, list)
                    or any(not isinstance(s, str) or not s for s in speakers)
                    or len(speakers) != len(set(speakers))
                    or (speakers and b > observed_end_sample)):
                raise ValueError('Speaker activity must cover ordered original audio.')
            cursor = b
            # The full anchor is the possible voice clip. Independent NVIDIA
            # activity in VAD-negative context must not be erased by Silero.
            owners.update(speakers)
            speech = any(s['start_sample'] < b and s['end_sample'] > a for s in row['speech_regions'])
            if speakers or speech:
                clean = clean and len(speakers) == 1 and bool(re.fullmatch(r'speaker_\d+', speakers[0]))
        if cursor != row['end_sample']:
            raise ValueError('Incomplete speaker-activity evidence.')
        row['speaker_activity'] = {'audio_revision': audio_revision,
            'observed_end_sample': observed_end_sample, 'regions': copy.deepcopy(regions)}
        row['speaker_candidates'] = sorted(owners)
        row['voice_eligible'] = (row['state'] == 'sealed' and clean and len(owners) == 1
            and observed_end_sample == row['end_sample'])
        return copy.deepcopy(row)
