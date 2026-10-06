"""Canonical speech objects; model snapshots replace text instead of appending it.

VAD defines retained utterance audio. NVIDIA activity remains a separate ledger.
All positions reference the original16kHz recording, never inferred word times.
"""
import copy
import hashlib
import re
import json
import os
from pathlib import Path

RATE = 16000
MAX_DECODE_SAMPLES = 392000
CORE_SAMPLES = 18*RATE
CONTEXT_SAMPLES = 3*RATE


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

    @staticmethod
    def _set_end(row, end):
        if row['end_sample'] != end:
            row['end_sample'] = end
            row['audio_revision'] += 1
            row.pop('alignment', None)
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
        if (self.closed or evidence.get('complete') is not True or evidence.get('decision') not in ('speech', 'no_speech')
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
        return copy.deepcopy(row)

    def apply_model(self, identity, revision, text, *, start_sample, end_sample, stage, complete):
        row = self.rows[identity]
        if (type(revision) is not int or revision != row['machine_revision'] or not isinstance(text, str)
                or type(start_sample) is not int or type(end_sample) is not int
                or (start_sample, end_sample) != (row['start_sample'], row['end_sample'])
                or stage not in ('live', 'refined') or type(complete) is not bool):
            raise ValueError('Stale or mismatched utterance result.')
        if complete and not text.strip():
            complete = False
        version = {'text': text, 'stage': stage, 'complete': complete,
                   'audio_anchor': {'start_sample': start_sample, 'end_sample': end_sample},
                   'base_revision': revision, 'audio_revision': row['audio_revision']}
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
        return copy.deepcopy(row)

    def decode_requests(self, identity):
        """Bounded core/context requests, without assigning words to cores."""
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

    def attach_alignment(self, identity, revision, text_sha256, words, *, audio_revision):
        row = self.rows[identity]
        if (type(revision) is not int or revision != row['machine_revision']
                or type(audio_revision) is not int or audio_revision != row['audio_revision']
                or text_sha256 != hashlib.sha256(row['text'].encode()).hexdigest()):
            raise ValueError('Alignment belongs to a different transcript revision.')
        if not isinstance(words, list) or not words:
            raise ValueError('Alignment did not provide word positions.')
        previous = row['start_sample']
        for word in words:
            a, b = word['start_sample'], word['end_sample']
            if (type(a) is not int or type(b) is not int or not previous <= a < b <= row['end_sample']
                    or not isinstance(word.get('text'), str) or not word['text'].strip()):
                raise ValueError('Alignment is outside original utterance audio.')
            previous = b
        row['alignment'] = {'machine_revision': revision, 'audio_revision': audio_revision,
                            'text_sha256': text_sha256, 'words': copy.deepcopy(words)}
        return copy.deepcopy(row)

    def attach_activity(self, identity, audio_revision, regions):
        """Retain NVIDIA activity independently; it cannot split Cohere words."""
        row = self.rows[identity]
        if type(audio_revision) is not int or audio_revision != row['audio_revision']:
            raise ValueError('Speaker activity belongs to different utterance audio.')
        cursor = row['start_sample']
        owners = set()
        clean = True
        for region in regions:
            a, b, speakers = region['start_sample'], region['end_sample'], region['speakers']
            if (type(a) is not int or type(b) is not int or not cursor == a < b <= row['end_sample']
                    or not isinstance(speakers, list)
                    or any(not isinstance(s, str) or not s for s in speakers)
                    or len(speakers) != len(set(speakers))):
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
        row['speaker_activity'] = {'audio_revision': audio_revision, 'regions': copy.deepcopy(regions)}
        row['speaker_candidates'] = sorted(owners)
        row['voice_eligible'] = row['state'] == 'sealed' and clean and len(owners) == 1
        return copy.deepcopy(row)
