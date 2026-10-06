"""CPU scheduling and conservative reconciliation for retained live audio.

No model calls or word alignment live here. Window ownership is disjoint;
context may overlap, but context text must never enter a neighboring window.
"""
import copy
import hashlib
import math
import uuid

RATE = 16000
CORE_SAMPLES = 18 * RATE
FAST_MAX_SAMPLES = int(24.5 * RATE)
CONTEXT_SAMPLES = 3 * RATE
MAX_PENDING = 2


def samples(seconds):
    value = float(seconds)
    if not math.isfinite(value) or value < 0:
        raise ValueError('Audio times must be finite and nonnegative.')
    return round(value * RATE)


def anchor_id(start, end, namespace='live'):
    return namespace + '-' + hashlib.sha256(f'{start}:{end}'.encode()).hexdigest()[:24]


def bounds(segment):
    return samples(segment['start']), samples(segment['end'])


def intersects(first, second):
    return first[0] < second[1] and second[0] < first[1]


def uncovered_samples(segment, candidates):
    """Exact candidate coverage gaps; no tolerance or word alignment."""
    a,b=bounds(segment);cursor=a;gaps=[]
    for start,end in sorted(bounds(s) for s in candidates if successful_coverage(s)):
        if end<=cursor or start>=b:continue
        if start>cursor:gaps.append({'start_sample':cursor,'end_sample':min(start,b)})
        cursor=max(cursor,min(end,b))
    if cursor<b:gaps.append({'start_sample':cursor,'end_sample':b})
    return gaps


class RollingPlan:
    """At most two small requests; backlog is a cursor into the saved WAV.

    Acks, rather than dispatch, advance durable completion. After a crash an
    in-flight request becomes queued. Cancel keeps audio and the completed cursor.
    """
    def __init__(self, state=None, *, recover=False):
        self.state = copy.deepcopy(state or {
            'version': 1, 'completed_sample': 0, 'pending': [],
            'last_scheduled_at': 0., 'cancelled': False})
        if (self.state.get('version') != 1 or type(self.state.get('completed_sample')) is not int
                or self.state['completed_sample'] < 0 or len(self.state.get('pending', [])) > MAX_PENDING):
            raise ValueError('Invalid rolling refinement checkpoint.')
        previous = self.state['completed_sample']
        for window in self.state['pending']:
            if (window['start_sample'] != previous
                    or not previous < window['end_sample'] <= previous + CORE_SAMPLES
                    or window['context_end_sample']-window['context_start_sample'] > 24*RATE):
                raise ValueError('Noncontiguous or oversized refinement request.')
            previous = window['end_sample']
            if recover and window['status'] == 'in_flight':window['status'] = 'queued'

    def snapshot(self):
        return copy.deepcopy(self.state)

    def schedule(self, available_seconds, segments, now, force=False):
        available = samples(available_seconds)
        if self.state['cancelled']:return []
        created = []
        while len(self.state['pending']) < MAX_PENDING:
            cursor = (self.state['pending'][-1]['end_sample'] if self.state['pending']
                      else self.state['completed_sample'])
            if available <= cursor:break
            horizon = available if force else max(cursor, available-CONTEXT_SAMPLES)
            target = min(cursor+CORE_SAMPLES, horizon)
            boundaries = sorted({bounds(s)[1] for s in segments
                                 if cursor < bounds(s)[1] <= target})
            full = available >= cursor+CORE_SAMPLES+CONTEXT_SAMPLES
            enough = len(boundaries) >= 3 and now-self.state['last_scheduled_at'] >= 8
            if not (force or full or enough):break
            # Prefer existing utterance boundaries; do not cut a row at a clock tick.
            end = max(boundaries) if boundaries else target
            if force and available <= cursor+CORE_SAMPLES:end = available
            if end <= cursor:break
            # A straddling source row is held for a subsequent window, never split.
            crossing = [bounds(s)[0] for s in segments if bounds(s)[0] < end < bounds(s)[1]]
            if crossing:end = min(end, min(crossing))
            if end <= cursor:break
            window = {'id': anchor_id(cursor, end, 'refine'), 'start_sample': cursor,
                      'end_sample': end, 'context_start_sample': max(0, cursor-CONTEXT_SAMPLES),
                      'context_end_sample': min(available, end+CONTEXT_SAMPLES),
                      'status': 'queued', 'attempts': 0}
            self.state['pending'].append(window);created.append(copy.deepcopy(window))
            self.state['last_scheduled_at'] = float(now)
        return created

    def dispatch(self, live_backlog_seconds=0):
        if self.state['cancelled'] or live_backlog_seconds > 2:return None
        if not self.state['pending'] or any(w['status']=='in_flight' for w in self.state['pending']):return None
        window = self.state['pending'][0]
        window.update(status='in_flight', attempts=window['attempts']+1,operation_id=uuid.uuid4().hex)
        return copy.deepcopy(window)

    def acknowledge(self, window_id, operation_id):
        if (not self.state['pending'] or self.state['pending'][0]['id'] != window_id
                or self.state['pending'][0].get('operation_id') != operation_id
                or self.state['pending'][0]['status'] != 'in_flight'):return False
        window = self.state['pending'].pop(0)
        self.state['completed_sample'] = window['end_sample']
        return True

    def failed(self, window_id, operation_id):
        if (not self.state['pending'] or self.state['pending'][0]['id'] != window_id
                or self.state['pending'][0].get('operation_id') != operation_id):return 'stale'
        window = self.state['pending'][0]
        if window['attempts'] < 2:
            window['status'] = 'queued';return 'retry'
        self.acknowledge(window_id,operation_id)
        return 'unresolved'

    def cancel(self):
        self.state.update(cancelled=True, pending=[])

    def resume(self):
        self.state['cancelled'] = False


def segment_version(segment):
    """Source snapshot guard includes words, timing and speaker ownership."""
    keys = ('id','start','end','speaker','text','machine_revision','protected_fields')
    import json
    value = {k:segment.get(k) for k in keys}
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def successful_non_speech(row):
    """Independent successful audio admission, never empty/failed ASR alone."""
    if row.get('text','').strip() or row.get('transcription_review'):return False
    state=row.get('audio_state');evidence=row.get('acoustic_evidence') or {}
    if state=='digital_silence':return True
    if state=='constant_signal':
        return evidence.get('source')=='pcm_constant' and type(evidence.get('sample_count')) is int and evidence['sample_count']>=320 and isinstance(evidence.get('constant_value'),(int,float)) and math.isfinite(evidence['constant_value'])
    if state=='model_non_speech':
        value=evidence.get('no_speech_probability')
        return evidence.get('source')=='whisper_sot' and isinstance(value,(int,float)) and math.isfinite(value) and .95<=value<=1
    return False

def successful_coverage(row):
    return (bool(row.get('text','').strip()) or successful_non_speech(row) or successful_context_empty(row)) and not row.get('transcription_review')

def successful_context_empty(row):
    """Two complete same-origin decodes establish no additional owned words.

    This is lexical continuity, not a claim of physically silent audio.
    """
    if row.get('text','').strip() or row.get('transcription_review') or row.get('audio_state')!='context_no_new_words':return False
    evidence=row.get('context_evidence') or {};start,end=bounds(row)
    if (evidence.get('source')!='same_origin_prefix' or evidence.get('prefix_complete') is not True
            or evidence.get('extended_complete') is not True or type(evidence.get('start_sample')) is not int
            or not 0<=evidence['start_sample']<start or evidence.get('prefix_end_sample')!=start
            or evidence.get('end_sample')!=end):return False
    reference,extended=evidence.get('reference_text'),evidence.get('extended_text')
    if not isinstance(reference,str) or not isinstance(extended,str):return False
    split=split_same_origin(reference,extended)
    return split is not None and not split[1]

def completed_empty_recognition(row):
    """Successful empty ASR can finish a blank row; it cannot erase old words."""
    review=row.get('transcription_review') or {}
    return not row.get('text','').strip() and review.get('reason')=='empty_result' and review.get('partial_text') is False

def reconcile_window(document, window, expected, candidates, speakers=None):
    """Return a new document plus the previous window revision, without mutating inputs.

    Protected/changed rows and their intersecting candidates remain untouched.
    Empty output cannot erase legible old words. Text is never duplicated across
    ownership windows or assigned by invented word timestamps.
    """
    owned = (window['start_sample'], window['end_sample'])
    maximum = FAST_MAX_SAMPLES if window.get('kind') == 'fast_tail' else CORE_SAMPLES
    if not 0 <= owned[0] < owned[1] or owned[1]-owned[0] > maximum:
        raise ValueError('Invalid refinement ownership window.')
    previous = [copy.deepcopy(s) for s in document['segments'] if intersects(bounds(s),owned)]
    output = copy.deepcopy(document)
    old_ids = {s['id'] for s in document['segments']}
    proposed = copy.deepcopy(candidates)
    for s in proposed:
        a,b = bounds(s)
        if not owned[0] <= a < b <= owned[1] or not isinstance(s.get('text'),str):
            raise ValueError('Refinement text must belong entirely to its audio window.')
    proposed.sort(key=lambda s:bounds(s))
    if any(bounds(a)[1]>bounds(b)[0] for a,b in zip(proposed,proposed[1:])):
        raise ValueError('Refinement candidates must not duplicate overlapping audio.')
    blocked = [s for s in previous if s.get('protected_fields')
               or expected.get(s['id']) != segment_version(s)
               or not owned[0] <= bounds(s)[0] < bounds(s)[1] <= owned[1]]
    # A failed/blank/incomplete candidate does not discard previous usable words.
    def covers(old, replacements):
        replacements = [s for s in replacements if intersects(bounds(s),bounds(old)) and successful_coverage(s)]
        intervals = sorted(bounds(s) for s in replacements)
        cursor = bounds(old)[0]
        for start,end in intervals:
            if start > cursor:break
            cursor = max(cursor,end)
        return cursor >= bounds(old)[1]
    for old in previous:
        if old['text'].strip() and not covers(old,proposed):blocked.append(old)
    blocked_ids = {s['id'] for s in blocked}
    # Dropping a candidate can affect another old row: close the protection set
    # until no remaining row loses its text because it shared that candidate.
    while True:
        accepted = [s for s in proposed if not any(intersects(bounds(s),bounds(b)) for b in blocked)]
        newly = [s for s in previous if s['id'] not in blocked_ids and s['text'].strip()
                 and not covers(s,accepted)]
        if not newly:break
        blocked.extend(newly);blocked_ids.update(s['id'] for s in newly)
    removed = {s['id'] for s in previous if s['id'] not in blocked_ids}
    output['segments'] = [s for s in output['segments'] if s['id'] not in removed]
    used = {s['id'] for s in output['segments']}
    for candidate in accepted:
        eligible = [s for s in previous if s['id'] in removed and s['id'] not in used
                    and intersects(bounds(s),bounds(candidate))]
        eligible.sort(key=lambda s:min(bounds(s)[1],bounds(candidate)[1])-max(bounds(s)[0],bounds(candidate)[0]),reverse=True)
        old = eligible[0] if eligible else None
        sid = old['id'] if old else anchor_id(*bounds(candidate),'refined')
        # Distinct ownership windows cannot manufacture a collision with another row.
        if sid in used or (not old and sid in old_ids):raise ValueError('Refinement ID collision.')
        fast = window.get('kind') == 'fast_tail'
        candidate.update(id=sid,machine_revision=(old.get('machine_revision',0)+1 if old else 1),
                         refinement_state=candidate.get('refinement_state','provisional') if fast else ('refined' if successful_coverage(candidate) or completed_empty_recognition(candidate) else 'unresolved'),
                         refinement_window=window['id'],finalized=bool(candidate.get('finalized')) if fast else True)
        candidate['audio_anchor'] = (copy.deepcopy(old.get('audio_anchor')) if old and bounds(old)==bounds(candidate) and old.get('audio_anchor')
                                     else {'start_sample':bounds(candidate)[0],'end_sample':bounds(candidate)[1]})
        output['segments'].append(candidate);used.add(sid)
    for key,name in (speakers or {}).items():output['speakers'].setdefault(key,name)
    output['segments'].sort(key=lambda s:bounds(s))
    return {'document':output,'previous_revision':{'window_id':window['id'],'segments':previous},
            'protected_ids':sorted(blocked_ids),'changed':output!=document,
            'coverage_gaps':{s['id']:uncovered_samples(s,proposed) for s in blocked}}


SINGLE_CARDINAL_WORDS = dict(zip(
    'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(),
    range(20)))
SINGLE_CARDINAL_WORDS.update(dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(),range(20,100,10))))
FRACTION_DENOMINATORS = frozenset(
    'half halves quarter quarters second third fourth fifth sixth seventh eighth ninth tenth '
    'eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth '
    'twentieth thirtieth fortieth fiftieth sixtieth seventieth eightieth ninetieth hundredth '
    'thousandth millionth'.split())

def split_same_origin(previous_text, extended_text):
    """Split a growing decode by matching its SAME audio-origin text prefix.

    This is lexical continuity, not word timestamps. A moving-window suffix
    match is deliberately forbidden: it can delete genuinely repeated phrases.
    Uncertain matching returns None; the full candidate remains available.
    """
    import difflib
    import re
    previous = list(re.finditer(r'\w+|[^\w\s]', previous_text))
    extended = list(re.finditer(r'\w+|[^\w\s]', extended_text))
    if not previous or not extended:return None
    old = [m.group().casefold() for m in previous]
    new = [m.group().casefold() for m in extended]
    def cardinal(token):
        if token in SINGLE_CARDINAL_WORDS:return SINGLE_CARDINAL_WORDS[token]
        # Leading-zero identifiers and non-ASCII/compound number forms retain
        # literal spelling. This is not a general numeric text normalizer.
        if len(token)<=2 and re.fullmatch(r'0|[1-9][0-9]*',token) and int(token) in SINGLE_CARDINAL_WORDS.values():return int(token)
        return None
    def isolated(matches,index,text):
        match=matches[index];before=text[:match.start()];after=text[match.end():]
        if before and before[-1] in '.+-−–—/:':return False
        if after and after[0] in '+-−–—/:':return False
        if len(after)>1 and after[0] in '.,' and after[1].isalnum():return False
        token=match.group().casefold()
        following=matches[index+1].group().casefold() if index+1<len(matches) else None
        preceding=matches[index-1].group().casefold() if index else None
        if following in ('/','+','-','−','–','—',':') or preceding in ('/','+','-','−','–','—',':'):return False
        # Do not split a changed compound value into a prefix plus a new word.
        if following in ('point','hundred','thousand','dozen','score','gross') or (following and re.fullmatch(r'[a-z]+illion',following)):return False
        def denominator(token):
            return token in FRACTION_DENOMINATORS or token.removesuffix('s') in FRACTION_DENOMINATORS
        if following and denominator(following):return False
        if following=='and' and index+2<len(matches):
            fraction_index=index+2
            fraction=matches[fraction_index].group().casefold()
            if fraction in ('a','an') or cardinal(fraction) is not None:
                fraction_index+=1
            if fraction_index<len(matches) and denominator(matches[fraction_index].group().casefold()):return False
        value=cardinal(token)
        if value in range(20,100,10) and cardinal(following or '') in range(1,10):return False
        if value in range(1,10) and cardinal(preceding or '') in range(20,100,10):return False
        return True
    def equivalent(index):
        if old[index]==new[index]:return True
        first,second=cardinal(old[index]),cardinal(new[index])
        return (first is not None and first==second and old[index].isascii() and new[index].isascii()
                and old[index].isdigit()!=new[index].isdigit()
                and isolated(previous,index,previous_text) and isolated(extended,index,extended_text))
    lexical=next((i for i in range(len(old)-1,-1,-1) if re.search(r'\w',old[i])),None)
    if lexical is None:return None
    punctuation=old[lexical+1:];old=old[:lexical+1]
    same = 0
    while same < min(len(old),len(new)) and equivalent(same):same += 1
    cut = same
    if same < len(old):
        # Only the unfinished last lexical token may be expanded/corrected.
        if (same != len(old)-1 or same >= len(new)
                or old[same] in SINGLE_CARDINAL_WORDS or new[same] in SINGLE_CARDINAL_WORDS
                or any(c.isdigit() for c in old[same]+new[same])
                or len(old[same]) < 3 or old[same][:3] != new[same][:3]
                or difflib.SequenceMatcher(None,old[same],new[same]).ratio() < .5):return None
        cut += 1
    if cut == 0:return None
    for token in punctuation:
        if cut<len(new) and new[cut]==token:cut+=1
        else:break
    position = extended[cut-1].end()
    return extended_text[:position].strip(), extended_text[position:].strip()
