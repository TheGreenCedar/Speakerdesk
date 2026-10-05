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


def reconcile_window(document, window, expected, candidates, speakers=None):
    """Return a new document plus the previous window revision, without mutating inputs.

    Protected/changed rows and their intersecting candidates remain untouched.
    Empty output cannot erase legible old words. Text is never duplicated across
    ownership windows or assigned by invented word timestamps.
    """
    owned = (window['start_sample'], window['end_sample'])
    if not 0 <= owned[0] < owned[1] or owned[1]-owned[0] > CORE_SAMPLES:
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
        replacements = [s for s in replacements if intersects(bounds(s),bounds(old)) and s['text'].strip()]
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
        candidate.update(id=sid,machine_revision=(old.get('machine_revision',0)+1 if old else 1),
                         refinement_state='refined' if candidate['text'].strip() else 'unresolved',
                         refinement_window=window['id'],finalized=True)
        candidate['audio_anchor'] = (copy.deepcopy(old.get('audio_anchor')) if old and old.get('audio_anchor')
                                     else {'start_sample':bounds(candidate)[0],'end_sample':bounds(candidate)[1]})
        output['segments'].append(candidate);used.add(sid)
    for key,name in (speakers or {}).items():output['speakers'].setdefault(key,name)
    output['segments'].sort(key=lambda s:bounds(s))
    return {'document':output,'previous_revision':{'window_id':window['id'],'segments':previous},
            'protected_ids':sorted(blocked_ids),'changed':output!=document}
