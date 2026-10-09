"""Opt-in bounded startup waveform hypothesis; no speech-absence claim."""
import hashlib
import json
import threading
import time
import numpy as np
from render_innovation import MAX_LAG, WINDOW, PARAMETERS as FORWARD_PARAMETERS, InnovationSourceSink, InnovationReceiptWriter

MAX_HOLD = 32000
MIN_PREFIX_SAMPLES = 512
PARAMETERS = dict(policy='bounded_continuous_render_history_startup_v2',
    maximum_hold_samples=MAX_HOLD, window_samples=WINDOW,
    minimum_prefix_samples=MIN_PREFIX_SAMPLES, maximum_lag_samples=MAX_LAG,
    coherence_floor=.995, competing_peak_tolerance=1e-6, lag_tolerance_samples=2,
    fixed_gain_residual_energy_fraction=1e-4, prefix_gain_refit=False,
    unsupported='native', unknown_pre_recording_reference='native',
    maximum_render_history_samples=MAX_LAG,
    render_history='contiguous covered same explicit render epoch only; microphone epoch never supplies render continuity',
    maximum_wall_hold_seconds=2., forward_policy=FORWARD_PARAMETERS,
    publication='immutable selected bytes shared by live and saved; no prefix revisions')
POLICY_SHA256 = hashlib.sha256(json.dumps(PARAMETERS,sort_keys=True).encode()).hexdigest()


def available_observation(raw, reference, a, b, *, reference_offset=0):
    """Search only retained causal lags; never pad unknown prefix history."""
    if b-a<MIN_PREFIX_SAMPLES:return None
    maximum=min(MAX_LAG,a+reference_offset)
    y=np.diff(raw[a:b].astype(np.float64))
    x=np.diff(reference[a+reference_offset-maximum:b+reference_offset].astype(np.float64))
    ey=float(y@y)
    if ey<=1e-8:return None
    size=1<<(len(x)+len(y)-1).bit_length()
    corr=np.fft.irfft(np.fft.rfft(x,size)*np.conj(np.fft.rfft(y,size)),size)[:maximum+1]
    sums=np.concatenate(([0.],np.cumsum(x*x)))
    energy=sums[len(y):]-sums[:-len(y)]
    score=np.abs(corr)/np.sqrt(np.maximum(energy*ey,1e-30));score[energy<=1e-8]=0
    peak=int(np.argmax(score))
    if score[peak]<.995 or np.count_nonzero(score>=score[peak]-1e-6)!=1:return None
    return maximum-peak


def verify_prefix(raw, reference, native, covered, path, *, history=None):
    """Transform an unpublished prefix using frozen support, window by window.

    Raw arrays begin at the microphone epoch's origin. Explicitly continuous
    render history may precede it; a reset/gap/unknown render epoch cannot.
    The caller must release by the
    sample/time cap, preserve native evidence, and bind published/saved bytes.
    This function does not infer epochs or fit a new gain on prefix samples.
    """
    raw=np.asarray(raw,dtype=np.float32);ref=np.asarray(reference,dtype=np.float32)
    native=np.asarray(native,dtype=np.float32);covered=np.asarray(covered,dtype=bool)
    n=len(raw)
    if raw.ndim!=1 or ref.shape!=(n,2) or native.shape!=(n,) or covered.shape!=(n,) or n>MAX_HOLD:
        raise ValueError('Startup prefix shape or hold bound differs.')
    if not np.isfinite(native).all():raise ValueError('Startup native prefix is nonfinite.')
    if path and (type(path.get('lag')) is not int or not 0<=path['lag']<=MAX_LAG
                or type(path.get('channel')) is not int or path['channel'] not in (0,1)
                or not np.isfinite(path.get('gain',np.nan)) or not 0<abs(path['gain'])<=1):
        raise ValueError('Startup frozen path differs.')
    h=0;reference_covered=covered
    if history is not None:
        prior=np.asarray(history['reference'],dtype=np.float32);known=np.asarray(history['covered'],dtype=bool)
        h=len(prior)
        if (not 0<h<=MAX_LAG or prior.shape!=(h,2) or known.shape!=(h,)
            or type(history['render_epoch']) is not int or history['render_epoch']<0
            or type(history['start_sample']) is not int or type(history['end_sample']) is not int
            or not 0<=history['start_sample']<history['end_sample'] or history['end_sample']-history['start_sample']!=h
            or not known.all() or not np.isfinite(prior).all() or not (np.abs(prior)<32767/32768).all()):
            raise ValueError('Invalid continuous render prefix history.')
        ref=np.concatenate((prior,ref));reference_covered=np.concatenate((known,covered))
    out=native.copy();decisions=[]
    for start in range(0,n,WINDOW):
        end=min(start+WINDOW,n);a=start;accepted=False;reason='unsupported_startup'
        if path:
            lag=path['lag'];channel=path['channel'];a=max(start,lag-h)
            if a>start:decisions.append(dict(start_sample=start,end_sample=min(a,end),method='native_fallback',reason='unknown_prefix_reference'))
            if a>=end:continue
            maximum=min(MAX_LAG,a+h)
            valid=covered[a:end].all() and reference_covered[a+h-maximum:end+h].all()
            valid=valid and np.isfinite(raw[a:end]).all() and np.isfinite(ref[a+h-maximum:end+h]).all()
            valid=valid and (np.abs(raw[a:end])<32767/32768).all() and (np.abs(ref[a+h-maximum:end+h])<32767/32768).all()
            reason='unsupported_prefix_path'
            measured=available_observation(raw,ref[:,channel],a,end,reference_offset=h) if valid else None
            if measured is not None and abs(measured-lag)<=2:
                y=raw[a:end].astype(np.float64)
                residual=y-path['gain']*ref[a+h-lag:end+h-lag,channel]
                if float(residual@residual/max(float(y@y),1e-30))<=1e-4:
                    out[a:end]=residual.astype(np.float32);accepted=True;reason='verified_available_prefix'
        decisions.append(dict(start_sample=a,end_sample=end,
            method='startup_innovation' if accepted else 'native_fallback',reason=reason))
    return out,decisions


class StartupSourceSink(InnovationSourceSink):
    """Hold only an unpublished epoch prefix; publish/save one immutable source.

    A timer releases unsupported native bytes within2 wall seconds. Capture
    epochs, gaps and EOF also flush native immediately. Forward selection is
    otherwise unchanged. There is no text editing or later PCM rewrite.
    """
    def __init__(self,sink,*,lock=None):
        self.publish=sink;self.gate_lock=lock if lock is not None else threading.RLock();self.parts=[]
        self.gate_epoch=None;self.origin=0;self.held_count=0;self.open=False
        self.timer=None;self.started=None;self.publication_log=[];self.generation=0;self.error=None
        self.render_history=None;self.gate_history=None
        super().__init__(self._hold)

    def _cancel(self):
        self.generation+=1
        if self.timer:self.timer.cancel();self.timer=None

    def _timeout(self,generation):
        with self.gate_lock:
            if generation!=self.generation or not self.open or time.monotonic()-self.started<2.:return
            try:self._release(None,'wall_timeout')
            except Exception as exc:self.error=exc

    def flush(self):
        with self.gate_lock:
            if self.error:raise self.error
            self._release(None,'epoch_or_EOF')

    def _publish(self,tracks):
        selected=tracks['microphone_clean']
        mix=np.clip(tracks['system']+selected,-1,1).astype('<f4')
        tracks['innovation_selected_mix']=mix
        self.publish(mix,tracks)

    def _history_at(self,start,epochs):
        history=self.render_history
        if (history is None or history['end_sample']!=start or not len(epochs)
            or epochs[0]<0 or not np.all(epochs==history['render_epoch'])):return None
        return {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in history.items()}

    def _remember_reference(self,piece,start):
        ref=piece['innovation_reference'];epochs=piece['innovation_render_epochs']
        known=piece['innovation_render_coverage'] & np.isfinite(ref).all(axis=1)
        known &= (np.abs(ref)<32767/32768).all(axis=1)
        if not len(epochs) or epochs[0]<0 or not np.all(epochs==epochs[0]):
            self.render_history=None;return
        epoch=int(epochs[0]);history=self.render_history
        # Only the valid suffix after a gap is available for a later boundary.
        missing=np.flatnonzero(~known);first=int(missing[-1])+1 if len(missing) else 0
        if first==len(ref):self.render_history=None;return
        prior=(history['reference'] if first==0 and history is not None
            and history['end_sample']==start and history['render_epoch']==epoch else np.empty((0,2),np.float32))
        retained=np.concatenate((prior,ref[first:]))[-MAX_LAG:].copy();end=start+len(ref)
        self.render_history=dict(reference=retained,covered=np.ones(len(retained),bool),render_epoch=epoch,
            start_sample=end-len(retained),end_sample=end)

    def _release(self,path,reason):
        if not self.parts:self._cancel();self.open=False;self.gate_history=None;return
        merged={key:np.concatenate([p[key] for p in self.parts]) for key in self.parts[0] if key!='innovation_decisions'}
        n=self.held_count;publication_end=self.origin+n
        relative=None
        if path and path['observed_end']<=publication_end:
            relative=dict(path)
            # Predictor lag is relative to this epoch; observed support retains
            # absolute recording timestamps in the retained receipt.
        history=self.gate_history
        if history and not np.all(merged['innovation_render_epochs']==history['render_epoch']):history=None
        selected,rows=verify_prefix(merged['microphone'],merged['innovation_reference'],
            merged['microphone_native'],merged['innovation_coverage'],relative,history=history)
        merged['microphone_clean']=selected
        for r in rows:
            epochs=merged['innovation_render_epochs'][r['start_sample']:r['end_sample']]
            r['render_epoch']=int(epochs[0]) if np.all(epochs==epochs[0]) else -1
            if r['method']=='startup_innovation' and r['start_sample']-path['lag']<0 and history:
                from audio import pcm16_bytes
                r['reference_history']={k:history[k] for k in ('start_sample','end_sample','render_epoch')}
                r['reference_history'].update(pcm_sha256=hashlib.sha256(pcm16_bytes(history['reference'].reshape(-1))).hexdigest(),
                    float32_sha256=hashlib.sha256(history['reference'].astype('<f4').tobytes()).hexdigest())
            r.update(start_sample=r['start_sample']+self.origin,end_sample=r['end_sample']+self.origin,
                epoch=self.gate_epoch,policy_sha256=POLICY_SHA256,
                path=dict(path) if r['method']=='startup_innovation' else None,
                publication_source_end=publication_end,startup_origin=self.origin,
                release_reason=reason)
        merged['innovation_decisions']=rows
        self.publication_log.append(dict(epoch=self.gate_epoch,start_sample=self.origin,end_sample=publication_end,
            reason=reason,hold_wall_seconds=max(0.,time.monotonic()-self.started),
            held_samples=n,policy_sha256=POLICY_SHA256))
        self.parts=[];self.held_count=0;self.open=False;self.gate_history=None;self._cancel()
        self._publish(merged)

    def _hold(self,mixed,tracks):
        with self.gate_lock:
            if self.error:raise self.error
            # Split at epoch boundaries before concatenating held evidence.
            rows=tracks['innovation_decisions'];position=0
            while position<len(tracks['microphone_clean']):
                row=next(r for r in rows if r['start_sample']<=rows[0]['start_sample']+position<r['end_sample'])
                epoch=row['epoch'];start=rows[0]['start_sample']+position
                end=max(r['end_sample'] for r in rows if r['epoch']==epoch and r['start_sample']>=start)
                size=end-start
                piece={k:v[position:position+size].copy() for k,v in tracks.items() if isinstance(v,np.ndarray)}
                piece_rows=[dict(r) for r in rows if start<=r['start_sample']<end]
                if self.gate_epoch!=epoch:
                    self._release(None,'epoch_change');self.gate_epoch=epoch;self.origin=start
                    self.gate_history=self._history_at(start,piece['innovation_render_epochs'])
                    self.open=True;self.started=time.monotonic()
                    self.generation+=1;generation=self.generation
                    self.timer=threading.Timer(2.,lambda generation=generation:self._timeout(generation))
                    self.timer.daemon=True;self.timer.start()
                self._remember_reference(piece,start)
                if self.open:
                    if time.monotonic()-self.started>=2.:self._release(None,'wall_timeout')
                if self.open:
                    available=MAX_HOLD-self.held_count;take=min(available,size)
                    path=self.selector.path
                    if path and start<path['observed_end']<=start+take:
                        take=path['observed_end']-start
                    self.parts.append({k:v[:take] for k,v in piece.items()});self.held_count+=take
                    if not piece['innovation_coverage'][:take].all():self._release(None,'reference_gap')
                    elif path and path['observed_end']<=start+take:self._release(path,'qualified_path')
                    elif self.held_count==MAX_HOLD:self._release(None,'source_cap')
                    if take<size:
                        piece={k:v[take:] for k,v in piece.items()};piece_rows=[dict(r) for r in piece_rows if r['end_sample']>start+take]
                        piece_rows[0]['start_sample']=start+take
                    else:piece=None
                if piece:
                    for r in piece_rows:r['policy_sha256']=POLICY_SHA256
                    piece['innovation_decisions']=piece_rows;self._publish(piece)
                position+=size


class StartupReceiptWriter(InnovationReceiptWriter):
    parameters=PARAMETERS
    policy_sha256=POLICY_SHA256

    def validate_decision(self,row):
        if row['policy_sha256']!=self.policy_sha256:raise ValueError('Startup receipt policy differs.')
        if row['method']=='startup_innovation':
            end=row.get('publication_source_end');origin=row.get('startup_origin');p=row.get('path')
            if (type(end) is not int or type(origin) is not int or not origin<=row['start_sample']<row['end_sample']<=end
                or end-origin>MAX_HOLD or not p or not origin<=p['support_start']<p['observed_end']<=end):
                raise ValueError('Startup support is outside unpublished bounded prefix.')
            history=row.get('reference_history')
            if history and (type(history.get('start_sample')) is not int or type(history.get('end_sample')) is not int
                or not 0<=history['start_sample']<history['end_sample']==origin
                or origin-history['start_sample']>MAX_LAG or row['start_sample']-p['lag']<history['start_sample']
                or type(history.get('render_epoch')) is not int or history['render_epoch']<0
                or history['render_epoch']!=row.get('render_epoch')
                or any(not isinstance(history.get(key),str) or len(history[key])!=64 for key in ('pcm_sha256','float32_sha256'))):
                raise ValueError('Startup continuous render history binding differs.')
        elif row['path'] and row['path']['observed_end']>row['start_sample']:
            raise ValueError('Forward startup source uses future support.')
