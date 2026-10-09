"""Causal development source selector; native fallback is never rewritten.

No speech/identity claim or model execution. Production admission is unqualified.
Every output uses only a path observed before that output's first sample.
"""
import hashlib
import json
import numpy as np

RATE = 16000
WINDOW = 2000
MAX_LAG = 8000
HISTORY = MAX_LAG+3*WINDOW
LIFETIME = 30*RATE
PARAMETERS = dict(policy='causal_scalar_render_innovation_v1', sample_rate=RATE,
    window_samples=WINDOW, maximum_causal_lag_samples=MAX_LAG, history_samples=HISTORY,
    minimum_contiguous_windows=3, coherence_floor=.995, competing_peak_tolerance=1e-6,
    lag_spread_samples=2, maximum_absolute_gain=1, heldout_energy_fraction=1e-4,
    lifetime_samples=LIFETIME, established_gain_adaptation=False, established_delay_adaptation=False,
    startup='retain exact native output before preceding support exists',
    route='reset on explicit epoch/gap; conflicting delay declines; only same frozen path can requalify')
POLICY_SHA256 = hashlib.sha256(json.dumps(PARAMETERS,sort_keys=True).encode()).hexdigest()


def observation(raw, reference, a, b):
    y = np.diff(raw[a:b].astype(np.float64))
    x = np.diff(reference[a-MAX_LAG:b].astype(np.float64))
    ey = float(y@y)
    if ey <= 1e-8:return None
    size = 1 << (len(x)+len(y)-1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(x,size)*np.conj(np.fft.rfft(y,size)),size)[:MAX_LAG+1]
    sums = np.concatenate(([0.],np.cumsum(x*x)))
    energy = sums[len(y):]-sums[:-len(y)]
    score = np.abs(corr)/np.sqrt(np.maximum(energy*ey,1e-30)); score[energy<=1e-8]=0
    peak = int(np.argmax(score))
    if score[peak]<.995 or np.count_nonzero(score>=score[peak]-1e-6)!=1:return None
    lag = MAX_LAG-peak; ref=reference[a-lag:b-lag].astype(np.float64)
    gain=float(ref@raw[a:b]/(ref@ref))
    return dict(lag=lag,gain=gain) if np.isfinite(gain) and 0<abs(gain)<=1 else None


class CausalInnovation:
    """Packet-independent causal output; clipping/gaps/reset preserve fallback.

    A scalar model can qualify only simple stable paths. Multipath, collinear
    near during initial support, and unseen route changes remain limitations.
    No same-context refit, future read, amplitude gate or text deduplication.
    """
    def __init__(self):
        self.end=0; self.epoch=None; self.path=None; self.origin=0
        self.frozen=None
        self.raw=np.empty(0,np.float32); self.reference=np.empty((0,2),np.float32)
        self.covered=np.empty(0,bool)

    def reset(self, epoch, at):
        self.epoch=epoch; self.path=None; self.origin=at
        self.frozen=None
        self.raw=np.empty(0,np.float32); self.reference=np.empty((0,2),np.float32)
        self.covered=np.empty(0,bool)

    def _candidate(self):
        if len(self.raw)<HISTORY or not self.covered.all():return None
        paths=[]; end=len(self.raw)
        for channel in range(2):
            if self.frozen and channel!=self.frozen['channel']:continue
            ref=self.reference[:,channel]
            observations=[observation(self.raw,ref,end-3*WINDOW+i*WINDOW,end-2*WINDOW+i*WINDOW) for i in range(3)]
            if any(o is None for o in observations):continue
            lags=[o['lag'] for o in observations]
            if max(lags)-min(lags)>2:continue
            lag=round(float(np.median(lags))); a,b=end-3*WINDOW,end-WINDOW
            x=ref[a-lag:b-lag].astype(np.float64); y=self.raw[a:b].astype(np.float64)
            gain=float(x@y/(x@x))
            if not np.isfinite(gain) or not 0<abs(gain)<=1:continue
            if self.frozen:
                if abs(lag-self.frozen['lag'])>2:continue
                # Within measurement tolerance, evaluate the ORIGINAL path.
                # Same-epoch lag adaptation can fit genuine near repeats.
                lag=self.frozen['lag'];gain=self.frozen['gain']
            a,b=end-WINDOW,end; y=self.raw[a:b].astype(np.float64)
            error=y-gain*ref[a-lag:b-lag]
            if float(error@error/max(float(y@y),1e-30))>1e-4:continue
            paths.append(dict(channel=channel,lag=lag,gain=gain,observed_end=self.end,
                support_start=self.end-3*WINDOW,valid_until=self.end+LIFETIME))
        if len(paths)==2 and not (np.array_equal(self.reference[:,0],self.reference[:,1])
                                 or np.array_equal(self.reference[:,0],-self.reference[:,1])):
            return None
        return paths[0] if paths else None

    def _observe(self):
        if self.path and len(self.raw)>=MAX_LAG+WINDOW and self.covered[-MAX_LAG-WINDOW:].all():
            channel=self.path['channel']; o=observation(self.raw,self.reference[:,channel],len(self.raw)-WINDOW,len(self.raw))
            if o and abs(o['lag']-self.path['lag'])>2:self.path=None
        if self.path is None:
            self.path=self._candidate()
            if self.path and self.frozen is None:
                self.frozen={key:self.path[key] for key in ('channel','gain','lag')}

    def process(self, start, epoch, raw, reference, native, covered):
        raw=np.asarray(raw,dtype=np.float32); ref=np.asarray(reference,dtype=np.float32)
        native=np.asarray(native,dtype=np.float32); covered=np.asarray(covered,dtype=bool)
        if (type(start) is not int or start!=self.end or type(epoch) is not int or epoch<0
            or raw.ndim!=1 or ref.shape!=(len(raw),2) or native.shape!=raw.shape or covered.shape!=raw.shape
            or not np.isfinite(native).all()):raise ValueError('Innovation source clock/shape differs.')
        if epoch!=self.epoch:self.reset(epoch,start)
        valid=covered & np.isfinite(raw) & np.isfinite(ref).all(axis=1)
        valid &= (np.abs(raw)<32767/32768) & (np.abs(ref)<32767/32768).all(axis=1)
        output=native.copy(); decisions=[]; offset=0
        while offset<len(raw):
            count=min(WINDOW-self.end%WINDOW,len(raw)-offset)
            changed=np.flatnonzero(valid[offset:offset+count]!=valid[offset])
            if len(changed):count=int(changed[0])
            a,b=offset,offset+count; first=self.end
            known=bool(valid[a])
            if not known:self.reset(epoch,first)
            used=None
            reason=('unsupported_startup' if self.frozen is None else 'unsupported_frozen_path') if self.path is None else 'innovation'
            if not known:reason='uncovered_nonfinite_or_clipped'
            elif self.path and self.path['valid_until']<first+count:
                self.path=None; reason='expired_support'
            if known and self.path:
                p=self.path; lag=p['lag']; combined=np.concatenate((self.reference[:,p['channel']],ref[a:b,p['channel']]))
                begin=len(self.reference)-lag; end=begin+count
                if begin>=0 and p['observed_end']<=first:
                    output[a:b]=(raw[a:b].astype(np.float64)-p['gain']*combined[begin:end]).astype(np.float32)
                    used=dict(p)
                else:reason='missing_causal_reference'
            decisions.append(dict(start_sample=first,end_sample=first+count,epoch=epoch,
                method='innovation' if used else 'native_fallback',reason=reason,path=used,policy_sha256=POLICY_SHA256))
            self.raw=np.concatenate((self.raw,raw[a:b])); self.reference=np.concatenate((self.reference,ref[a:b]))
            self.covered=np.concatenate((self.covered,covered[a:b] & known)); self.end+=count
            if len(self.raw)>HISTORY:
                trim=len(self.raw)-HISTORY; self.raw=self.raw[trim:]; self.reference=self.reference[trim:]
                self.covered=self.covered[trim:]; self.origin+=trim
            if self.end%WINDOW==0:self._observe()
            offset=b
        return output,decisions


class InnovationSourceSink:
    """Bind aligned raw/reference evidence to emitted native source samples.

    Native and selected mic remain separate arrays. CaptureSink writes both;
    ordinary preview/voice continue to select the catalog-bound retained input.
    """
    def __init__(self,sink):
        self.sink=sink; self.selector=CausalInnovation(); self.pending=[]; self.queued=0

    def retain(self,tracks,start,epoch):
        count=len(tracks['microphone'])
        ref=tracks['system_reference']
        covered=tracks['microphone_present'] & tracks['system_present']
        render_epochs=np.asarray(tracks.get('system_reference_epoch',np.full(count,-1,dtype=np.int64)))
        if render_epochs.shape!=(count,) or render_epochs.dtype.kind not in 'iu':
            raise ValueError('Invalid render reference epoch evidence.')
        self.pending.append([start,epoch,tracks['microphone'].copy(),ref.copy(),covered.copy(),
            render_epochs.copy(),tracks['system_present'].copy()]); self.queued+=count
        if self.queued>RATE*2:raise ValueError('Innovation source evidence exceeded bound.')

    def emit(self,mixed,tracks):
        count=len(tracks['microphone_clean']); offset=0; values=[]; decisions=[]; references=[]; coverage=[]
        render_epochs=[];render_coverage=[]
        while offset<count:
            if not self.pending:raise ValueError('Missing retained innovation reference.')
            start,epoch,raw,ref,covered,epochs,present=self.pending[0]; size=min(count-offset,len(raw))
            if not np.array_equal(raw[:size],tracks['microphone'][offset:offset+size]):
                raise ValueError('Innovation raw/native source binding differs.')
            value,rows=self.selector.process(start,epoch,raw[:size],ref[:size],
                tracks['microphone_clean'][offset:offset+size],covered[:size])
            references.append(ref[:size]); coverage.append(covered[:size])
            render_epochs.append(epochs[:size]);render_coverage.append(present[:size])
            values.append(value); decisions.extend(rows); self.queued-=size; offset+=size
            if size==len(raw):self.pending.pop(0)
            else:self.pending[0]=[start+size,epoch,raw[size:],ref[size:],covered[size:],epochs[size:],present[size:]]
        selected=np.concatenate(values)
        selected_mix=np.clip(tracks['system']+selected,-1,1).astype('<f4')
        self.sink(selected_mix,{**tracks,'microphone_native':tracks['microphone_clean'].copy(),
            'microphone_clean':selected,'innovation_decisions':decisions,
            'innovation_native_mix':mixed.copy(),'innovation_selected_mix':selected_mix,
            'innovation_reference':np.concatenate(references),'innovation_coverage':np.concatenate(coverage),
            'innovation_render_epochs':np.concatenate(render_epochs),'innovation_render_coverage':np.concatenate(render_coverage)})


class InnovationReceiptWriter:
    """Local retained native PCM and selected-input receipts on one clock.

    Receipt hashes bind the physical PCM16 representation, not a claimed absence
    of human speech. Raw/stereo evidence remains in its original capture files.
    This writer does not authorize the development selector for production.
    """
    parameters=PARAMETERS
    policy_sha256=POLICY_SHA256

    def validate_decision(self,row):
        if (row['policy_sha256']!=self.policy_sha256
            or (row['path'] and row['path']['observed_end']>row['start_sample'])):
            raise ValueError('Innovation decision does not precede retained output.')

    def __init__(self,folder):
        import wave
        from pathlib import Path
        self.folder=Path(folder); self.end=0; self.closed=False
        self.handle=(self.folder/'microphone_native.wav').open('x+b')
        self.native=wave.open(self.handle,'wb'); self.native.setparams((1,2,RATE,0,'NONE','none'))
        self.mix_handle=None; self.native_mix=None
        try:
            self.mix_handle=(self.folder/'audio_native.wav').open('x+b')
            self.native_mix=wave.open(self.mix_handle,'wb'); self.native_mix.setparams((1,2,RATE,0,'NONE','none'))
            self.receipts=(self.folder/'innovation-receipts.jsonl').open('x')
        except Exception:
            if self.native_mix:self.native_mix.close()
            if self.mix_handle:self.mix_handle.close()
            self.native.close(); self.handle.close(); raise
        self.digests={name:hashlib.sha256() for name in
            ('microphone','system','system_reference','microphone_native','microphone_clean','audio_native','audio')}

    def write(self,start,tracks):
        from audio import pcm16_bytes
        count=len(tracks['microphone_clean']); end=start+count
        rows=tracks['innovation_decisions']; position=start
        if self.closed or start!=self.end or count<=0:raise ValueError('Innovation retention clock differs.')
        for row in rows:
            if row['start_sample']!=position or row['end_sample']<=position:
                raise ValueError('Innovation decision does not precede retained output.')
            self.validate_decision(row)
            position=row['end_sample']
        if position!=end:raise ValueError('Innovation decisions do not cover retained output.')
        source={**tracks,'system_reference':tracks['innovation_reference'],
            'audio_native':tracks['innovation_native_mix'],'audio':tracks['innovation_selected_mix']}
        payload={}
        for name in self.digests:
            array=np.asarray(source[name])
            shape=(count,2) if name=='system_reference' else (count,)
            if array.shape!=shape:raise ValueError('Innovation retention input shape differs.')
            payload[name]=pcm16_bytes(array.reshape(-1))
        row=dict(start_sample=start,end_sample=end,policy_sha256=self.policy_sha256,
            pcm_sha256={name:hashlib.sha256(data).hexdigest() for name,data in payload.items()},
            covered_samples=int(np.count_nonzero(tracks['innovation_coverage'])),
            coverage_sha256=hashlib.sha256(tracks['innovation_coverage'].tobytes()).hexdigest(),decisions=rows)
        self.native.writeframes(payload['microphone_native'])
        self.native_mix.writeframes(payload['audio_native'])
        self.receipts.write(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n')
        for name,data in payload.items():self.digests[name].update(data)
        self.handle.flush(); self.mix_handle.flush(); self.receipts.flush(); self.end=end

    def close(self):
        if self.closed:return
        self.native.close(); self.handle.close(); self.native_mix.close(); self.mix_handle.close()
        self.receipts.close(); self.closed=True
        receipt_digest=hashlib.sha256()
        with (self.folder/'innovation-receipts.jsonl').open('rb') as receipts:
            for chunk in iter(lambda:receipts.read(65536),b''):receipt_digest.update(chunk)
        summary=dict(schema_version=1,samples=self.end,policy_sha256=self.policy_sha256,
            parameters=self.parameters,pcm_sha256={name:d.hexdigest() for name,d in self.digests.items()},
            receipts_sha256=receipt_digest.hexdigest(),
            production_admissible=False)
        (self.folder/'innovation-evidence.json').write_text(json.dumps(summary,sort_keys=True)+'\n')
