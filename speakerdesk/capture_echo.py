"""Explicit-reference acoustic echo cancellation on aligned capture samples.

CPU DSP only. The reference is never rendered. Original tracks remain intact;
only the microphone contribution to the canonical mix is processed.
"""
import ctypes
import sys
from pathlib import Path

import numpy as np

RATE = 16000
FRAME = 160
# Pinned AEC3 linear output has one 64-sample block-framing delay.
DELAY = 64
LOOKAHEAD = RATE//2


def library_path():
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)/'libspeakerdesk_echo.dylib'
    return Path(__file__).resolve().parents[1]/'desktop/capture/echo/libspeakerdesk_echo.dylib'


class ReferenceClock:
    """Bounded broadband correlation validates echo and estimates clock skew.

    Speech harmonics alone must not activate cancellation. First differences
    emphasize broadband excitation. A 125 ms window searches
    speaker latency up to half a second; only strong, consistent matches update
    the drift estimate. Weak double-talk matches retain the last reliable rate.
    """
    def __init__(self):
        self.far = np.empty((0,2), np.float32)
        self.mic = np.empty(0, np.float32)
        self.start = self.end = 0
        self.matches = []
        self.match_channels = []
        self.route_matches = []
        self.echo = False
        self.skew = 0.
        self.phase = 0.

    def append(self, far, mic):
        self.far = np.concatenate((self.far, far))
        self.mic = np.concatenate((self.mic, mic))
        self.end += len(mic)
        if len(self.far) > RATE*2:
            trim = len(self.far)-RATE*2
            self.far = self.far[trim:]; self.mic = self.mic[trim:]; self.start += trim
        window, search = RATE//8, RATE//2
        if len(self.far) < window+search:
            return
        x = np.diff(self.far[-window-search:],axis=0).astype(np.float64)
        y = np.diff(self.mic[-window:]).astype(np.float64)
        if np.dot(y,y) < 1e-4:
            return
        # A single tone has ambiguous delay: it is not evidence of acoustic
        # echo even when microphone/reference correlation is high.
        spectral = np.abs(np.fft.rfft(x,axis=0))**2
        tonal=spectral.max(axis=0)>.1*np.maximum(spectral.sum(axis=0),1e-20)
        if tonal.all():return
        size = 1 << (len(x)+len(y)-1).bit_length()
        corr = np.fft.irfft(np.fft.rfft(x,size,axis=0)*np.conj(np.fft.rfft(y,size))[:,None],size,axis=0)[:len(x)-len(y)+1]
        energy = np.concatenate((np.zeros((1,2)),np.cumsum(x*x,axis=0)))
        energy = energy[len(y):]-energy[:-len(y)]
        norm = np.sqrt(np.maximum(energy*np.dot(y,y),1e-20))
        score = corr/norm
        score[energy<1e-4]=0
        score[:,tonal]=0
        # A transducer/channel can invert polarity; sign is learned by the
        # adaptive filter and must not prevent a valid broadband delay match.
        peak, channel = np.unravel_index(int(np.argmax(np.abs(score))),score.shape)
        corr,score = corr[:,channel],score[:,channel]
        if abs(score[peak]) < .3:
            return
        # A reflected path can be stronger than the direct path. After the
        # strong match validates echo, align to a significant earlier local
        # peak so the causal adaptive filter can model both. Exclude the
        # adjacent autocorrelation lobe and search only 16 ms of pre-echo.
        magnitude = np.abs(score)
        threshold = max(.12, .4*magnitude[peak])
        first = peak+8
        last = min(len(score)-1, peak+RATE*16//1000)
        earlier = [index for index in range(first,last)
                   if magnitude[index] >= threshold and magnitude[index] > magnitude[index-1]
                   and magnitude[index] >= magnitude[index+1]]
        if earlier:peak = earlier[-1]
        fraction = 0.
        if 0 < peak < len(corr)-1:
            a,b,c = corr[peak-1:peak+2]
            fraction = float(np.clip(.5*(a-c)/(a-2*b+c), -.5, .5))
        delay = search-peak-fraction
        center = self.end-window/2
        # Require three consistent new matches before accepting a route jump.
        # One weak/noisy match must not replace an established clock estimate.
        if self.matches and abs(delay-self.matches[-1][1]) > RATE*.003:
            if self.route_matches and abs(delay-self.route_matches[-1][1]) > RATE*.003:
                self.route_matches.clear()
            self.route_matches.append((center,delay,channel))
            if len(self.route_matches)<3:return
            self.matches=[match[:2] for match in self.route_matches[-3:]]
            self.match_channels=[match[2] for match in self.route_matches[-3:]]
            self.route_matches=[]
            self.skew=0.
            return
        self.route_matches=[]
        self.matches.append((center, delay))
        self.matches = self.matches[-12:]
        self.match_channels.append(channel)
        self.match_channels = self.match_channels[-12:]
        if len(self.matches) >= 3:
            self.echo = True
        if len(self.matches) >= 4 and self.matches[-1][0]-self.matches[0][0] >= RATE*.75:
            t,d = np.asarray(self.matches, dtype=np.float64).T
            # Both speaker channels share a sample clock, but their acoustic
            # paths need not share a delay. Fit the common drift within each
            # channel; changing the strongest channel must not invent skew.
            channels = np.asarray(self.match_channels)
            reliable = np.zeros(len(channels), dtype=bool)
            for source in np.unique(channels):
                use = channels == source
                if np.count_nonzero(use) < 4:continue
                reliable |= use
                t[use] -= t[use].mean(); d[use] -= d[use].mean()
            t,d = t[reliable],d[reliable]
            energy = float(np.dot(t,t))
            if energy == 0:return
            slope = float(np.dot(t,d)/energy)
            error = float(np.sqrt(np.mean((d-slope*t)**2)))
            if abs(slope) <= .0005 and error < .6:
                self.skew = -slope

    def reference(self, start, count):
        if not len(self.far):return np.zeros(count*2,np.float32)
        positions = np.arange(start,start+count,dtype=np.float64)+self.phase+np.arange(count)*self.skew
        self.phase = float(np.clip(self.phase+count*self.skew,-LOOKAHEAD,LOOKAHEAD))
        index = np.arange(self.start,self.end)
        return np.column_stack([np.interp(positions,index,self.far[:,channel],left=0.,right=float(self.far[-1,channel]))
                                for channel in range(2)]).astype(np.float32).reshape(-1)


class EchoMixer:
    """Frame buffering, exact delay compensation and Pause/Stop tail draining."""
    def __init__(self, sink):
        self.state = None
        self.sink = sink
        path = library_path()
        if not path.is_file():
            raise RuntimeError('Speaker echo cancellation is missing from this build. Rebuild the capture audio component.')
        self.lib = ctypes.CDLL(str(path))
        pointer = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
        try:
            self.lib.speakerdesk_echo_abi.restype=ctypes.c_int
            if self.lib.speakerdesk_echo_abi()!=3:raise RuntimeError('Capture echo component has an incompatible version.')
        except AttributeError as exc:
            raise RuntimeError('Capture echo component has an incompatible version.') from exc
        self.lib.speakerdesk_echo_create.restype = ctypes.c_void_p
        self.lib.speakerdesk_echo_destroy.argtypes = [ctypes.c_void_p]
        self.lib.speakerdesk_echo_reset.argtypes = [ctypes.c_void_p]
        self.lib.speakerdesk_echo_process.argtypes = [ctypes.c_void_p,pointer,pointer,pointer,ctypes.c_size_t,ctypes.c_int]
        self.state = self.lib.speakerdesk_echo_create()
        if not self.state:
            raise RuntimeError('Speaker echo cancellation could not start.')
        self._reset_buffers()

    def _reset_buffers(self):
        self.clock = ReferenceClock()
        self.pending = {s:np.empty(0,np.float32) for s in ('microphone','system')}
        self.input_mic = np.empty(0,np.float32)
        self.position = 0
        self.discard = DELAY
        self.raw_delay = np.zeros(DELAY,np.float32)

    def close(self):
        if self.state:
            self.lib.speakerdesk_echo_destroy(self.state)
            self.state = None

    def __del__(self):
        self.close()

    def add(self, tracks, final=False):
        mic,far = tracks['microphone'],tracks['system']
        reference = tracks.get('system_reference',np.repeat(far[:,None],2,axis=1))
        if len(mic):self.clock.append(reference,mic)
        for s in self.pending:
            self.pending[s] = np.concatenate((self.pending[s],tracks[s]))
        self.input_mic = np.concatenate((self.input_mic,mic))
        length = len(self.input_mic)
        process_length = ((length+FRAME-1)//FRAME*FRAME+FRAME) if final else max(0,length-LOOKAHEAD)//FRAME*FRAME
        padded = np.pad(self.input_mic,(0,max(0,process_length-length)))
        outputs = []
        for offset in range(0,process_length,FRAME):
            original = padded[offset:offset+FRAME]
            microphone = np.ascontiguousarray(np.clip(original,-1,1),dtype=np.float32)
            render = np.clip(self.clock.reference(self.position,FRAME),-1,1)
            cleaned = np.empty(FRAME,np.float32)
            delay = round((self.clock.matches[-1][1]+self.clock.phase)*1000/RATE) if self.clock.matches else -1
            delay = min(1000,max(0,delay)) if self.clock.matches else -1
            if self.lib.speakerdesk_echo_process(self.state,render,microphone,cleaned,FRAME,delay):
                raise RuntimeError('Speaker echo cancellation rejected an audio frame.')
            # Without independently verified echo, preserve the microphone exactly.
            raw = np.concatenate((self.raw_delay,original))
            self.raw_delay = raw[FRAME:].copy()
            if not self.clock.echo:
                cleaned = raw[:FRAME]
            outputs.append(cleaned)
            self.position += FRAME
        self.input_mic = self.input_mic[min(process_length,length):]
        if outputs:
            clean = np.concatenate(outputs)
            skip = min(len(clean),self.discard); clean = clean[skip:];self.discard -= skip
            count = min(len(clean),len(self.pending['microphone']))
            if count:
                originals = {s:self.pending[s][:count] for s in self.pending}
                # Retain the actual compensated AEC output, before independent
                # PCM quantization. Subtracting quantized mix/system loses it
                # when clipping occurs and is not an input provenance contract.
                originals['microphone_clean'] = clean[:count].copy()
                self.sink(np.clip(originals['system']+clean[:count],-1,1).astype('<f4'), originals)
                self.pending = {s:self.pending[s][count:] for s in self.pending}
        if final:
            if any(len(v) for v in self.pending.values()):
                raise RuntimeError('Speaker echo cancellation did not drain its audio tail.')
            if self.lib.speakerdesk_echo_reset(self.state):
                raise RuntimeError('Speaker echo cancellation could not reset after Pause.')
            self._reset_buffers()
