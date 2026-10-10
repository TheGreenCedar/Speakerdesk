"""Varying PCM for CPU-only model substitutes; this is not genuine speech."""
import numpy as np
def varying_pcm(samples,value=.1,dtype=np.float32):
 audio=np.full(samples,value,dtype=dtype)
 audio[1::2]*=-1
 return audio


class SpeechEvidencePeer:
 """Predetermined CPU evidence; no trained VAD or acoustic classification."""
 def __init__(self, decision='speech'):self.decision=decision
 def admission(self,start,end):
  from speech_admission import SILERO_SPEC
  return {'source':'silero_v6','model_revision':SILERO_SPEC['revision'],
          'start_sample':start,'end_sample':end,'complete':self.decision!='pending',
          'maximum_probability':.01 if self.decision=='no_speech' else .8,
          'decision':self.decision,'speech_regions':([{'start_sample':start,'end_sample':end}]
          if self.decision=='speech' else [])}


class SpeechFramePeer:
 """Predetermined frame score for CPU worker/MLX boundary tests only."""
 def initial_state(self):return 0
 def feed(self,chunk,state):return .8,state+1
 def session(self,start_sample=0,*,archive=None,conditioning=None):
  from speech_admission import SpeechSession,HOT_FRAMES
  return SpeechSession(self,start_sample,max_frames=HOT_FRAMES if archive else None,archive=archive,conditioning=conditioning)
 def inspect_frames(self,audio,start_sample=0,*,archive=None,conditioning=None):
  session=self.session(start_sample,archive=archive,conditioning=conditioning);session.feed(audio,start_sample,final=True)
  return session.evidence
