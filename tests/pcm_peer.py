"""Varying PCM for CPU-only model substitutes; this is not genuine speech."""
import numpy as np
def varying_pcm(samples,value=.1,dtype=np.float32):
 audio=np.full(samples,value,dtype=dtype)
 audio[1::2]*=-1
 return audio
