"""GPU CTC provider qualified only against frozen English AMI coarse gates.
No ASR, model downloader, CPU neural fallback or fitted timing adjustment.
"""
import hashlib
from pathlib import Path
from word_alignment import MODEL_SHA256,MAX_AUDIO_SAMPLES,parse_vocabulary,align_ctc_scores,FrameClock,AcceptancePolicy
from alignment_model import CALIBRATION_ID,TOKEN_SHA256,MANIFEST_SHA256
class CoarseAlignment:
 def __init__(self,directory,*,cache_directory=None):
  from mlx_ctc_forward import load_gpu_model
  self.model,self.conversion=load_gpu_model(cache_limit_bytes=512*1024**2,directory=Path(directory))
  if self.conversion['converted_weights_sha256']!=MODEL_SHA256:raise ValueError('GPU model identity differs.')
  tokens=(Path(directory)/'tokens.txt').read_bytes()
  if hashlib.sha256(tokens).hexdigest()!=TOKEN_SHA256:raise ValueError('Alignment vocabulary identity differs.')
  self.vocabulary=parse_vocabulary(tokens.decode())
  from alignment_cache import AlignmentCache
  self.cache=AlignmentCache((MODEL_SHA256,TOKEN_SHA256,CALIBRATION_ID,'ctc_emission_cell_envelope',320,0,-20,'mlx-0.32.2-cached-float32-gpu','ctc-segmentation-1.7.4'),directory=cache_directory)
  self.actual_gpu_forwards=0
 def align(self,audio,text,*,start_sample,language):
  if language!='en':return None
  import numpy as np
  audio=np.asarray(audio,dtype=np.float32)
  if audio.ndim!=1 or not 0<len(audio)<=MAX_AUDIO_SAMPLES or not np.isfinite(audio).all() or type(start_sample) is not int or start_sample<0:raise ValueError('Invalid coarse audio anchor')
  if not isinstance(text,str):raise ValueError('Invalid supplied Cohere text')
  data=audio.astype('<f4').tobytes();key=self.cache.key(text,data,start_sample,start_sample+len(audio),language)
  cached=self.cache.get(key)
  if cached is not None:return cached
  self.cache.provider_calls+=1
  from mlx_ctc_forward import forward_scores
  execution={}
  scores=forward_scores(self.model,audio,execution=execution);self.actual_gpu_forwards+=1
  if execution.get('backend')!='mlx_metal_gpu' or execution.get('evaluated_and_GPU_synchronized') is not True:raise RuntimeError('GPU alignment execution evidence missing.')
  materialized=np.array(scores,copy=True)
  result=align_ctc_scores(text,materialized,self.vocabulary,clock=FrameClock(320,0,CALIBRATION_ID),policy=AcceptancePolicy(-20,CALIBRATION_ID),audio_start_sample=start_sample,audio_num_samples=len(audio))
  result.update(audio_float32_sha256=hashlib.sha256(data).hexdigest(),qualified_scope='bounded_ami_english_coarse_envelopes',candidate_runtime='mlx_cached_dequantized_float32_GPU',provider_execution=execution,source_model_sha256=self.conversion['source_model_sha256'],conversion_manifest_sha256=MANIFEST_SHA256)
  self.cache.put(key,result)
  del scores,materialized
  return result
