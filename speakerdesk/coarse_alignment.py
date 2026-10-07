"""Isolated GPU CTC provider qualified only against frozen English AMI gates.
No ASR, model downloader, CPU neural fallback or fitted timing adjustment.
"""
import hashlib,json
from pathlib import Path
from word_alignment import MODEL_SHA256,MAX_AUDIO_SAMPLES,parse_vocabulary,align_ctc_scores,FrameClock,AcceptancePolicy
ROOT=Path('/Users/albert/Documents/Codex/2026-10-07/task-9')
CALIBRATION_ID='ami-english-coarse-mlx-f32-v1-5ab4e661e62f-38ff6a225e75'
TOKEN_SHA256='a7a044c52cb29cbe8b0dc1953e92cefd4ca16b0ed968177b6beab21f9a7d0b31'
class CoarseAlignment:
 def __init__(self,directory,*,cache_directory=None):
  audit=json.loads((ROOT/'evidence/mlx-ctc-independent-quality/timing-audit.json').read_text())
  assert audit['bounded_coarse_timing_pass'] and audit['threshold_or_offset_fitted'] is False
  assert audit['plan_sha256']=='5ab4e661e62f757bc1498121b2c392b46216f4ad3db55f42e16448b43bf64396'
  report=json.loads((ROOT/'evidence/mlx-ctc-independent-quality/report.json').read_text())
  assert report['conversion_identity']==MODEL_SHA256 and report['calibration_id']==CALIBRATION_ID
  from mlx_ctc_forward import load_gpu_model
  self.model,self.conversion=load_gpu_model(cache_limit_bytes=512*1024**2,directory=ROOT/'mlx-ctc-prototype')
  assert self.conversion['converted_weights_sha256']==MODEL_SHA256
  tokens=(ROOT/'mlx-ctc-prototype/tokens.txt').read_bytes();assert hashlib.sha256(tokens).hexdigest()==TOKEN_SHA256
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
  scores=forward_scores(self.model,audio);self.actual_gpu_forwards+=1
  materialized=np.array(scores,copy=True)
  result=align_ctc_scores(text,materialized,self.vocabulary,clock=FrameClock(320,0,CALIBRATION_ID),policy=AcceptancePolicy(-20,CALIBRATION_ID),audio_start_sample=start_sample,audio_num_samples=len(audio))
  result.update(audio_float32_sha256=hashlib.sha256(data).hexdigest(),qualified_scope='bounded_ami_english_coarse_envelopes',candidate_runtime='mlx_cached_dequantized_float32_GPU')
  self.cache.put(key,result)
  del scores,materialized
  return result
