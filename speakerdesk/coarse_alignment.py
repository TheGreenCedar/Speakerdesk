"""Pinned English coarse CTC envelopes for supplied Cohere text only.

Optional until the approved runtime/model are packaged. No transcription
decode, download, native compilation fallback or unverified language policy.
"""
import hashlib
from pathlib import Path
from word_alignment import (MODEL_SHA256, MAX_AUDIO_SAMPLES, parse_vocabulary,
                            align_ctc_scores, FrameClock, AcceptancePolicy)

CALIBRATION_ID = 'ami-english-coarse-v1-5ab4e661e62f'
TOKEN_SHA256 = 'a7a044c52cb29cbe8b0dc1953e92cefd4ca16b0ed968177b6beab21f9a7d0b31'


class CoarseAlignment:
    def __init__(self, directory):
        directory=Path(directory)
        for name,digest in (('model.int8.onnx',MODEL_SHA256),('tokens.txt',TOKEN_SHA256)):
            with (directory/name).open('rb') as stream:
                if hashlib.file_digest(stream,'sha256').hexdigest()!=digest:
                    raise ValueError('Unverified coarse alignment artifact.')
        import onnxruntime as ort
        if ort.__version__!='1.30.0':raise ValueError('Pinned alignment runtime unavailable.')
        ort.disable_telemetry_events()
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL;options.enable_cpu_mem_arena=False
        self.session=ort.InferenceSession(str(directory/'model.int8.onnx'),
            sess_options=options,providers=['CPUExecutionProvider'])
        self.vocabulary=parse_vocabulary((directory/'tokens.txt').read_text())

    def align(self,audio,text,*,start_sample,language):
        if language!='en':return None
        import numpy as np
        audio=np.asarray(audio,dtype=np.float32)
        if (audio.ndim!=1 or not 0<len(audio)<=MAX_AUDIO_SAMPLES or not np.isfinite(audio).all()
                or type(start_sample) is not int or start_sample<0):
            raise ValueError('Invalid coarse alignment audio anchor.')
        x=(audio-audio.mean())/np.sqrt(audio.var()+np.float32(1e-5))
        logits,=self.session.run(None,{'x':x[None,:].astype(np.float32)})
        scores=logits[0].copy();scores-=np.max(scores,axis=1,keepdims=True)
        scores-=np.log(np.exp(scores).sum(axis=1,keepdims=True))
        result=align_ctc_scores(text,scores,self.vocabulary,
            clock=FrameClock(320,0,CALIBRATION_ID),policy=AcceptancePolicy(-20,CALIBRATION_ID),
            audio_start_sample=start_sample,audio_num_samples=len(audio))
        result['audio_float32_sha256']=hashlib.sha256(audio.astype('<f4').tobytes()).hexdigest()
        result['qualified_scope']='bounded_ami_english_coarse_envelopes'
        return result
