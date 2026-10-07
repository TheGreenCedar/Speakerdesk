"""Resident exact-input reuse of raw language probabilities; no transcript state."""
from collections import OrderedDict
import copy
import hashlib
import importlib.metadata
import json
import math


def validate_language_scores(probabilities):
    if (not probabilities or any(not isinstance(code, str) or not math.isfinite(float(p))
                                or not 0 <= float(p) <= 1 for code, p in probabilities.items())
            or not .98 <= sum(float(p) for p in probabilities.values()) <= 1.02):
        raise ValueError('Invalid language detection probabilities.')


class LanguageProbeCache:
    """Bounded resident raw scores, never routed language, speech or ASR text.

    An exact physical crop may recur during final refinement. Policy/context
    decisions are recomputed by each transcriber, and failures remain retryable.
    No PCM, model tensors or persisted transcript records are retained here.
    """
    def __init__(self, *, max_entries=1024, max_bytes=4*1024**2):
        if type(max_entries) is not int or max_entries<1 or type(max_bytes) is not int or max_bytes<1:
            raise ValueError('Invalid language probe cache bound.')
        self.max_entries=max_entries;self.max_bytes=max_bytes
        self.entries=OrderedDict();self.bytes=0;self.detector=None;self.model_identity=None
        self.runtime_identity=[]
        for package in ('mlx', 'mlx-audio'):
            try:version=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:version=None
            self.runtime_identity.append((package,version))

    def detect(self, detector, audio, *, start_sample, language_epoch, model_identity):
        import numpy as np
        if (type(start_sample) is not int or start_sample<0
                or type(language_epoch) is not int or language_epoch<0):
            raise ValueError('Invalid language probe scope.')
        # A replacement provider or configuration cannot inherit another's scores.
        if self.detector is not detector or self.model_identity!=(model_identity,self.runtime_identity):
            self.entries.clear();self.bytes=0
            self.detector=detector;self.model_identity=copy.deepcopy((model_identity,self.runtime_identity))
        pcm=np.asarray(audio)
        key=(language_epoch,start_sample,pcm.dtype.str,pcm.shape,
             hashlib.sha256(pcm.tobytes()).hexdigest())
        data=self.entries.pop(key,None)
        if data is not None:
            self.entries[key]=data
            return json.loads(data)
        probabilities=detector.detect(audio)
        validate_language_scores(probabilities)
        try:data=json.dumps(probabilities,separators=(',',':'),allow_nan=False).encode()
        except (TypeError,ValueError):return probabilities
        if len(data)<=self.max_bytes:
            self.entries[key]=data;self.bytes+=len(data)
            while len(self.entries)>self.max_entries or self.bytes>self.max_bytes:
                _,old=self.entries.popitem(last=False);self.bytes-=len(old)
        return probabilities
