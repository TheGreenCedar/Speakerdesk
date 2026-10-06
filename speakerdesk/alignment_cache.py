"""Bounded raw provider evidence reuse; attachment/CAS/language stay outside."""
from collections import OrderedDict
import hashlib
import json


class AlignmentCache:
    def __init__(self, identity, *, max_entries=32, max_bytes=2*1024**2):
        if type(max_entries) is not int or type(max_bytes) is not int or min(max_entries,max_bytes)<1:
            raise ValueError('Invalid alignment evidence cache bounds.')
        self.identity=tuple(identity);self.max_entries=max_entries;self.max_bytes=max_bytes
        self.entries=OrderedDict();self.bytes=0;self.hits=0;self.provider_calls=0

    def key(self, text, audio_bytes, start, end, language):
        return (self.identity,hashlib.sha256(text.encode('utf-8')).hexdigest(),
                hashlib.sha256(audio_bytes).hexdigest(),start,end,language)

    def get(self,key):
        data=self.entries.get(key)
        if data is None:return None
        self.entries.move_to_end(key);self.hits+=1
        return json.loads(data)

    def put(self,key,result):
        if not isinstance(result,dict):return False
        identity,text_sha,audio_sha,start,end,language=key
        model_sha,token_sha,calibration,timing_kind,*_=identity
        if (language!='en' or result.get('status')!='aligned' or result.get('complete') is not True
                or result.get('text_source')!='cohere' or result.get('text_sha256')!=text_sha
                or result.get('audio_float32_sha256')!=audio_sha
                or result.get('audio_anchor')!={'start_sample':start,'end_sample':end}
                or result.get('model_sha256')!=model_sha
                or result.get('frame_calibration_id')!=calibration
                or result.get('score_calibration_id')!=calibration
                or result.get('timing_kind')!=timing_kind
                or result.get('qualified_scope')!='bounded_ami_english_coarse_envelopes'):
            return False
        try:data=json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')
        except (ValueError,TypeError):return False
        if len(data)>self.max_bytes:return False
        prior=self.entries.pop(key,None)
        if prior is not None:self.bytes-=len(prior)
        self.entries[key]=data;self.bytes+=len(data)
        while len(self.entries)>self.max_entries or self.bytes>self.max_bytes:
            _,old=self.entries.popitem(last=False);self.bytes-=len(old)
        return True

    def metrics(self):
        return {'provider_calls':self.provider_calls,'cache_hits':self.hits,
                'cache_entries':len(self.entries),'cache_payload_bytes':self.bytes,
                'cache_max_entries':self.max_entries,'cache_max_payload_bytes':self.max_bytes}
