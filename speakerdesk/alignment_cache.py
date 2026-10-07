"""Bounded raw provider evidence reuse; attachment/CAS/language stay outside."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path


class AlignmentCache:
    def __init__(self, identity, *, max_entries=32, max_bytes=2*1024**2,
                 directory=None, disk_max_entries=256, disk_max_bytes=32*1024**2):
        if any(type(x) is not int or x<1 for x in (max_entries,max_bytes,disk_max_entries,disk_max_bytes)):
            raise ValueError('Invalid alignment evidence cache bounds.')
        self.identity=tuple(identity);self.max_entries=max_entries;self.max_bytes=max_bytes
        self.entries=OrderedDict();self.bytes=0;self.hits=0;self.provider_calls=0
        # Only this worker's validated writes are eligible. Existing files are
        # never discovered or trusted across workers, calibration, or models.
        self.directory=Path(directory) if directory is not None else None
        self.disk_write_disabled=False
        if self.directory is not None:
            try:self.directory.mkdir()
            except OSError:self.directory=None;self.disk_write_disabled=True
        self.disk=OrderedDict();self.disk_bytes=0;self.disk_hits=0
        self.disk_max_entries=disk_max_entries;self.disk_max_bytes=disk_max_bytes

    def key(self, text, audio_bytes, start, end, language):
        return (self.identity,hashlib.sha256(text.encode('utf-8')).hexdigest(),
                hashlib.sha256(audio_bytes).hexdigest(),start,end,language)

    def get(self,key):
        data=self.entries.get(key)
        if data is None:
            saved=self.disk.get(key)
            if saved is None:return None
            path,digest,length=saved
            try:
                if path.stat().st_size!=length:return None
                data=path.read_bytes()
            except OSError:return None
            if hashlib.sha256(data).hexdigest()!=digest:return None
            self.disk.move_to_end(key);self.disk_hits+=1
            self._hot(key,data)
        else:self.entries.move_to_end(key)
        self.hits+=1
        return json.loads(data)

    def _hot(self,key,data):
        prior=self.entries.pop(key,None)
        if prior is not None:self.bytes-=len(prior)
        if len(data)>self.max_bytes:return
        self.entries[key]=data;self.bytes+=len(data)
        while len(self.entries)>self.max_entries or self.bytes>self.max_bytes:
            _,old=self.entries.popitem(last=False);self.bytes-=len(old)

    def _persist(self,key,data):
        if self.directory is None or self.disk_write_disabled or len(data)>self.disk_max_bytes:return
        name=hashlib.sha256(json.dumps(key,separators=(',',':')).encode()).hexdigest()+'.json'
        path=self.directory/name
        # Reserve capacity before writing. A failed deletion remains counted
        # and disables disk writes; it can never create unbounded orphan files.
        if key in self.disk and not self._drop_disk(key):return
        while len(self.disk)>=self.disk_max_entries or self.disk_bytes+len(data)>self.disk_max_bytes:
            if not self._drop_disk(next(iter(self.disk))):return
        try:
            with path.open('xb') as stream:stream.write(data)
        except OSError:
            try:path.unlink()
            except FileNotFoundError:pass
            except OSError:
                try:length=path.stat().st_size
                except OSError:length=len(data)
                self.disk[key]=(path,hashlib.sha256(data).hexdigest(),length);self.disk_bytes+=length
                self.disk_write_disabled=True
            return
        self.disk[key]=(path,hashlib.sha256(data).hexdigest(),len(data));self.disk_bytes+=len(data)

    def _drop_disk(self,key):
        path,_,length=self.disk[key]
        try:path.unlink()
        except FileNotFoundError:pass
        except OSError:self.disk_write_disabled=True;return False
        self.disk.pop(key);self.disk_bytes-=length;return True

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
        if len(data)>self.max_bytes and (self.directory is None or len(data)>self.disk_max_bytes):return False
        self._hot(key,data);self._persist(key,data)
        return True

    def metrics(self):
        return {'provider_calls':self.provider_calls,'cache_hits':self.hits,
                'cache_entries':len(self.entries),'cache_payload_bytes':self.bytes,
                'cache_max_entries':self.max_entries,'cache_max_payload_bytes':self.max_bytes,
                'disk_cache_hits':self.disk_hits,'disk_cache_entries':len(self.disk),
                'disk_cache_payload_bytes':self.disk_bytes,
                'disk_cache_max_entries':self.disk_max_entries,'disk_cache_max_payload_bytes':self.disk_max_bytes,
                'disk_cache_writes_disabled':self.disk_write_disabled}
