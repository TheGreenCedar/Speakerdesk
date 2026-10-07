"""Durable selected machine authority; hot eviction never invokes ASR again."""
from collections import OrderedDict
import hashlib,json
from pathlib import Path


class AuthorityStore:
    def __init__(self,directory,*,max_entries=32,max_bytes=4*1024**2):
        self.directory=Path(directory);self.directory.mkdir()
        self.hot=OrderedDict();self.bytes=0;self.max_entries=max_entries;self.max_bytes=max_bytes
    @staticmethod
    def name(key):return hashlib.sha256(json.dumps(key,separators=(',',':')).encode()).hexdigest()+'.json'
    def _hot(self,key,data):
        prior=self.hot.pop(key,None)
        if prior is not None:self.bytes-=len(prior)
        if len(data)>self.max_bytes:return
        self.hot[key]=data;self.bytes+=len(data)
        while len(self.hot)>self.max_entries or self.bytes>self.max_bytes:
            _,old=self.hot.popitem(last=False);self.bytes-=len(old)
    def get(self,key):
        data=self.hot.get(key)
        if data is None:
            path=self.directory/self.name(key)
            if not path.exists():return None
            data=path.read_bytes()
        value=json.loads(data)
        if value['key']!=list(key):raise ValueError('Selected authority belongs to different original audio or operation.')
        self._hot(key,data)
        return value['part']
    def reference(self,key):
        path=self.directory/self.name(key)
        return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    def put(self,key,part):
        # An authority is immutable within its operation. A new retry uses a
        # new operation identity, never an overwrite of the chosen raw record.
        data=json.dumps({'key':list(key),'part':part},ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
        path=self.directory/self.name(key)
        if path.exists():
            if path.read_bytes()!=data:raise ValueError('Selected authority cannot be silently rewritten.')
        else:
            with path.open('xb') as stream:stream.write(data)
        self._hot(key,data)
