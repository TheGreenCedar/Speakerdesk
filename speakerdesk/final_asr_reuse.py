"""Exact selected-live Cohere result reuse, with an observer-first contract.

This stores raw results only. Current speech/language routing, final timing and
ownership still execute. Reuse requires a qualification bound to model identity.
"""
from collections import OrderedDict
import copy
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def provider_state(provider):
    """Loaded configuration/precision metadata, without evaluating parameters."""
    config=getattr(provider,'config',None)
    if is_dataclass(config):config=asdict(config)
    if config is not None and not isinstance(config,dict):config=repr(config)
    features=getattr(provider,'feature_extractor',None)
    model=getattr(provider,'model',None);tokenizer=getattr(provider,'tokenizer',None)
    return {'config':config,'model_instance':id(model),'tokenizer_instance':id(tokenizer),
        'feature_config':{k:v for k,v in vars(features).items()
            if isinstance(v,(str,int,float,bool,type(None)))} if features is not None else None,
        'training':getattr(model,'training',None),
        'projection_dtype':str(getattr(getattr(getattr(model,'proj_out',None),'weight',None),'dtype',None))}


def selected_authority(authority,raw):
    try:
        data=Path(authority['path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=authority['sha256']:return False
        selected=json.loads(data)['part']
        return (selected.get('complete') is True and selected.get('text')==raw['text']
            and len(selected.get('passages',[]))==1
            and not selected['passages'][0].get('transcription_review'))
    except (OSError,ValueError,TypeError,KeyError):return False


class FinalAsrReuse:
    def __init__(self, identity, *, mode='observe', qualification=None,
                 max_entries=256, max_bytes=4*1024**2):
        if mode not in ('observe','reuse') or max_entries<1 or max_bytes<1:
            raise ValueError('Invalid final ASR reuse configuration.')
        self.identity=copy.deepcopy(identity);self.identity_sha256=digest(identity)
        self.mode=mode;self.qualified=(qualification==self.identity_sha256
            and identity.get('device') in ('gpu','neural_engine'))
        self.max_entries=max_entries;self.max_bytes=max_bytes
        self.entries=OrderedDict();self.bytes=0;self.provider=None
        self.begin(None)

    def begin(self,scope):
        # A request owns a bounded pending set. Only selected complete live
        # authorities may admit these results after recognition/part validation.
        self.scope=copy.deepcopy(scope);self.pending=[];self.decisions=[]
        self.active=scope is not None

    def resolve(self,provider,audio,settings,routing,compute):
        import numpy as np
        if self.provider is not provider:
            self.entries.clear();self.bytes=0;self.provider=provider
        pcm=np.asarray(audio)
        scope=self.scope or {}
        binding=scope.get('binding') or {}
        signature={'model_identity_sha256':self.identity_sha256,
            'loaded_provider_state':provider_state(provider),
            'utterance_id':binding.get('utterance_id'),'language_epoch':binding.get('language_epoch'),
            'decoder_pcm_sha256':hashlib.sha256(pcm.tobytes()).hexdigest(),
            'dtype':pcm.dtype.str,'shape':list(pcm.shape),'settings':settings,'routing':routing}
        key=digest(signature)
        data=self.entries.get(key);saved=json.loads(data) if data is not None else None
        allowed=(scope.get('stage')=='refined' and scope.get('purpose')=='automatic_final'
            and scope.get('eligible') is True and bool(binding.get('expected_source_sha256'))
            and signature['loaded_provider_state']['training'] is False)
        if saved and not selected_authority(saved['authority'],saved['result']):
            self.entries.pop(key);self.bytes-=len(data);saved=None
        reason=('matching_selected_live_result' if saved and allowed else
                'not_automatic_final' if scope.get('purpose')!='automatic_final' else
                'current_revision_ineligible' if not allowed else 'no_exact_selected_live_result')
        decision={'signature_sha256':key,'signature':signature,'consumer_binding':binding,
            'operation_id':scope.get('operation_id'),'stage':scope.get('stage'),
            'purpose':scope.get('purpose'),'eligible':bool(saved and allowed),
            'decision':reason,'provider_executed':True,'reused':False}
        if saved and allowed:
            self.entries.move_to_end(key)
            decision['source_authority']=saved['authority']
            decision['source_binding']=saved['binding']
        if saved and allowed and self.mode=='reuse' and self.qualified:
            raw=saved['result'];result=SimpleNamespace(**copy.deepcopy(raw))
            decision.update(provider_executed=False,reused=True,decision='qualified_exact_live_result')
        else:
            # Observe always executes the real current attempt. Failures are
            # recorded by the caller and never promoted into a live authority.
            try:result=compute()
            except Exception:
                decision['decision']='provider_error';self.decisions.append(decision);raise
            if saved and allowed:
                decision['shadow_text_equal']=saved['result']['text']==result.text
                decision['shadow_tokens_equal']=saved['result']['tokens']==list(result.tokens)
                if self.mode=='reuse' and not self.qualified:decision['decision']='qualification_missing'
        raw={'text':result.text,'tokens':list(result.tokens),'language':getattr(result,'language',settings['language'])}
        valid=(isinstance(raw['text'],str) and bool(raw['text'].strip())
            and 0<len(raw['tokens'])<settings['max_new_tokens']
            and all(type(token) is int and token>=0 for token in raw['tokens'])
            and raw['language']==settings['language'])
        decision['raw_text_sha256']=hashlib.sha256(raw['text'].encode()).hexdigest()
        decision['raw_tokens_sha256']=digest(raw['tokens']);decision['result_complete']=valid
        self.decisions.append(decision)
        if (scope.get('stage')=='live' and scope.get('eligible') is True and valid
                and signature['loaded_provider_state']['training'] is False):
            self.pending.append((key,raw,copy.deepcopy(binding)))
        return result,copy.deepcopy(decision)

    def finish(self,authority=None):
        if (authority and self.scope and self.scope.get('stage')=='live'
                and self.scope.get('eligible') is True):
            for key,result,binding in self.pending:
                if not selected_authority(authority,result):continue
                data=json.dumps({'result':result,'binding':binding,'authority':authority},
                    ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
                if len(data)>self.max_bytes:continue
                prior=self.entries.pop(key,None)
                if prior is not None:self.bytes-=len(prior)
                self.entries[key]=data;self.bytes+=len(data)
                while len(self.entries)>self.max_entries or self.bytes>self.max_bytes:
                    _,old=self.entries.popitem(last=False);self.bytes-=len(old)
        decisions=copy.deepcopy(self.decisions);self.begin(None)
        return decisions
