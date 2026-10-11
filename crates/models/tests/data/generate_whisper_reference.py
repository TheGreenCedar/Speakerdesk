"""Validation only: pinned upstream Whisper language pass, without transcription dependencies."""
import argparse, ast, dataclasses, hashlib, importlib.metadata, importlib.util
import json, math, sys, time, types, typing, wave
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
for name in ('source','dsp','model','wav','output'):p.add_argument(name,type=Path)
args=p.parse_args()
hashes={'whisper.py':'02a03277902d45abc314e9b6c5e54efcf9192213640ab8bd19bfaa1e21900ffa',
        'audio.py':'c042e5716c5e3c9b87977b86ba0464bba85443f076e8a888937e82c5460f12c0',
        'decoding.py':'2672e376d51e26e0da9da6f898fb7fbf2f7c049636f2e9d2ef3789462bd0991c'}
for name,digest in hashes.items():assert hashlib.sha256((args.source/name).read_bytes()).hexdigest()==digest
assert hashlib.sha256(args.dsp.read_bytes()).hexdigest()=='c47da00eea77f3bce396d4eb36c694211f40d9e5571393b685eda77f8978b98c'
assert hashlib.sha256((args.model/'model.safetensors').read_bytes()).hexdigest()=='1267601753d2996d68dc065d1de786895a7ad9a874d1d3548a5274c1bc067444'
spec=importlib.util.spec_from_file_location('reference_dsp',args.dsp)
dsp=importlib.util.module_from_spec(spec);spec.loader.exec_module(dsp)
module=types.ModuleType('native_whisper_reference');sys.modules[module.__name__]=module
ns=module.__dict__;ns.update(vars(typing))
ns.update(mx=mx,nn=nn,np=np,math=math,dataclass=dataclasses.dataclass,Path=Path,
          hanning=dsp.hanning,mel_filters=dsp.mel_filters,stft=dsp.stft,
          N_FFT=400,HOP_LENGTH=160,N_SAMPLES=480000,SAMPLE_RATE=16000)
for file,names in [('audio.py',{'pad_or_trim','log_mel_spectrogram'}),('decoding.py',{'detect_language'}),
                   ('whisper.py',{'ModelDimensions','MultiHeadAttention','ResidualAttentionBlock','AudioEncoder','TextDecoder','Model','sinusoids'})]:
    if file=='whisper.py':ns.update(detect_language_function=ns['detect_language'],decode_function=None)
    nodes=[n for n in ast.parse((args.source/file).read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]+nodes,type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree),str(args.source/file),'exec'),ns)
mx.set_default_device(mx.gpu)
model=ns['Model'](ns['ModelDimensions'].from_dict(json.loads((args.model/'config.json').read_text())))
model.load_weights(list(model.sanitize(mx.load(str(args.model/'model.safetensors'))).items()),strict=True)
mx.eval(model.parameters())
tokens=json.loads((args.model/'added_tokens.json').read_text())
import re
languages={k[2:-2]:v for k,v in tokens.items() if re.fullmatch(r'<\|[a-z]{2,3}\|>',k)}
tokenizer=types.SimpleNamespace(language='en',language_token=languages['en'],sot=tokens['<|startoftranscript|>'],sot_sequence=(tokens['<|startoftranscript|>'],languages['en']),all_language_tokens=tuple(languages.values()),all_language_codes=tuple(languages))
with wave.open(str(args.wav)) as wav:
    assert wav.getparams()[:3]==(1,2,16000)
    pcm=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(np.float32)/32768
rows=[]
for name,audio in [('whole',pcm),('first_probe',pcm[:48000]),('partial_frame',pcm[16000:16549]),('silence',np.zeros(512,dtype=np.float32))]:
    mel=ns['log_mel_spectrogram'](ns['pad_or_trim'](mx.array(audio))).astype(model.dtype)
    started=time.perf_counter();_,probabilities=ns['detect_language'](model,mel,tokenizer);elapsed=time.perf_counter()-started
    indices=[(frame,band) for frame in (0,1,2,17,100,299,600,908,1499,2999) for band in (0,1,7,30,79)]
    features=mel.astype(mx.float32).tolist()
    rows.append(dict(name=name,samples=len(audio),probabilities=probabilities,seconds=elapsed,mel=[dict(frame=f,band=b,value=features[f][b]) for f,b in indices]))
result=dict(source_revision='feb25a37b07923bae556e59111995071d66afa0d',source_sha256=hashes,
            mlx_version=importlib.metadata.version('mlx'),wav_sha256=hashlib.sha256(args.wav.read_bytes()).hexdigest(),cases=rows)
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({r['name']:sorted(r['probabilities'].items(),key=lambda x:-x[1])[:3] for r in rows},indent=2))
