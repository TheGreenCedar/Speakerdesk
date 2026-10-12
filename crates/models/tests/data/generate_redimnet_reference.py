"""Validation only: unchanged 0.6.6 MLX voice graph, without admission/receipt machinery.

The retained vectors assess conversion drift on two synthetic clips. They do not
establish human voice recognition quality or compatibility for every saved profile.
"""
import argparse, hashlib, importlib.metadata, json, sys, time, wave
from pathlib import Path
import mlx.core as mx
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
for name in ('source','model','wav','output'):p.add_argument(name,type=Path)
args=p.parse_args()
hashes={'mlx_voice_mil.py':'c9684ce72fb8f633937f074c5f66fe4a224437e8872f4ac09e6e6eeea86ebd4b',
    'voice_waveform.py':'a3cc6dbe7f1411e1a8ed1f60a7885c139fcf150da82f659e6ceeb74655d59bee',
    'voice_profiles.py':'9651006e9b851275136f6b4db1c94c10a3dbf110814da57cd9c0914073eeaeeb',
    'redimnet2_artifact.json':'01d69f47a1ec8af7c84a254ed836c421657c6933ca36d9430a314e811e716ecd'}
for name,digest in hashes.items():assert hashlib.sha256((args.source/name).read_bytes()).hexdigest()==digest
sys.path.insert(0,str(args.source))
from mlx_voice_mil import Graph, GPUProgram
from voice_waveform import prepared_audio, verify_artifact
from voice_profiles import normalized

verify_artifact(args.model)
with wave.open(str(args.wav)) as wav:
    assert wav.getparams()[:3]==(1,2,16000)
    pcm=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(np.float32)/32768
program=GPUProgram(Graph(args.model/'ReDimNet2B6.mlmodelc'),mx)
mx.set_cache_limit(64*1024**2)
rows=[]
for speaker,audio in enumerate((pcm[:64000],pcm[80000:])):
    assert len(audio)>=32000
    start=time.perf_counter()
    output=program.forward(prepared_audio(audio)[None,:]);mx.eval(output)
    vector=normalized(np.asarray(output).reshape(-1).astype(np.float64).tolist(),192)
    rows.append(dict(speaker=speaker,samples=len(audio),vector=vector,seconds=time.perf_counter()-start))
result=dict(source_revision='59e7b0c860f72ca94c24e6eab18ac820b144ee24',
    source_sha256=hashes,
    mlx_version=importlib.metadata.version('mlx'),wav_sha256=hashlib.sha256(args.wav.read_bytes()).hexdigest(),cases=rows)
assert not args.output.exists()
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({r['speaker']:r['seconds'] for r in rows}))
