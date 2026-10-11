"""Validation only: unchanged pinned Python conversion and CTC forward on public fixtures."""
import argparse, hashlib, importlib.metadata, json, sys, time, wave
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
for name in ('source', 'model', 'wav', 'output'):
    p.add_argument(name, type=Path)
args = p.parse_args()
sys.path.insert(0, str(args.source))
from alignment_model import _convert, manifest, digest_file, MODEL_SHA256, DERIVED_DIRECTORY
from mlx_ctc_components import source_files
from mlx_ctc_components.models.mms.mms import Model
from mlx_ctc_components.models.wav2vec.wav2vec import ModelConfig
from mlx_ctc_forward import forward_scores

metadata = manifest()
for name, path in source_files()[0].items():
    assert digest_file(path) == metadata['runtime']['component_source_sha256'][name]
assert digest_file(args.model/'model.int8.onnx') == metadata['source_model_sha256']
assert digest_file(args.model/'tokens.txt') == metadata['tokens_sha256']
folder = args.model/DERIVED_DIRECTORY
folder.mkdir(exist_ok=True)
weights = folder/'weights.npz'
if not weights.exists():
    _convert(args.model/'model.int8.onnx', weights, metadata)
assert digest_file(weights) == MODEL_SHA256
mx.set_default_device(mx.gpu)
mx.set_cache_limit(0)
model = Model(ModelConfig.from_dict(metadata['config']))
model.wav2vec2.encoder.pos_conv_embed.conv = nn.Conv1d(1024, 1024, 128, padding=64, groups=16, bias=True)
model.load_weights(list(mx.load(str(weights)).items()), strict=True)
model.eval()
mx.eval(model.parameters())
with wave.open(str(args.wav)) as wav:
    assert wav.getparams()[:3] == (1, 2, 16000)
    pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2').astype(np.float32)/32768
ids = [27, 0, 9811, 4, 7, 112, 29, 83, 53, 18]
rows = []
for name, audio in [('whole', pcm), ('400', pcm[16000:16400]), ('719', pcm[16000:16719]),
                    ('720', pcm[16000:16720]), ('partial_tail', pcm[16000:16549]), ('silence', np.zeros(720, dtype=np.float32))]:
    start = time.perf_counter()
    scores = forward_scores(model, audio)
    selected = scores[:, mx.array(ids)]
    mx.eval(selected)
    rows.append(dict(name=name, samples=len(audio), frames=scores.shape[0], gathered=selected.tolist(), seconds=time.perf_counter()-start))
result = dict(model_sha256=MODEL_SHA256, tokens_sha256=metadata['tokens_sha256'],
              forward_sha256=digest_file(args.source/'mlx_ctc_forward.py'),
              components=metadata['runtime']['component_source_sha256'],
              mlx_version=importlib.metadata.version('mlx'), wav_sha256=digest_file(args.wav), token_ids=ids, cases=rows)
args.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps([{k: v for k, v in row.items() if k != 'gathered'} for row in rows], indent=2))
