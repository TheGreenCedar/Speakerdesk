"""Byte-identical MIT MLX Audio 0.5.7 components for CTC forward only.

Keep this package independent of public mlx_audio.stt initialization. In
particular, do not import unrelated STT families or replace public namespaces.
The original generate/from_pretrained methods are not used by the adapter.
"""
from pathlib import Path

BASE_SOURCE_SHA256 = '8f1924827f9fb0d39e56c003bd36c3574e747449518a309f9e0450bd8a407ed5'


def source_files():
    root = Path(__file__).resolve().parent / 'models'
    return {
        'mlx_audio/stt/models/mms/mms.py': root / 'mms/mms.py',
        'mlx_audio/stt/models/wav2vec/wav2vec.py': root / 'wav2vec/wav2vec.py',
    }, root / 'base.py'
