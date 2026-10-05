"""Speakerdesk bundles only Whisper's local language-detection backend.

Avoid eager imports of unrelated STT architectures and excluded dependencies.
The pinned Whisper implementation itself is unchanged (mlx-audio, MIT).
"""
from . import whisper
