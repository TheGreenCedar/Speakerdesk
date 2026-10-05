"""Speakerdesk packaging: import diarization models on demand.

The upstream initializer eagerly imports unrelated Smart Turn / STT models.
Only this initializer is changed; NVIDIA model implementations are unchanged.
"""
