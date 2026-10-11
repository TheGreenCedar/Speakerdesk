"""Generate region-planning goldens: speech_crops, decode_regions,
crop_context, utterance_tail_chunks on synthetic turn/utterance inputs.

    PYTHONPATH=.cache/v066/speakerdesk .cache/oracle-venv/bin/python tools/oracle/gen_regions.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_lib as oracle  # noqa: E402

oracle.activate_oracle()
import inference_worker  # noqa: E402
import pipeline  # noqa: E402

oracle.imported_from_worktree(inference_worker, pipeline)


def run():
    # speech_crops: speaker activity windows -> contiguous crops.
    crops_cases = {
        'speech-crops-disjoint': {
            'turns': [{'start': 1.0, 'end': 4.0, 'speaker': 'speaker_0'},
                      {'start': 7.0, 'end': 9.5, 'speaker': 'speaker_1'}],
            'max_seconds': 24.5, 'coverage': None},
        'speech-crops-overlap': {
            'turns': [{'start': 1.0, 'end': 6.0, 'speaker': 'speaker_1'},
                      {'start': 3.0, 'end': 8.0, 'speaker': 'speaker_0'},
                      {'start': 6.5, 'end': 10.0, 'speaker': 'speaker_1'}],
            'max_seconds': 24.5, 'coverage': None},
        'speech-crops-long-turn': {
            'turns': [{'start': 2.0, 'end': 60.0, 'speaker': 'speaker_0'}],
            'max_seconds': 24.5, 'coverage': None},
        'speech-crops-coverage-gap': {
            'turns': [{'start': 5.0, 'end': 8.0, 'speaker': 'speaker_0'},
                      {'start': 20.0, 'end': 24.0, 'speaker': 'speaker_1'}],
            'max_seconds': 24.5, 'coverage': [0.0, 30.0]},
        'speech-crops-adjacent': {
            'turns': [{'start': 0.0, 'end': 3.0, 'speaker': 'speaker_0'},
                      {'start': 3.0, 'end': 6.0, 'speaker': 'speaker_0'}],
            'max_seconds': 24.5, 'coverage': None},
    }
    for name, spec in crops_cases.items():
        oracle.dump_golden(
            'regions', name, spec,
            oracle.outcome(pipeline.speech_crops, spec['turns'],
                           spec['max_seconds'], coverage=spec['coverage']))

    # decode_regions: bridging, envelope splitting, gap handling.
    decode_cases = {
        'decode-regions-basic': {
            'turns': [{'start': 1.0, 'end': 4.0, 'speaker': 'speaker_0'},
                      {'start': 7.0, 'end': 9.0, 'speaker': 'speaker_1'}],
            'max_seconds': 24.5, 'coverage': None},
        'decode-regions-bridge-overlap': {
            # A, a brief A+B flicker, A resumes within TURN_GAP_SECONDS.
            'turns': [{'start': 0.0, 'end': 4.0, 'speaker': 'speaker_0'},
                      {'start': 4.0, 'end': 4.3, 'speaker': 'speaker_0'},
                      {'start': 4.0, 'end': 4.3, 'speaker': 'speaker_1'},
                      {'start': 4.3, 'end': 8.0, 'speaker': 'speaker_0'}],
            'max_seconds': 24.5, 'coverage': None},
        'decode-regions-bridge-silence': {
            # A, a brief unassigned sliver, A resumes.
            'turns': [{'start': 0.0, 'end': 4.0, 'speaker': 'speaker_0'},
                      {'start': 4.5, 'end': 8.0, 'speaker': 'speaker_0'}],
            'max_seconds': 24.5, 'coverage': [0.0, 8.0]},
        'decode-regions-no-bridge-owner-switch': {
            # A -> A+B -> B is a real turn boundary, never bridged.
            'turns': [{'start': 0.0, 'end': 4.0, 'speaker': 'speaker_0'},
                      {'start': 4.0, 'end': 4.3, 'speaker': 'speaker_0'},
                      {'start': 4.0, 'end': 4.3, 'speaker': 'speaker_1'},
                      {'start': 4.3, 'end': 8.0, 'speaker': 'speaker_1'}],
            'max_seconds': 24.5, 'coverage': None},
        'decode-regions-no-bridge-long-gap': {
            'turns': [{'start': 0.0, 'end': 4.0, 'speaker': 'speaker_0'},
                      {'start': 4.0, 'end': 5.0, 'speaker': 'speaker_0'},
                      {'start': 4.0, 'end': 5.0, 'speaker': 'speaker_1'},
                      {'start': 5.0, 'end': 8.0, 'speaker': 'speaker_0'}],
            'max_seconds': 24.5, 'coverage': None},
        'decode-regions-long-envelope': {
            'turns': [{'start': 0.0, 'end': 4.0, 'speaker': 'speaker_0'},
                      {'start': 4.0, 'end': 4.3, 'speaker': 'speaker_0'},
                      {'start': 4.0, 'end': 4.3, 'speaker': 'speaker_1'},
                      {'start': 4.3, 'end': 50.0, 'speaker': 'speaker_0'}],
            'max_seconds': 24.5, 'coverage': None},
    }
    for name, spec in decode_cases.items():
        oracle.dump_golden(
            'regions', name, spec,
            oracle.outcome(pipeline.decode_regions, spec['turns'],
                           spec['max_seconds'], coverage=spec['coverage']))

    # crop_context: padding into silence only.
    context_cases = {
        'crop-context-padded': {
            'chunks': [{'start': 5.0, 'end': 8.0, 'speakers': ['speaker_0']},
                       {'start': 12.0, 'end': 14.0, 'speakers': ['speaker_1']}],
            'duration': 20.0, 'padding': 0.22},
        'crop-context-edges': {
            'chunks': [{'start': 0.1, 'end': 2.0, 'speakers': ['speaker_0']},
                       {'start': 18.0, 'end': 19.9, 'speakers': ['speaker_0']}],
            'duration': 20.0, 'padding': 0.22},
        'crop-context-adjacent-midpoint': {
            'chunks': [{'start': 2.0, 'end': 5.0, 'speakers': ['speaker_0']},
                       {'start': 5.5, 'end': 9.0, 'speakers': ['speaker_1']}],
            'duration': 20.0, 'padding': 0.22},
    }
    for name, spec in context_cases.items():
        # crop_context pads in place and returns None; record the mutation.
        chunks = [dict(chunk) for chunk in spec['chunks']]
        result = oracle.outcome(pipeline.crop_context, chunks,
                                spec['duration'], spec['padding'])
        if 'value' in result:
            result = {'chunks': chunks}
        oracle.dump_golden('regions', name, spec, result)

    # utterance_tail_chunks: continuation admission over an unassigned gap.
    rate = 16000
    tail_cases = {
        'utterance-tail-admit': {
            'chunks': [
                {'start': 0.0, 'end': 10.0, 'speakers': ['speaker_0'],
                 'audio_start': 0.0, 'audio_end': 10.2},
                {'start': 10.0, 'end': 12.0, 'speakers': [],
                 'audio_start': 10.0, 'audio_end': 12.0}],
            'utterances': [{'speech_regions': [
                {'start_sample': 80000, 'end_sample': 170000}],
                'end_sample': 170000}],
            'max_seconds': 18},
        'utterance-tail-reject-owner-overlap': {
            'chunks': [
                {'start': 0.0, 'end': 10.0,
                 'speakers': ['speaker_0', 'speaker_1'],
                 'audio_start': 0.0, 'audio_end': 10.2},
                {'start': 10.0, 'end': 12.0, 'speakers': [],
                 'audio_start': 10.0, 'audio_end': 12.0}],
            'utterances': [{'speech_regions': [
                {'start_sample': 80000, 'end_sample': 170000}],
                'end_sample': 170000}],
            'max_seconds': 18},
        'utterance-tail-reject-following-speech': {
            'chunks': [
                {'start': 0.0, 'end': 10.0, 'speakers': ['speaker_0'],
                 'audio_start': 0.0, 'audio_end': 10.2},
                {'start': 10.0, 'end': 12.0, 'speakers': ['speaker_1'],
                 'audio_start': 10.0, 'audio_end': 12.0}],
            'utterances': [{'speech_regions': [
                {'start_sample': 80000, 'end_sample': 170000}],
                'end_sample': 170000}],
            'max_seconds': 18},
        'utterance-tail-reject-too-long': {
            'chunks': [
                {'start': 0.0, 'end': 10.0, 'speakers': ['speaker_0'],
                 'audio_start': 0.0, 'audio_end': 10.2},
                {'start': 10.0, 'end': 30.0, 'speakers': [],
                 'audio_start': 10.0, 'audio_end': 30.0}],
            'utterances': [{'speech_regions': [
                {'start_sample': 80000, 'end_sample': 300000}],
                'end_sample': 300000}],
            'max_seconds': 18},
        'utterance-tail-reject-inside-bounds': {
            # Utterance ends before the crop boundary: nothing to continue.
            'chunks': [
                {'start': 0.0, 'end': 10.0, 'speakers': ['speaker_0'],
                 'audio_start': 0.0, 'audio_end': 10.2},
                {'start': 10.0, 'end': 12.0, 'speakers': [],
                 'audio_start': 10.0, 'audio_end': 12.0}],
            'utterances': [{'speech_regions': [
                {'start_sample': 80000, 'end_sample': 150000}],
                'end_sample': 150000}],
            'max_seconds': 18},
        'utterance-tail-admit-drop-gap': {
            # Utterance covers the whole gap: the unassigned chunk is removed.
            'chunks': [
                {'start': 0.0, 'end': 10.0, 'speakers': ['speaker_0'],
                 'audio_start': 0.0, 'audio_end': 10.2},
                {'start': 10.0, 'end': 12.0, 'speakers': [],
                 'audio_start': 10.0, 'audio_end': 12.0}],
            'utterances': [{'speech_regions': [
                {'start_sample': 80000, 'end_sample': 192000}],
                'end_sample': 192000}],
            'max_seconds': 18},
    }
    for name, spec in tail_cases.items():
        oracle.dump_golden(
            'regions', name, spec,
            oracle.outcome(inference_worker.utterance_tail_chunks,
                           spec['chunks'], spec['utterances'], rate=rate,
                           max_seconds=spec['max_seconds']))


if __name__ == '__main__':
    run()
