"""Generate pcm16_bytes goldens: quantization, clipping, rounding, rejection.

    PYTHONPATH=.cache/v066/speakerdesk .cache/oracle-venv/bin/python tools/oracle/gen_audio.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import oracle_lib as oracle  # noqa: E402

oracle.activate_oracle()
import audio  # noqa: E402

oracle.imported_from_worktree(audio)


def pcm16(values):
    """Call the oracle and record output bytes as hex (or the raised error)."""
    result = oracle.outcome(audio.pcm16_bytes, values)
    if 'value' in result:
        result = {'pcm16_hex': result['value'].hex()}
    return result


def run():
    oracle.dump_golden(
        'audio', 'pcm16-quantization',
        {'audio': [0.0, 0.5, -0.5, 0.25, -0.25,
                   1.0 / 32768, -1.0 / 32768, 0.1]},
        pcm16([0.0, 0.5, -0.5, 0.25, -0.25,
               1.0 / 32768, -1.0 / 32768, 0.1]))

    oracle.dump_golden(
        'audio', 'pcm16-clipping',
        {'audio': [-1.0, 1.0, -1.5, 1.5, -0.99999, 0.99999]},
        pcm16([-1.0, 1.0, -1.5, 1.5, -0.99999, 0.99999]))

    # np.rint is round-half-to-even: x*32768 at exact .5 boundaries.
    oracle.dump_golden(
        'audio', 'pcm16-rounding-half-even',
        {'audio': [0.5 / 32768, 1.5 / 32768, 2.5 / 32768,
                   -0.5 / 32768, -1.5 / 32768, 3.5 / 32768]},
        pcm16([0.5 / 32768, 1.5 / 32768, 2.5 / 32768,
               -0.5 / 32768, -1.5 / 32768, 3.5 / 32768]))

    oracle.dump_golden(
        'audio', 'pcm16-quiet-integer',
        {'audio': [1.0 / 32768, 2.0 / 32768, -3.0 / 32768]},
        pcm16([1.0 / 32768, 2.0 / 32768, -3.0 / 32768]))

    oracle.dump_golden(
        'audio', 'pcm16-reject-nonfinite',
        {'audio': [0.0, {'$float': 'nan'}, 0.5]},
        oracle.outcome(audio.pcm16_bytes, [0.0, float('nan'), 0.5]))

    oracle.dump_golden(
        'audio', 'pcm16-reject-infinite',
        {'audio': [{'$float': 'inf'}]},
        oracle.outcome(audio.pcm16_bytes, [float('inf')]))

    oracle.dump_golden(
        'audio', 'pcm16-reject-non-mono',
        {'audio': [[0.0, 0.5], [0.25, -0.5]]},
        oracle.outcome(audio.pcm16_bytes, [[0.0, 0.5], [0.25, -0.5]]))


if __name__ == '__main__':
    run()
