"""Shared helpers for the v0.6.6 behavior oracle.

Every generator imports the *v0.6.6* implementation from the detached
worktree at .cache/v066 (created by `git worktree add .cache/v066 v0.6.6`),
never from the working tree, and writes deterministic JSON under
tests/goldens/.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKTREE = (ROOT / '.cache' / 'v066').resolve()
ORACLE_TAG = 'v0.6.6'
ORACLE_COMMIT = '91f7593'
GOLDENS = ROOT / 'tests' / 'goldens'


def activate_oracle():
    """Put the worktree's speakerdesk/ on sys.path and refuse to run without it."""
    sources = WORKTREE / 'speakerdesk'
    if not sources.is_dir() or not (sources / 'app.py').is_file():
        raise SystemExit(
            'Oracle sources missing. Create the detached v0.6.6 worktree first:\n'
            '  git worktree add .cache/v066 v0.6.6')
    sys.path.insert(0, str(sources))
    return sources


def imported_from_worktree(*modules):
    """Guard against accidentally importing working-tree modules."""
    for module in modules:
        origin = Path(module.__file__).resolve()
        if not origin.is_relative_to(WORKTREE):
            raise SystemExit(
                f'{module.__name__} resolved to {origin}, outside the v0.6.6 '
                'worktree. The oracle must not import the working tree.')
    return True


def expand(spec):
    """Materialize compact recipes used to describe oversized inputs."""
    if isinstance(spec, dict):
        if '$repeat_segments' in spec:
            entry = spec['$repeat_segments']
            return [expand(entry['segment']) for _ in range(entry['count'])]
        if '$repeat_text' in spec:
            return spec['$repeat_text']['text'] * spec['$repeat_text']['count']
        if '$float' in spec:
            return float(spec['$float'])
        return {key: expand(value) for key, value in spec.items()}
    if isinstance(spec, list):
        return [expand(item) for item in spec]
    return spec


def outcome(callable_, *args, **kwargs):
    """Capture a call result or raised exception deterministically."""
    try:
        return {'value': callable_(*args, **kwargs)}
    except Exception as exc:  # noqa: BLE001 - goldens record failure shapes
        return {'error': type(exc).__name__, 'message': str(exc)}


def dump_golden(area, name, input_, output):
    """Write one {"input": ..., "output": ...} case file, sorted keys, UTF-8."""
    path = GOLDENS / area / f'{name}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps({'input': input_, 'output': output},
                      ensure_ascii=False, sort_keys=True, indent=2)
    path.write_text(text + '\n', encoding='utf-8')
    size = path.stat().st_size
    if size > 200 * 1024:
        raise SystemExit(f'{path} exceeds the 200 KiB golden budget ({size} bytes)')
    print(path.relative_to(ROOT))
