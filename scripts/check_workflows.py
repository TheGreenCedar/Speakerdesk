"""Fail closed if a workflow weakens unsigned-cache / trusted-signing separation."""
import re
from pathlib import Path
import yaml

CACHE_ENABLED = "github.event.inputs.cache != 'false'"


def validate(directory):
    for path in directory.glob('*.yml'):
        workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        assert workflow['permissions'] == {'contents': 'read'}, f'Excess workflow privilege: {path}'
        assert 'pull_request_target' not in workflow['on'], 'Privileged PR trigger is forbidden'
        assert 'secrets.' not in str(workflow), 'Producer/PR CI cannot access signing credentials'
        for name, job in workflow['jobs'].items():
            assert not job.get('permissions'), 'Job must inherit read-only permissions'
            if name in {'rust', 'rust-macos', 'apple'}:
                env = job['env']
                assert env['SCCACHE_GHA_ENABLED'] == 'false', 'Only explicit trusted saves may persist objects'
                assert env['CARGO_INCREMENTAL'] == '0', 'Incremental Rust compilation bypasses sccache'
                assert env['RUSTC_WRAPPER'] == "${{ " + CACHE_ENABLED + " && 'sccache' || '' }}", 'Uncached runs must bypass the compiler cache'
            for step in job['steps']:
                action = step.get('uses', '')
                if action:
                    assert re.fullmatch(r'[^@]+@[a-f0-9]{40}', action), f'Unpinned action: {action}'
                if step.get('name') == 'Finish and report compiler cache':
                    assert CACHE_ENABLED in step.get('if', '').split(' && '), 'Uncached runs have no compiler server to stop'
                if action.startswith('actions/cache/'):
                    assert CACHE_ENABLED in step.get('if', '').split(' && '), 'Uncached runs must skip restore/save'
                    paths = set(step['with']['path'].splitlines())
                    allowed = {'~/.cache/pip'} if name == 'source' else {
                        '~/.cargo/registry', '~/.cargo/git', '${{ runner.temp }}/speakerdesk-sccache'}
                    if name == 'apple':
                        allowed |= {'.cache/npm', '.cache/components/capture'}
                    assert paths <= allowed, 'Unexpected persisted cache path'
                if action.startswith('actions/cache/save@'):
                    condition = step.get('if', '')
                    clauses = condition.split(' && ')
                    assert "github.ref == 'refs/heads/main'" in clauses
                    assert 'success()' in clauses
                    assert any(clause in clauses for clause in ["github.event_name == 'push'",
                                                               "github.event_name == 'workflow_dispatch'",
                                                               "contains(fromJSON('[\"push\", \"workflow_dispatch\"]'), github.event_name)"])
                    assert '||' not in condition
                    if name != 'source':
                        assert "steps.budget.outputs.within-limit == 'true'" in clauses, 'Cache saves must respect the byte limit'
        if path.name in {'apple-build.yml', 'ci.yml'}:
            cache_input = workflow['on']['workflow_dispatch']['inputs']['cache']
            assert cache_input['type'] == 'boolean' and cache_input['default'] == 'true'
        if path.name == 'apple-build.yml':
            assert set(workflow['on']) == {'workflow_dispatch'}, 'No automatic native builds'
            assert workflow['concurrency']['cancel-in-progress'] == 'false', 'Never cancel producer/release runs'
            assert workflow['jobs']['apple']['env']['APPLE_SIGNING_IDENTITY'] == '-'
            build = next(step for step in workflow['jobs']['apple']['steps'] if step.get('name') == 'Build candidate without model weights')
            assert build['env']['SPEAKERDESK_CAPTURE_KEY'] == "${{ " + CACHE_ENABLED + " && steps.identity.outputs.capture-key || '' }}", 'Uncached runs must bypass the Swift component'
        if path.name == 'ci.yml':
            assert workflow['on']['push']['branches'] == ['main'], 'Feature pushes duplicate PR CI'
            assert workflow['concurrency']['cancel-in-progress'] == "${{ github.event_name == 'pull_request' }}"


if __name__ == '__main__':
    validate(Path(__file__).resolve().parents[1] / '.github/workflows')
    print('Validated pinned actions, read-only PR/producer jobs and trusted cache writes.')
