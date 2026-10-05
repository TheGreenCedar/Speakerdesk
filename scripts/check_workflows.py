"""Fail closed if a workflow weakens unsigned-cache / trusted-signing separation."""
import re
from pathlib import Path
import yaml


def validate(directory):
    for path in directory.glob('*.yml'):
        workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        assert workflow['permissions'] == {'contents': 'read'}, f'Excess workflow privilege: {path}'
        assert 'pull_request_target' not in workflow['on'], 'Privileged PR trigger is forbidden'
        assert 'secrets.' not in str(workflow), 'Producer/PR CI cannot access signing credentials'
        for job in workflow['jobs'].values():
            assert not job.get('permissions'), 'Job must inherit read-only permissions'
            for step in job['steps']:
                action = step.get('uses', '')
                if action:
                    assert re.fullmatch(r'[^@]+@[a-f0-9]{40}', action), f'Unpinned action: {action}'
                if action.startswith('actions/cache/save@'):
                    condition = step.get('if', '')
                    clauses = condition.split(' && ')
                    assert "github.ref == 'refs/heads/main'" in clauses
                    assert 'success()' in clauses
                    assert any(clause in clauses for clause in ["github.event_name == 'push'",
                                                               "github.event_name == 'workflow_dispatch'"])
                    assert '||' not in condition
        if path.name == 'apple-build.yml':
            assert set(workflow['on']) == {'workflow_dispatch'}, 'No automatic native builds'
            assert workflow['concurrency']['cancel-in-progress'] == 'false', 'Never cancel producer/release runs'
            paths = {step['with']['path'] for job in workflow['jobs'].values() for step in job['steps']
                     if step.get('uses', '').startswith('actions/cache/')}
            allowed = {'.cache/uv\n', '.cache/npm\n', '~/.cargo/registry\n~/.cargo/git\n',
                       '${{ runner.temp }}/speakerdesk-sccache\n', '.cache/components/runtime\n',
                       '.cache/components/capture\n'}
            assert paths <= allowed, 'Unexpected persisted cache path'
            assert workflow['jobs']['apple']['env']['APPLE_SIGNING_IDENTITY'] == '-'
        if path.name == 'ci.yml':
            assert workflow['on']['push']['branches'] == ['main'], 'Feature pushes duplicate PR CI'
            assert workflow['concurrency']['cancel-in-progress'] == "${{ github.event_name == 'pull_request' }}"


if __name__ == '__main__':
    validate(Path(__file__).resolve().parents[1] / '.github/workflows')
    print('Validated pinned actions, read-only PR/producer jobs and trusted cache writes.')
