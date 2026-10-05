"""Describe private CI artifacts without promoting a public latest version."""
import hashlib
import json
import os
from pathlib import Path

root=Path(__file__).resolve().parents[1]
config=json.loads((root/'desktop/src-tauri/tauri.conf.json').read_text())
version=config['version']
directory=root/'release'
files=[]
for suffix in ['.dmg','.app.zip']:
    path=directory/f'Speakerdesk_{version}_AppleSilicon{suffix}'
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(8*1024**2):digest.update(block)
    files.append({'filename':path.name,'bytes':path.stat().st_size,'sha256':digest.hexdigest()})
notarized=os.getenv('SPEAKERDESK_NOTARIZED')=='1'
manifest={'schema_version':1,'product':'Speakerdesk','version':version,
          'source_commit':os.environ['GITHUB_SHA'],'platform':'macos','architecture':'arm64',
          'minimum_os':config['bundle']['macOS']['minimumSystemVersion'],'channel':'candidate','public_ready':False,
          'signing':'developer-id' if notarized else 'adhoc','notarized':notarized,
          'native_meeting_qa':'pending','files':files}
manifest['producer']={'repository':os.environ['GITHUB_REPOSITORY'],
                      'workflow_path':'.github/workflows/apple-build.yml',
                      'run_id':int(os.environ['GITHUB_RUN_ID']),
                      'run_attempt':int(os.environ['GITHUB_RUN_ATTEMPT'])}
(directory/'artifact-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(directory/'SHA256SUMS').write_text(''.join(f'{item["sha256"]}  {item["filename"]}\n' for item in files))
print(json.dumps(manifest,indent=2))
