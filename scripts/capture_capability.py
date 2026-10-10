"""Publish the reviewed route capability bound to final signed helper bytes."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

helper=Path(sys.argv[1]).resolve(strict=True)
response=subprocess.run([str(helper),'--check'],capture_output=True,text=True,check=True,timeout=5)
info=json.loads(response.stdout)
if (info.get('capture_started') is not False
        or info.get('capability_version') != 'apple_input_node_output_route_v1'
        or info.get('export_contract') != 'apple_processed_mono16k_v1'):
    raise ValueError('Helper does not implement the reviewed Apple route contract.')
entry={'helper_sha256':hashlib.sha256(helper.read_bytes()).hexdigest(),
       'export_contract':info['export_contract'],'capability_version':info['capability_version']}
path=Path(str(helper)+'.capability.json')
path.write_text(json.dumps(entry,sort_keys=True)+'\n')
