"""Disposable IPC fixture. No audio reads, native imports or neural computation."""
import json
import os
import sys
import subprocess
import time
import threading
from pathlib import Path

if os.environ.get('SPEAKERDESK_FAKE_VOICE_MODE') == 'bootloader_hang_second':
    child = subprocess.Popen([sys.executable, '-u', __file__, sys.argv[1]],
        env={**os.environ, 'SPEAKERDESK_FAKE_VOICE_MODE':'hang_second'})
    sys.exit(child.wait())

init = json.loads(sys.argv[1])
model = json.loads(Path(init['config_path']).read_text())['model']
print(json.dumps({'type':'ready', 'model':model}), flush=True)
count = 0
for line in sys.stdin:
    request = json.loads(line)
    if request['op'] == 'shutdown':break
    count += 1
    mode = os.environ.get('SPEAKERDESK_FAKE_VOICE_MODE')
    if mode == 'exit':sys.exit(1)
    if mode == 'hang' or (mode == 'hang_second' and count == 2):time.sleep(30)
    if mode == 'oversized':print('x'*70000, flush=True);continue
    response = {'type':'embedding', 'id':request['id']+(1 if mode == 'wrong_id' else 0),
        'owner_thread_id':threading.get_ident(),
        'model':model, 'clean':True, 'vector':[1.]+[0.]*(model['dimension']-1),
        'metrics':{'completed_GPU_declarations':809, 'CPU_neural_retry':False,
                   'fixture_request_count':count, 'zero_neural_fixture':True}}
    print(json.dumps(response), flush=True)
