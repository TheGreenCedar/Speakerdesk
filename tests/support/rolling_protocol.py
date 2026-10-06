"""Production JSONL resident loop with test-only CPU models. No GPU/devices."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'speakerdesk'))
import live_refinement
from test_live_refinement import CPUModels

config=json.loads(sys.argv[1]);folder=Path(config['audio_path']).parent
(folder/'worker-started').touch()


class RecordingInput:
    def __iter__(self):
        for line in sys.__stdin__:
            with (folder/'worker-commands.jsonl').open('a') as log:log.write(line)
            yield line


sys.stdin=RecordingInput()
live_refinement.Models=CPUModels
if config.get('test_pause_lag'):
    class LaggedCPUModels(CPUModels):
        def feed(self,audio,final=False):
            turns,end=super().feed(audio,final=final)
            return turns,end if final else max(0,end-.032)
    live_refinement.Models=LaggedCPUModels
def emit(message):
    if (config.get('test_hold_old_refinement') and message['type']=='refinement_result'
            and message['language_epoch']==0):
        (folder/'refinement-response-held').touch()
        deadline=time.monotonic()+10
        while not (folder/'refinement-response-release').exists():
            if time.monotonic()>=deadline:raise RuntimeError('Test refinement response was not released.')
            time.sleep(.01)
    print(json.dumps(message),flush=True)
try:live_refinement.run(config,emit)
except Exception as error:
    emit({'type':'error','error':str(error)});sys.exit(1)
