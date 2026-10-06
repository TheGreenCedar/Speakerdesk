"""Production JSONL resident loop with test-only CPU models. No GPU/devices."""
import json
from pathlib import Path
import sys
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
def emit(message):print(json.dumps(message),flush=True)
try:live_refinement.run(config,emit)
except Exception as error:
    emit({'type':'error','error':str(error)});sys.exit(1)
