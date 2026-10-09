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
if config.get('test_canonical_peer'):
    from test_canonical_runtime import Peer
    def canonical_cpu_models(cfg):
        # Fabricated typed receipts exercise JSONL/host validation only. This
        # explicitly substituted CPU peer never qualifies acoustic acceptance.
        import uuid
        from admission_receipt import execution,retained_pcm_digest
        peer=Peer();peer.admission_execution=execution(cfg['job_id'],uuid.uuid4().hex)
        def inspection(phase,request_id):
            end=peer.speech_live.evidence.end_sample
            return {**peer.admission_execution,'phase':phase,'request_id':request_id,
                'inspection_state':'observed_prefix','start_sample':0,'end_sample':end,
                'received_sample':peer.received,'closed':phase=='stop',
                'audio_encoding':'pcm_s16le','pcm_sha256':retained_pcm_digest(cfg['audio_path'],end),
                'speech_samples':end,'uncertain_samples':0,'negative_constant_samples':0,'model_negative_samples':0,'decision':'speech'}
        peer.inspection_receipt=inspection
        peer.for_source=lambda source_config:canonical_cpu_models(source_config)
        return peer
    live_refinement.Models=canonical_cpu_models
else:
    # Retain the explicit legacy rolling-window regression contracts. The new
    # canonical peer lane below exercises the production activation separately.
    config['canonical_utterances']=False
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
