"""Sequential offline MLX workers for NVIDIA diarization and Cohere transcription."""
import json
import math
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from audio import crop

ROOT = Path(__file__).resolve().parent
MODELS = {'nemotron': ('nvidia/Nemotron-3-Diarization', 8),
          'sortformer': ('nvidia/diar_streaming_sortformer_4spk-v2.1', 4)}
_worker_lock = threading.Lock()
_workers = set()
_stopping = False


def shutdown_workers():
    """Release model memory when the desktop application closes."""
    global _stopping
    with _worker_lock:
        _stopping = True
        workers = list(_workers)
    for worker in workers:
        worker.terminate()
    for worker in workers:
        try:
            worker.wait(timeout=3)
        except subprocess.TimeoutExpired:
            worker.kill()


def model_config():
    home=ROOT.parent
    models=Path(os.getenv('SPEAKERDESK_MODELS',home/'models'))
    return {'diar_path':os.getenv('DIAR_MODEL_PATH',str(models/'nemotron')),
            'cohere_path':os.getenv('COHERE_MODEL_PATH',str(models/'cohere-speech')),
            'diar_python':os.getenv('DIAR_PYTHON',str(home/'.venv/bin/python')),
            'asr_python':os.getenv('ASR_PYTHON',str(home/'.venv-asr/bin/python')),
            'diar_kind':'nemotron', 'device':'mlx'}


def preflight(config):
    issues=[]
    for name,expected in [('diar_path','nemotron_diarization'),('cohere_path','cohere_asr')]:
        path=Path(config[name])
        try:
            metadata=json.loads((path/'config.json').read_text())
            if metadata.get('model_type')!=expected or not (path/'model.safetensors').is_file():
                raise ValueError('Incorrect model format')
        except (OSError,ValueError):
            issues.append(f'Missing or incompatible local {expected} MLX checkpoint.')
    for name in ['diar_python','asr_python']:
        if not Path(config[name]).is_file():
            issues.append(f'Set {name.upper()} to an inference environment Python executable.')
    if config['device'] != 'mlx':
        issues.append('This Mac setup uses MLX on Apple Silicon.')
    return issues


def parse_turns(lines, duration, speaker_limit):
    """NeMo contract: one file -> list of 'start end speaker_N' lines."""
    turns=[]
    for line in lines:
        if not isinstance(line,str) or len(line.split()) != 3:
            raise ValueError('Unexpected NVIDIA diarization output format.')
        a,b,speaker=line.split()
        start,end=float(a),float(b)
        match=re.fullmatch(r'speaker_(\d+)',speaker)
        if not match or int(match[1]) >= speaker_limit or not all(map(math.isfinite,(start,end))):
            raise ValueError('Invalid NVIDIA speaker or time output.')
        start=max(0,start);end=min(duration,end)
        if end > start:
            turns.append({'start':start,'end':end,'speaker':speaker})
    return sorted(turns,key=lambda t:(t['start'],t['end']))


def speech_crops(turns, max_seconds=24.5):
    """Keep simultaneous speakers as a shared region: never invent text ownership."""
    boundaries=sorted({t[k] for t in turns for k in ('start','end')})
    regions=[]
    for start,end in zip(boundaries,boundaries[1:]):
        active=sorted({t['speaker'] for t in turns if t['start'] < end and t['end'] > start})
        if not active:
            continue
        if regions and regions[-1]['speakers']==active and abs(regions[-1]['end']-start)<1e-6:
            regions[-1]['end']=end
        else:
            regions.append({'start':start,'end':end,'speakers':active})
    chunks=[]
    for region in regions:
        start=region['start']
        while start < region['end']-1e-6:
            end=min(start+max_seconds,region['end'])
            chunks.append({'start':start,'end':end,'speakers':region['speakers']})
            start=end
    return chunks


def crop_context(chunks,duration,padding=0.22):
    """Pad into adjacent silence only, never across another speech region."""
    for i,chunk in enumerate(chunks):
        previous=chunks[i-1]['end'] if i else 0
        following=chunks[i+1]['start'] if i+1<len(chunks) else duration
        chunk['audio_start']=max(0,chunk['start']-padding,(previous+chunk['start'])/2)
        chunk['audio_end']=min(duration,chunk['end']+padding,(following+chunk['end'])/2)


def run_worker(python, task, request, folder):
    request_path=folder/f'{task}-request.json'; output_path=folder/f'{task}-output.json'
    request_path.write_text(json.dumps(request))
    env={**os.environ,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_DATASETS_OFFLINE':'1',
         'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'}
    try:
        command=([sys.executable,'--worker',task,str(request_path),str(output_path)] if getattr(sys,'frozen',False)
                 else [python,str(ROOT/'inference_worker.py'),task,str(request_path),str(output_path)])
        with _worker_lock:
            if _stopping:
                raise RuntimeError('Speakerdesk is closing.')
            proc=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            _workers.add(proc)
        try:
            stdout,stderr=proc.communicate(timeout=7200)
        except subprocess.TimeoutExpired:
            proc.kill();proc.communicate()
            raise
        finally:
            with _worker_lock:_workers.discard(proc)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f'{task} exceeded the two-hour inference limit.') from None
    log_path=folder/f'{task}-worker.log'
    log_path.write_text(stdout+'\n'+stderr)
    log_path.chmod(0o600)
    if proc.returncode:
        # The worker writes a safe diagnostic; dependency tracebacks stay off the UI.
        detail=json.loads(output_path.read_text()).get('error') if output_path.exists() else None
        raise RuntimeError((detail or f'{task} worker could not start. Check the inference environment.')+f' Diagnostic log: {log_path}')
    return json.loads(output_path.read_text())


def infer(audio_path, duration, language, folder, progress, config=None):
    config=config or model_config()
    issues=preflight(config)
    if issues:
        raise RuntimeError(' '.join(issues))
    model_id,limit=MODELS[config['diar_kind']]
    progress('Labeling speakers with NVIDIA…')
    result=run_worker(config['diar_python'],'diarize',{'model_path':config['diar_path'],
        'model_kind':config['diar_kind'],'audio':str(audio_path),'device':config['device']},folder)
    turns=parse_turns(result['turns'],duration,limit)
    if not turns:
        raise RuntimeError('NVIDIA found no speech. No transcript was generated.')
    chunks=speech_crops(turns)
    crop_context(chunks,duration)
    if len(chunks)>10000:
        raise RuntimeError('Too many speaker transitions. Split the recording into smaller files.')
    progress(f'Transcribing {len(chunks)} speech regions with Cohere…')
    crop_dir=folder/'crops';crop_dir.mkdir(exist_ok=True)
    try:
        for i,chunk in enumerate(chunks):
            path=crop_dir/f'{i}.wav';crop(audio_path,path,chunk['audio_start'],chunk['audio_end']);chunk['audio']=str(path)
        asr=run_worker(config['asr_python'],'transcribe',{'model_path':config['cohere_path'],
            'chunks':chunks,'language':language,'device':config['device']},folder)
    finally:
        for path in crop_dir.glob('*.wav'):path.unlink()
        crop_dir.rmdir()
    if len(asr['texts']) != len(chunks):
        raise RuntimeError('Cohere returned an incomplete transcript.')
    speakers={s:f'Speaker {int(s.split("_")[1])+1}' for s in sorted({t['speaker'] for t in turns})}
    segments=[]
    for i,(chunk,text) in enumerate(zip(chunks,asr['texts'])):
        if len(chunk['speakers'])>1:
            speaker='overlap';speakers[speaker]='Overlapping speakers'
        else:speaker=chunk['speakers'][0]
        segments.append({'id':f'seg-{i}','start':chunk['start'],'end':chunk['end'],'speaker':speaker,
                         'text':text,'speaker_candidates':chunk['speakers'], 'voice_eligible':len(chunk['speakers'])==1,
                         'review':len(chunk['speakers'])>1 or chunk['end']-chunk['start']<0.5,
                         'timing':'audio_crop','confidence':None})
    return {'schema_version':1,'speakers':speakers,'segments':segments,'diarization':turns,
            'provenance':{'kind':'local_inference','asr_model':'CohereLabs/cohere-transcribe-03-2026',
                          'diarization_model':model_id,'language':language,'device':config['device'],
                          'timing':'NVIDIA speech regions split into ≤25-second crops; no word alignment',
                          'speaker_limit':limit,
                          'diarization_checkpoint':'mlx-community/Nemotron-3-Diarization@59ed2dbfc1346dcea9d423c71306a3a2499c568f',
                          'asr_checkpoint':'spokedotso/cohere-transcribe-03-2026-mlx-4bit@064e51eab6db47066cbeaa85e2894b5691bd8d12',
                          'metrics':{'diarization':result.get('metrics',{}),'asr':asr.get('metrics',{})},
                          'libraries':{'diarization':result.get('versions',{}),'asr':asr.get('versions',{})}},
            'warnings':['Times are speech crop boundaries, not word timestamps. Review subtitle timing.',
                        'ASR crops include up to 0.22 seconds of neighboring silence to preserve word endings.',
                        'Overlapping speech is mixed audio; text cannot be reliably assigned to an individual speaker.',
                        f'This diarizer supports at most {limit} speakers; excess speakers cannot be reliably detected.']}
