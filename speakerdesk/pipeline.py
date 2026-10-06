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
from language_detection import detector_issues, LID_CHECKPOINT
from speech_admission import speech_issues, SILERO_SPEC
from transcript import LANGUAGES
from review import retain_unassigned_audio

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
            'lid_path':os.getenv('LID_MODEL_PATH',str(models/'whisper-language')),
            'speech_path':os.getenv('SPEECH_MODEL_PATH',str(models/'silero-speech')),
            'alignment_path':os.getenv('ALIGNMENT_MODEL_PATH',str(models/'coarse-alignment')),
            'diar_python':os.getenv('DIAR_PYTHON',str(home/'.venv/bin/python')),
            'asr_python':os.getenv('ASR_PYTHON',str(home/'.venv-asr/bin/python')),
            'diar_kind':'nemotron', 'device':'mlx'}


def preflight(config, language=None):
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
    issues.extend(speech_issues(config.get('speech_path','')))
    if language == 'auto':
        issues.extend(detector_issues(config.get('lid_path', '')))
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


def speech_crops(turns, max_seconds=24.5, *, coverage=None):
    """Keep simultaneous speakers as a shared region: never invent text ownership."""
    boundaries=sorted({t[k] for t in turns for k in ('start','end')})
    if coverage is not None:
        first,last=coverage
        boundaries=sorted({first,last,*[x for x in boundaries if first<x<last]})
    regions=[]
    for start,end in zip(boundaries,boundaries[1:]):
        active=sorted({t['speaker'] for t in turns if t['start'] < end and t['end'] > start})
        if not active and coverage is None:
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


TURN_GAP_SECONDS = .65


def activity_slice(regions,start,end):
    return [{**part,'start':max(start,part['start']),'end':min(end,part['end']),
             'speakers':list(part['speakers'])}
            for part in regions if part['start']<end and part['end']>start]


def decode_regions(turns, max_seconds=24.5, *, coverage=None):
    """Continuous ASR envelopes; activity boundaries are not word boundaries.

    Bridge only an internal brief gap flanked by the same single speaker.
    Keep the original activity ledger: a bridged envelope is not clean voice
    enrollment evidence. Real switches, long pauses and ownership edges remain.
    """
    import copy
    regions=speech_crops(turns,max_seconds=100000,coverage=coverage)
    envelopes=[];index=0
    while index<len(regions):
        region=copy.deepcopy(regions[index]);index+=1
        owner=region['speakers'];ledger=[copy.deepcopy(region)]
        while index+1<len(regions):
            middle,following=regions[index:index+2]
            # A brief overlapping activation may share context only when the
            # SAME clean speaker resumes. A -> A+B -> B is a real turn boundary.
            compatible=(not middle['speakers'] or
                        (len(middle['speakers'])>1 and set(owner)<=set(middle['speakers'])))
            if (len(owner)!=1 or not compatible or following['speakers']!=owner
                    or middle['end']-middle['start']>TURN_GAP_SECONDS
                    or abs(middle['start']-region['end'])>1e-6
                    or abs(following['start']-middle['end'])>1e-6):break
            ledger.extend(copy.deepcopy([middle,following]));region['end']=following['end']
            region['speakers']=sorted(set(region['speakers'])|set(middle['speakers']));index+=2
        if len(ledger)>1:region['activity_regions']=ledger
        envelopes.append(region)
    if max_seconds is None:return envelopes
    chunks=[]
    for region in envelopes:
        start=region['start']
        while start<region['end']-1e-6:
            end=min(start+max_seconds,region['end']);chunk={**region,'start':start,'end':end}
            if 'activity_regions' in region:
                chunk['activity_regions']=activity_slice(region['activity_regions'],start,end)
            chunks.append(chunk);start=end
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
    issues=preflight(config, language)
    if issues:
        raise RuntimeError(' '.join(issues))
    model_id,limit=MODELS[config['diar_kind']]
    progress('Labeling speakers with NVIDIA…')
    result=run_worker(config['diar_python'],'diarize',{'model_path':config['diar_path'],
        'model_kind':config['diar_kind'],'audio':str(audio_path),'device':config['device']},folder)
    turns=parse_turns(result['turns'],duration,limit)
    chunks=decode_regions(turns, max_seconds=18,coverage=(0.,duration))
    crop_context(chunks,duration,padding=0 if language == 'auto' else .22)
    if len(chunks)>10000:
        raise RuntimeError('Too many speaker transitions. Split the recording into smaller files.')
    progress(f'Transcribing {len(chunks)} audio regions with Cohere…')
    crop_dir=folder/'crops';crop_dir.mkdir(exist_ok=True)
    try:
        for i,chunk in enumerate(chunks):
            path=crop_dir/f'{i}.wav';crop(audio_path,path,chunk['audio_start'],chunk['audio_end']);chunk['audio']=str(path)
        asr=run_worker(config['asr_python'],'transcribe',{'model_path':config['cohere_path'],
            'chunks':chunks,'language':language,'lid_path':config.get('lid_path'),
            'speech_path':config.get('speech_path'),'audio':str(audio_path),
            'device':config['device']},folder) if chunks else {'regions':[],'metrics':{}}
    finally:
        for path in crop_dir.glob('*.wav'):path.unlink()
        crop_dir.rmdir()
    if len(asr['regions']) != len(chunks):
        raise RuntimeError('Cohere returned an incomplete transcript.')
    speakers={s:f'Speaker {int(s.split("_")[1])+1}' for s in sorted({t['speaker'] for t in turns})}
    segments=[]
    for chunk,regions in zip(chunks,asr['regions']):
        if len(chunk['speakers'])>1:
            speaker='overlap';speakers[speaker]='Overlapping speakers'
        elif chunk['speakers']:speaker=chunk['speakers'][0]
        else:speaker='unassigned';speakers[speaker]='Unknown speaker'
        for region in regions:
            start=max(chunk['start'],chunk['audio_start']+region['start'])
            end=min(chunk['end'],chunk['audio_start']+region['end'])
            if end <= start:continue
            segments.append({**region,'id':f'seg-{len(segments)}','start':start,'end':end,'speaker':speaker,
                             'speaker_candidates':chunk['speakers'],'voice_eligible':len(chunk['speakers'])==1 and not region.get('audio_state') and not chunk.get('activity_regions'),
                             **({'activity_regions':activity_slice(chunk['activity_regions'],start,end)} if chunk.get('activity_regions') else {}),
                             'review':region['review'] or len(chunk['speakers'])!=1 or end-start<.5,
                             'timing':'audio_crop','confidence':None})
    return retain_unassigned_audio({'schema_version':1,'speakers':speakers,'segments':segments,'diarization':turns,
            'provenance':{'kind':'local_inference','asr_model':'CohereLabs/cohere-transcribe-03-2026',
                          'diarization_model':model_id,'language':language,'device':config['device'],
                          'language_detector':LID_CHECKPOINT if language=='auto' else None,
                          'speech_detector':SILERO_SPEC['repo']+'@'+SILERO_SPEC['revision'],
                          'timing':('Retained audio with ≤3-second language probes and ≤6-second ASR crops; no word alignment' if language=='auto' else 'Retained audio split into ≤25-second crops; no word alignment'),
                          'speaker_limit':limit,
                          'diarization_checkpoint':'mlx-community/Nemotron-3-Diarization@59ed2dbfc1346dcea9d423c71306a3a2499c568f',
                          'asr_checkpoint':'spokedotso/cohere-transcribe-03-2026-mlx-4bit@064e51eab6db47066cbeaa85e2894b5691bd8d12',
                          'metrics':{'diarization':result.get('metrics',{}),'asr':asr.get('metrics',{})},
                          'libraries':{'diarization':result.get('versions',{}),'asr':asr.get('versions',{})}},
            'warnings':['Times are speech crop boundaries, not word timestamps. Review subtitle timing.',
                        ('Uncertain language uses recent established context or a marked supported-language guess when available. These attempts can be wrong; audio without a supported language remains available for review.' if language=='auto' else 'The selected language applies to every audio crop, including speech without a detected speaker.'),
                        'Overlapping speech is mixed audio; text cannot be reliably assigned to an individual speaker.',
                        f'This diarizer supports at most {limit} speakers; excess speakers cannot be reliably detected.']}, duration)


def retry_passage(audio_path, start, end, language, folder, config=None):
    """One explicitly chosen language/crop; no speaker relabeling or full rerun."""
    if language not in LANGUAGES or not all(map(math.isfinite, (start,end))) or not 0 <= start < end or end-start > 24.5:
        raise ValueError('Choose a supported language and a passage of at most 24.5 seconds.')
    config = config or model_config()
    metadata = json.loads((Path(config['cohere_path'])/'config.json').read_text())
    if (metadata.get('model_type') != 'cohere_asr'
            or not (Path(config['cohere_path'])/'model.safetensors').is_file()
            or not Path(config['asr_python']).is_file() or config['device'] != 'mlx'):
        raise ValueError('Finish local transcription model setup before retrying this passage.')
    if speech_issues(config.get('speech_path','')):
        raise ValueError('Finish local speech detection model setup before retrying this passage.')
    import tempfile
    with tempfile.TemporaryDirectory(prefix='passage-retry-', dir=folder) as temporary:
        target = Path(temporary); audio = target/'passage.wav'
        crop(audio_path,audio,start,end)
        result = run_worker(config['asr_python'],'transcribe',{'model_path':config['cohere_path'],
            'chunks':[{'audio':str(audio),'speakers':['manual_retry']}],
            'language':language,'device':'mlx','speech_path':config['speech_path'],
            'audio':str(audio)},target)
        regions = result['regions']
        if len(regions) != 1 or len(regions[0]) != 1:
            raise RuntimeError('The retry returned an incomplete passage result.')
        return {**regions[0][0], 'start':start, 'end':end}
