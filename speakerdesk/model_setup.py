"""Pinned public model downloads with progress, resumption and checksum checks."""
import hashlib
import os
import shutil
import ssl
import threading
import urllib.request
import certifi
from pathlib import Path
from flask import abort, jsonify, request
from language_detection import LID_SPEC
from speech_admission import SILERO_SPEC
from alignment_artifact import ALIGNMENT_SPEC
from alignment_setup import AlignmentSetup, save_enabled, enabled
from voice_setup import VoiceSetup
from voice_profiles import runtime_available

SPECS=[
    {'name':'Speaker separation','repo':'mlx-community/Nemotron-3-Diarization','revision':'59ed2dbfc1346dcea9d423c71306a3a2499c568f',
     'directory':'nemotron','bytes':198632220,'sha256':'21e8427d1795c9c46c5800f56b16061734ffd0dcadd71d9bcf0b4d6ef7261da5',
     'files':['config.json','model.safetensors','README.md']},
    {'name':'Transcription','repo':'spokedotso/cohere-transcribe-03-2026-mlx-4bit','revision':'064e51eab6db47066cbeaa85e2894b5691bd8d12',
     'directory':'cohere-speech','bytes':1505759837,'sha256':'ae947c13ba1cb8ce24c8bf72ded2520dcda1894ebf5a54fbb803c03bc28ef7fb',
     'files':['config.json','model.safetensors','generation_config.json','preprocessor_config.json','processor_config.json','special_tokens_map.json','tokenizer.json','tokenizer.model','tokenizer_config.json','README.md']},
    LID_SPEC,
    SILERO_SPEC,
    ALIGNMENT_SPEC,
]


def register_setup(app, activate_voice):
    root=Path(os.getenv('SPEAKERDESK_MODELS',Path(__file__).resolve().parents[1]/'models'))
    lock=threading.RLock()
    updates=app.extensions['speakerdesk']['updates']
    verified_files={}
    state={'status':'idle','phase':'','downloaded_bytes':0,'total_bytes':sum(s.get('download_bytes',s['bytes']) for s in SPECS),'error':None}
    alignment=AlignmentSetup(root,lock,app.logger)
    app.extensions['speakerdesk']['alignment_setup']=alignment
    voice=VoiceSetup(root,lock,activate_voice,app.logger)
    state['total_bytes']+=voice.state['total_bytes'] if voice.released() else 0
    app.extensions['speakerdesk']['voice_setup']=voice

    def voice_available():
        backend,calibration=app.extensions['speakerdesk'].get('voice_runtime',(None,None))
        return runtime_available(backend,calibration)

    def verified(path, digest):
        stamp=path.stat()
        signature=(stamp.st_size,stamp.st_mtime_ns,stamp.st_ino,digest)
        cached=verified_files.get(path)
        if cached and cached[0]==signature:return cached[1]
        with path.open('rb') as content:
            matches=hashlib.file_digest(content,'sha256').hexdigest()==digest
        verified_files[path]=(signature,matches)
        return matches

    def source_installed(spec):
        folder=root/spec['directory']
        weight=spec.get('weight_file','model.safetensors')
        return (all((folder/name).is_file() for name in spec['files'])
                and (folder/weight).stat().st_size==spec['bytes']
                and all(verified(folder/name,digest)
                        for name,digest in ({weight:spec['sha256']} if spec.get('sha256') else {}).items())
                and all(verified(folder/name,digest)
                        for name,digest in spec.get('file_sha256',{}).items()))

    def installed(spec):
        if spec['directory']==ALIGNMENT_SPEC['directory']:return alignment.ready()
        return source_installed(spec)

    def update(**changes):
        with lock:state.update(changes)

    def download(include_alignment=True, alignment_only=False):
        try:
            root.mkdir(parents=True,exist_ok=True)
            selected=[] if alignment_only else [s for s in SPECS if s['directory']!=ALIGNMENT_SPEC['directory'] and not s.get('optional')]
            missing=sum(s['bytes'] for s in selected if not source_installed(s))
            if not alignment_only and voice.released() and not voice.installed():missing+=voice.state['total_bytes']
            if not alignment_only and shutil.disk_usage(root).free < missing+512*1024**2:
                raise RuntimeError('There is not enough free disk space. Free at least 2.4 GB and retry.')
            finished=0
            for spec in selected:
                if source_installed(spec):
                    finished+=spec['bytes'];continue
                folder=root/spec['directory'];folder.mkdir(exist_ok=True)
                weight=spec.get('weight_file','model.safetensors')
                update(phase=spec['name'])
                for name in spec['files']:
                    dest=folder/name
                    if dest.exists() and name!=weight:
                        expected=spec.get('file_sha256',{}).get(name)
                        if not expected or hashlib.sha256(dest.read_bytes()).hexdigest()==expected:continue
                    partial=folder/(name+'.part')
                    offset=partial.stat().st_size if partial.exists() else 0
                    url=f'https://huggingface.co/{spec["repo"]}/resolve/{spec["revision"]}/{name}?download=true'
                    headers={'Range':f'bytes={offset}-'} if offset else {}
                    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=60,
                                                context=ssl.create_default_context(cafile=certifi.where())) as response:
                        if response.status!=206:offset=0
                        with partial.open('ab' if offset else 'wb') as output:
                            while block:=response.read(1024**2):
                                output.write(block);offset+=len(block)
                                if name==weight:update(downloaded_bytes=finished+offset)
                    if name==weight:
                        update(phase=f'Verifying {spec["name"].lower()}')
                        digest=hashlib.sha256()
                        with partial.open('rb') as content:
                            while block:=content.read(8*1024**2):digest.update(block)
                        if partial.stat().st_size!=spec['bytes'] or digest.hexdigest()!=spec['sha256']:
                            partial.unlink(missing_ok=True)
                            raise RuntimeError('Model verification failed. Retry the download.')
                    elif name in spec.get('file_sha256',{}):
                        if hashlib.sha256(partial.read_bytes()).hexdigest()!=spec['file_sha256'][name]:
                            partial.unlink(missing_ok=True)
                            raise RuntimeError('Model metadata verification failed. Retry the download.')
                    partial.replace(dest)
                finished+=spec['bytes']
            update(downloaded_bytes=finished)
            if not alignment_only and voice.released():
                voice.update(status='downloading',phase='Preparing voice recognition',error=None)
                voice.download()
                if voice.state['status']=='failed':raise RuntimeError(voice.state['error'])
            if include_alignment and (alignment_only or not alignment.ready()):
                update(downloaded_bytes=finished+(voice.state['total_bytes'] if not alignment_only and voice.released() else 0))
                alignment.update(status='downloading',error=None)
                alignment.download(activate=alignment_only or enabled(root))
            update(status='ready',phase='Ready',downloaded_bytes=state['total_bytes'])
        except Exception as exc:
            if alignment.state['status']=='downloading':
                alignment.update(status='failed',error=str(exc),phase='Timing setup needs attention')
            update(status='failed',error=str(exc),phase='Setup needs attention')

    @app.before_request
    def setup_admission():
        # Setup owns a disposable neural check; serial admission avoids running
        # it beside a live/import worker. Edits and playback remain available.
        starts_inference=(request.endpoint in ('start_meeting','run_job','retry_segment') or
                          (request.endpoint=='refinement_control' and request.view_args.get('action')=='resume'))
        if starts_inference:
            with updates.lock,lock:
                if state['status']=='downloading':
                    abort(409,description='Finish local model setup before starting transcription.')

    @app.get('/api/setup')
    def status():
        with lock:result=state.copy()
        result['alignment']=alignment.status(busy=result['status']=='downloading' or voice.state['status']=='downloading',supported=voice.supported() and not updates.busy())
        result['alignment']['required']=True
        result['alignment']['can_change']=False
        result['models']=[{'id':s['directory'],'name':s['name'],'bytes':s['bytes'],'installed':installed(s),'optional':bool(s.get('optional'))} for s in SPECS]
        result['core_ready']=all(m['installed'] for m in result['models'] if not m['optional'])
        result['voice']=voice.status(core_busy=result['status']=='downloading',available=voice_available())
        result['voice']['enabled']=app.extensions['speakerdesk']['recognition_preference'].enabled()
        result['models'].append({'name':'Voice recognition','bytes':voice.state['total_bytes'],'installed':result['voice']['installed'],'optional':not result['voice']['released']})
        result['ready']=result['core_ready'] and (not result['voice']['released'] or (result['voice']['installed'] and result['voice']['available']))
        result['supported']=voice.supported()
        result['required_free_bytes']=sum(s['bytes'] for s in SPECS if s['directory']!=ALIGNMENT_SPEC['directory'] and not s.get('optional'))+(voice.state['total_bytes'] if result['voice']['released'] else 0)+result['alignment']['required_free_bytes']
        if result['status']=='downloading' and voice.state['status']=='downloading':
            result['phase']=voice.state['phase']
            result['downloaded_bytes']=state['downloaded_bytes']+voice.state['downloaded_bytes']
        if alignment.state['status']=='downloading':
            result['phase']=alignment.state['phase']
            result['downloaded_bytes']=state['downloaded_bytes']+alignment.state['downloaded_bytes']
        if result['ready'] and result['status'] not in ('downloading','failed'):result['status']='ready'
        return jsonify(result)

    @app.post('/api/setup')
    def start():
        body=request.get_json(silent=True)
        if body is None:body={}
        if not isinstance(body,dict) or type(body.get('alignment',True)) is not bool:
            return jsonify(error='Invalid optional alignment selection.'),400
        if body.get('alignment',True) is not True:
            return jsonify(error='Transcript timing is required for transcription. Download all required models.'),400
        include_alignment=True
        if not voice.supported():
            return jsonify(error='This version requires an Apple Silicon Mac with macOS 15 or later.'),409
        # Lock order: runtime admission, then setup state. Downloads never hold
        # setup state while acquiring runtime admission or activating a model.
        with updates.lock,lock:
            if updates.busy():return jsonify(error='Finish the active meeting or transcription before model setup.'),409
            if voice.state['status']=='downloading':return jsonify(error='Wait for voice setup to finish.'),409
            if state['status']=='downloading':return jsonify(state),202
            state.update(status='downloading',error=None,phase='Preparing download',
                total_bytes=sum(s.get('download_bytes',s['bytes']) for s in SPECS if not s.get('optional') or include_alignment)+(voice.state['total_bytes'] if voice.released() else 0))
            state['downloaded_bytes']=0
            updates.start_thread('model setup',download,include_alignment,name='model-setup')
        return jsonify(state),202

    @app.post('/api/setup/voice')
    def start_voice():
        if not voice.supported():
            return jsonify(error='Update Speakerdesk on an Apple Silicon Mac with macOS 15 or later to use voice recognition.'),409
        if not voice.released():
            return jsonify(error='Voice recognition needs a verified GPU/ANE runtime. Saved names and voices are kept.'),409
        with updates.lock,lock:
            if state['status']=='downloading':return jsonify(error='Wait for meeting model setup to finish.'),409
            if voice.state['status']=='downloading':return jsonify(voice.state),202
            voice.state.update(status='downloading',phase='Preparing voice download',error=None)
            updates.start_thread('voice setup',voice.download,name='voice-model-setup')
        return jsonify(voice.state),202

    @app.post('/api/setup/alignment')
    def start_alignment():
        if alignment.metadata is None:return jsonify(error='Transcript timing is unavailable in this build. Update Speakerdesk.'),409
        if not voice.supported():
            return jsonify(error='Timing requires an Apple Silicon Mac with macOS 15 or later.'),409
        with updates.lock,lock:
            if updates.busy():return jsonify(error='Finish the active meeting or transcription before timing setup.'),409
            if state['status']=='downloading' or voice.state['status']=='downloading':
                return jsonify(error='Wait for model setup to finish.'),409
            state.update(status='downloading',phase='Preparing timing download',error=None,
                         total_bytes=alignment.state['total_bytes'],downloaded_bytes=0)
            alignment.update(status='downloading',phase='Preparing timing download',error=None)
            updates.start_thread('timing setup',download,True,True,name='timing-model-setup')
        return jsonify(alignment.status(busy=True)),202

    @app.patch('/api/setup/alignment')
    def change_alignment():
        body=request.get_json(silent=True)
        if not isinstance(body,dict) or type(body.get('enabled')) is not bool:
            return jsonify(error='Invalid transcript timing request.'),400
        if body['enabled'] is not True:
            return jsonify(error='Transcript timing is required for transcription and cannot be disabled.'),409
        with updates.lock,lock:
            if updates.busy() or state['status']=='downloading':
                return jsonify(error='Finish the active meeting, transcription or setup before changing timing.'),409
            if body['enabled'] and not alignment.ready():
                return jsonify(error='Download and check the timing model before enabling it.'),409
            save_enabled(root,body['enabled'])
        return jsonify(alignment.status())
