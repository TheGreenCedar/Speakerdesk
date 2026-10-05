"""Pinned public model downloads with progress, resumption and checksum checks."""
import hashlib
import os
import platform
import shutil
import ssl
import threading
import urllib.request
import certifi
from pathlib import Path
from flask import jsonify

SPECS=[
    {'name':'Speaker recognition','repo':'mlx-community/Nemotron-3-Diarization','revision':'59ed2dbfc1346dcea9d423c71306a3a2499c568f',
     'directory':'nemotron','bytes':198632220,'sha256':'21e8427d1795c9c46c5800f56b16061734ffd0dcadd71d9bcf0b4d6ef7261da5',
     'files':['config.json','model.safetensors','README.md']},
    {'name':'Transcription','repo':'spokedotso/cohere-transcribe-03-2026-mlx-4bit','revision':'064e51eab6db47066cbeaa85e2894b5691bd8d12',
     'directory':'cohere-speech','bytes':1505759837,'sha256':'ae947c13ba1cb8ce24c8bf72ded2520dcda1894ebf5a54fbb803c03bc28ef7fb',
     'files':['config.json','model.safetensors','generation_config.json','preprocessor_config.json','processor_config.json','special_tokens_map.json','tokenizer.json','tokenizer.model','tokenizer_config.json','README.md']},
]


def register_setup(app):
    root=Path(os.getenv('SPEAKERDESK_MODELS',Path(__file__).resolve().parents[1]/'models'))
    lock=threading.Lock()
    state={'status':'idle','phase':'','downloaded_bytes':0,'total_bytes':sum(s['bytes'] for s in SPECS),'error':None}

    def installed(spec):
        folder=root/spec['directory']
        return all((folder/name).is_file() for name in spec['files']) and (folder/'model.safetensors').stat().st_size==spec['bytes']

    def update(**changes):
        with lock:state.update(changes)

    def download():
        try:
            root.mkdir(parents=True,exist_ok=True)
            missing=sum(s['bytes'] for s in SPECS if not installed(s))
            if shutil.disk_usage(root).free < missing+512*1024**2:
                raise RuntimeError('There is not enough free disk space. Free at least 2.3 GB and retry.')
            finished=0
            for spec in SPECS:
                if installed(spec):finished+=spec['bytes'];continue
                folder=root/spec['directory'];folder.mkdir(exist_ok=True)
                update(phase=spec['name'])
                for name in spec['files']:
                    dest=folder/name
                    if dest.exists() and name!='model.safetensors':continue
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
                                if name=='model.safetensors':update(downloaded_bytes=finished+offset)
                    if name=='model.safetensors':
                        update(phase=f'Verifying {spec["name"].lower()}')
                        digest=hashlib.sha256()
                        with partial.open('rb') as content:
                            while block:=content.read(8*1024**2):digest.update(block)
                        if partial.stat().st_size!=spec['bytes'] or digest.hexdigest()!=spec['sha256']:
                            partial.unlink(missing_ok=True)
                            raise RuntimeError('Model verification failed. Retry the download.')
                    partial.replace(dest)
                finished+=spec['bytes']
            update(status='ready',phase='Ready',downloaded_bytes=state['total_bytes'])
        except Exception as exc:
            update(status='failed',error=str(exc),phase='Setup needs attention')

    @app.get('/api/setup')
    def status():
        with lock:result=state.copy()
        result['models']=[{'name':s['name'],'bytes':s['bytes'],'installed':installed(s)} for s in SPECS]
        result['ready']=all(m['installed'] for m in result['models'])
        result['supported']=platform.system()=='Darwin' and platform.machine()=='arm64'
        if result['ready']:result['status']='ready'
        return jsonify(result)

    @app.post('/api/setup')
    def start():
        if platform.system()!='Darwin' or platform.machine()!='arm64':
            return jsonify(error='This version requires an Apple Silicon Mac.'),409
        with lock:
            if state['status']=='downloading':return jsonify(state),202
            state.update(status='downloading',error=None,phase='Preparing download')
        threading.Thread(target=download,daemon=True,name='model-setup').start()
        return jsonify(state),202
