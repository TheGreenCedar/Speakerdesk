"""Single-user loopback app; SQLite job persistence and one serial worker."""
import copy
from contextlib import contextmanager
import json
import os
import secrets
import shutil
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException
from werkzeug.utils import secure_filename
from audio import normalize
from pipeline import infer, model_config, preflight, MODELS
from transcript import LANGUAGES, validate, export
from model_setup import register_setup
from live_meeting import register_meetings, LIVE

ROOT=Path(__file__).resolve().parent
ACTIVE=('preparing','queued','processing')+LIVE


def create_app(data_dir=None):
    app=Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=256*1024*1024, TRUSTED_HOSTS=['localhost','127.0.0.1','[::1]'])
    data=Path(data_dir or os.getenv('SPEAKERDESK_DATA',ROOT/'data')).resolve()
    data.mkdir(parents=True,exist_ok=True)
    os.chmod(data,0o700)
    database=data/'jobs.sqlite'
    token=secrets.token_urlsafe(32)
    executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='speakerdesk')
    lock=threading.RLock()

    @contextmanager
    def db():
        conn=sqlite3.connect(database,timeout=10)
        conn.row_factory=sqlite3.Row
        try:
            with conn:yield conn
        finally:conn.close()

    with db() as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
        for row in conn.execute('SELECT * FROM jobs').fetchall():
            job=json.loads(row['payload'])
            if job['status'] in ACTIVE:
                job.update(status='failed',message=('The meeting was interrupted. Captured audio and transcript were preserved.' if job.get('kind')=='meeting' else 'The app stopped during processing. Retry this job.'),updated=time.time())
                conn.execute('UPDATE jobs SET payload=? WHERE id=?',(json.dumps(job),job['id']))

    def get(jid):
        if len(jid)!=32 or any(c not in '0123456789abcdef' for c in jid):abort(404)
        with db() as conn:row=conn.execute('SELECT payload FROM jobs WHERE id=?',(jid,)).fetchone()
        if not row:abort(404)
        return json.loads(row['payload'])

    def put(job):
        job['updated']=time.time()
        with db() as conn:
            conn.execute('INSERT OR REPLACE INTO jobs VALUES (?,?)',(job['id'],json.dumps(job,ensure_ascii=False)))

    def patch(jid,**changes):
        with lock:
            job=get(jid);job.update(**changes);put(job)

    def folder(jid):return data/jid

    def prepare(jid):
        try:
            job=get(jid)
            duration=normalize(folder(jid)/job['source_file'],folder(jid)/'audio.wav')
            patch(jid,status='uploaded',duration=duration,message='Ready for local inference or manual editing.')
        except Exception as exc:
            patch(jid,status='failed',message=str(exc) if isinstance(exc,ValueError) else 'Audio preparation failed. Check file format and free disk space.')

    def process(jid):
        try:
            job=get(jid)
            if not job.get('duration'):
                prepare(jid);job=get(jid)
                if job['status']=='failed':return
            patch(jid,status='processing',message='Starting local inference…')
            document=infer(folder(jid)/'audio.wav',job['duration'],job['language'],folder(jid),
                           lambda message:patch(jid,message=message))
            document=validate(document,job['duration'])
            patch(jid,status='ready',document=document,revision=job['revision']+1,message='Transcript ready. Review names, text and timing.')
        except Exception as exc:
            patch(jid,status='failed',message=str(exc) if isinstance(exc,(ValueError,RuntimeError)) else 'Processing failed. Check the inference environment and storage.')

    @app.before_request
    def local_access():
        if request.method in ('POST','PUT','PATCH','DELETE'):
            if not secrets.compare_digest(request.headers.get('X-Speakerdesk-Token',''),token):
                abort(403,description='Reload the app before making changes.')

    @app.after_request
    def security_headers(response):
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Referrer-Policy']='no-referrer'
        response.headers['Content-Security-Policy']="default-src 'self'; style-src 'self'; script-src 'self'; media-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'"
        response.headers['Cache-Control']='no-store'
        return response

    @app.errorhandler(Exception)
    def error(exc):
        if isinstance(exc,HTTPException):return jsonify(error=exc.description),exc.code
        if isinstance(exc,(ValueError,KeyError,TypeError)):
            return jsonify(error=str(exc) if isinstance(exc,ValueError) else 'Invalid request.'),400
        app.logger.exception('Request failed')
        return jsonify(error='Local operation failed. Check disk space and the app terminal.'),500

    @app.get('/')
    def index():return render_template('index.html',token=token)

    @app.get('/api/notices')
    def notices():
        path=ROOT/'THIRD_PARTY_NOTICES.txt'
        if not path.exists():path=ROOT.parent/'packaging/THIRD_PARTY_NOTICES.txt'
        return send_file(path,as_attachment=True,download_name='Speakerdesk-third-party-notices.txt',mimetype='text/plain')

    @app.get('/api/config')
    def config():
        cfg=model_config();issues=preflight(cfg)
        return jsonify(languages=LANGUAGES,readiness={'configured':not issues,'issues':issues,
            'model':MODELS.get(cfg['diar_kind'],('unknown',0))[0],
            'speaker_limit':MODELS.get(cfg['diar_kind'],('',0))[1],
            'device':cfg['device'],'note':'Model execution is checked when inference starts. Configuration is not proof of working inference.'},
            decoder='FFmpeg' if shutil.which('ffmpeg') else ('macOS AudioToolbox' if shutil.which('afconvert') else 'PCM WAV only'))

    @app.get('/api/jobs')
    def jobs():
        with db() as conn:items=[json.loads(row['payload']) for row in conn.execute('SELECT payload FROM jobs')]
        return jsonify(sorted([{k:v for k,v in j.items() if k!='document'} for j in items],key=lambda j:j['created'],reverse=True))

    @app.post('/api/jobs')
    def upload():
        files=request.files.getlist('files');language=request.form.get('language','en')
        if not files or len(files)>16:raise ValueError('Choose 1–16 audio files per batch.')
        if language not in LANGUAGES:raise ValueError('Select a supported recording language.')
        for item in files:
            if Path(item.filename or '').suffix.lower() not in ('.wav','.mp3','.m4a','.flac','.ogg','.aiff','.aif','.mp4','.aac','.webm'):
                raise ValueError('Supported formats: WAV, MP3, M4A, FLAC, OGG, AIFF, MP4, AAC and WebM.')
        with lock:
            with db() as conn:active=sum(json.loads(row['payload'])['status'] in ACTIVE for row in conn.execute('SELECT payload FROM jobs'))
            if active+len(files)>16:abort(429,description='The local queue is full. Wait for current jobs to finish.')
            created=[]
            for item in files:
                jid=uuid.uuid4().hex;dest=folder(jid);dest.mkdir(mode=0o700)
                source='source'+Path(item.filename).suffix.lower();item.save(dest/source)
                job={'id':jid,'name':secure_filename(item.filename) or 'Recording','language':language,
                     'source_file':source,'status':'preparing','message':'Preparing local audio…',
                     'duration':None,'created':time.time(),'revision':0,'document':None}
                put(job);created.append(job);executor.submit(prepare,jid)
        return jsonify(created),201

    @app.get('/api/jobs/<jid>')
    def job_detail(jid):return jsonify(get(jid))

    @app.get('/api/jobs/<jid>/audio')
    def audio(jid):
        get(jid)
        path=folder(jid)/'audio.wav'
        if not path.is_file():abort(409,description='Audio is not ready yet.')
        return send_file(path,mimetype='audio/wav',conditional=True)

    @app.post('/api/jobs/<jid>/run')
    def run_job(jid):
        with lock:
            job=get(jid)
            if job['status'] in ACTIVE:abort(409,description='This job is already running.')
            if job.get('document') and job.get('document',{}).get('segments'):abort(409,description='Create another upload to rerun inference without overwriting edits.')
            issues=preflight(model_config())
            if issues:abort(409,description=' '.join(issues))
            patch(jid,status='queued',message='Waiting for local inference…');executor.submit(process,jid)
        return jsonify(get(jid)),202

    @app.put('/api/jobs/<jid>/transcript')
    def save(jid):
        body=request.get_json()
        with lock:
            job=get(jid)
            if job['status'] in ACTIVE or not job.get('duration'):abort(409,description='Wait for audio preparation and inference to finish.')
            if body.get('revision')!=job['revision']:abort(409,description='This transcript changed in another tab. Reload before saving.')
            incoming=validate(body.get('document'),job['duration'])
            # Provenance is server-owned. Editing cannot pretend to be fresh inference.
            if body.get('imported'):
                document={'schema_version':1,'speakers':incoming['speakers'],'segments':incoming['segments'],
                          'provenance':{'kind':'imported','timing':'User imported segment times; source unverified'},
                          'warnings':['Imported transcript. Text, speaker labels and timing have not been verified by this app.']}
            elif job.get('document'):
                document=copy.deepcopy(job['document'])
                document.update(speakers=incoming['speakers'],segments=incoming['segments'])
            else:
                document={'schema_version':1,'speakers':incoming['speakers'],'segments':incoming['segments'],
                          'provenance':{'kind':'manual','timing':'User supplied recording times'},
                          'warnings':['This transcript contains manually entered or imported text and timing.']}
            document['edited_at']=time.time()
            patch(jid,document=document,revision=job['revision']+1,status='ready',message='Saved locally.')
        return jsonify(get(jid))

    @app.get('/api/jobs/<jid>/export/<kind>')
    def download(jid,kind):
        if kind not in ('txt','srt','vtt','json'):abort(404)
        job=get(jid)
        if not job.get('document'):abort(409,description='Create a transcript before exporting.')
        content,mimetype=export(job['document'],kind)
        response=app.response_class(content,content_type=mimetype)
        filename=f'{Path(job["name"]).stem}.{kind}'
        response.headers['Content-Disposition']=f'attachment; filename="{secure_filename(filename) or "transcript."+kind}"'
        return response

    @app.delete('/api/jobs/<jid>')
    def delete(jid):
        with lock:
            job=get(jid)
            if job['status'] in ACTIVE:abort(409,description='Wait for this job to finish before deleting it.')
            shutil.rmtree(folder(jid))
            with db() as conn:conn.execute('DELETE FROM jobs WHERE id=?',(jid,))
        return jsonify(deleted=True)

    app.extensions['speakerdesk']={'executor':executor,'data':data}
    register_setup(app)
    register_meetings(app,get,put,patch,folder)
    return app


if __name__=='__main__':
    app=create_app()
    app.run(host='127.0.0.1',port=int(os.getenv('PORT','8790')),debug=False,threaded=True,use_reloader=False)
