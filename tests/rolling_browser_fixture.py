"""Synthetic CPU UI fixture using the real API/SQLite/host reconciler and assets."""
import copy
from pathlib import Path
import signal
import sys
import tempfile
from unittest.mock import patch
from flask import jsonify,request
from werkzeug.serving import make_server
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
from app import create_app
from language_detection import LANGUAGE_CHOICES
from rolling_refinement import RATE,segment_version


def main():
    output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='speakerdesk-rolling-browser-') as temporary:
        app=create_app(temporary);manager=app.extensions['speakerdesk']['meetings'];jid='d'*32
        rows=[{'id':f'row-{n}','start':n*3,'end':(n+1)*3,'text':f'Synthetic provisional passage {n+1}. More words will arrive with context.',
               'speaker':'speaker_0','language':'en','language_generation':0,'language_epoch':0,'language_mode':'en',
               'machine_revision':1,'refinement_state':'provisional','finalized':True,'fast_origin_sample':n*3*RATE,
               'audio_anchor':{'start_sample':n*3*RATE,'end_sample':(n+1)*3*RATE}}
              for n in range(24)]
        rows.extend({'id':f'blank-{n}','start':72+n*3,'end':75+n*3,'text':'','speaker':'speaker_0',
                     'machine_revision':1,'refinement_state':'provisional','language_generation':0,'language_epoch':0,'language_mode':'en','language':'en'} for n in range(24))
        job={'id':jid,'kind':'meeting','name':'Rolling context · synthetic CPU fixture','status':'recording','language':'en',
             'message':'Synthetic outputs for UI validation; no capture or trained model execution.','duration':144,'created':1,'revision':1,
             'document':{'schema_version':1,'speakers':{'speaker_0':'Confirmed Albert'},'segments':rows,
                         'provenance':{'kind':'local_inference','timing':'Synthetic audio regions; no word timestamps.'}}}
        manager.refinement.initialize(job);job['rolling_sources']={s['id']:copy.deepcopy(s) for s in rows};manager.put(job)
        initial=copy.deepcopy(job)
        manager.jid=jid;manager.duration=manager.processed=144;manager.two_pass=True
        class IdleCPUWorker:
            def poll(self):return None
        manager.worker=IdleCPUWorker()
        @app.post('/fixture/reset')
        def reset():
            manager.put(copy.deepcopy(initial));manager.active_language='en';manager.active_epoch=0
            return jsonify(ok=True)
        @app.post('/fixture/saved-owned')
        def saved_owned():
            job=manager.get(jid);job.update(status='ready',refinement_status='refining');manager.put(job)
            manager.jid=jid if request.get_json().get('owned') else None
            return jsonify(ok=True)
        app.view_functions['config']=lambda:jsonify(languages=LANGUAGE_CHOICES,default_language='auto',readiness={'configured':True,'automatic_language':True},decoder='CPU fixture')
        app.view_functions['status']=lambda:jsonify(models=[],status='ready',ready=True,core_ready=True,supported=True,total_bytes=1,downloaded_bytes=1,error=None,
            voice={'status':'ready','available':False,'enabled':False,'message':'CPU UI fixture'})
        @app.post('/fixture/advance')
        def advance():
            body=request.get_json();start=body.get('start',0);end=start+6;job=manager.get(jid)
            win={'id':f'fixture-{start}','kind':'fast_tail','start_sample':start*RATE,'end_sample':end*RATE}
            candidates=[{'start':start,'end':end,'text':f'Synthetic larger-context words from {start} seconds. Yes yes yes.',
                         'speaker':'speaker_0','language':'en','language_mode':'en','language_epoch':0,'language_generation':0,
                         'fast_origin_sample':start*RATE,'refinement_state':'provisional','finalized':True}]
            manager.refinement.provisional(jid,{'window':win,'expected':{s['id']:segment_version(s) for s in job['document']['segments'] if s['start']<end and s['end']>start},
                'candidates':candidates,'language_epoch':0,'speakers':{'speaker_0':'Guessed name'}})
            return jsonify(manager.get(jid))
        server=make_server('127.0.0.1',0,app,threaded=True);(output/'port.txt').write_text(str(server.server_port))
        signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
        print(f'CPU rolling browser fixture: http://127.0.0.1:{server.server_port}',flush=True)
        with patch('live_meeting.preflight',return_value=[]):
            try:server.serve_forever()
            except KeyboardInterrupt:pass
            finally:
                server.server_close();manager.worker=None;manager.jid=None;manager.close()
                app.extensions['speakerdesk']['executor'].shutdown(wait=True,cancel_futures=True)


if __name__=='__main__':main()
