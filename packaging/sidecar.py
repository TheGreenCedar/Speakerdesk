"""Self-contained runtime: server or one disposable MLX worker process."""
import os
import sys
import threading
from pathlib import Path

APP_ROOT=Path(sys._MEIPASS) if getattr(sys,'frozen',False) else Path(__file__).resolve().parents[1]/'speakerdesk'
sys.path.insert(0,str(APP_ROOT))

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--model-capability':
        import importlib.metadata
        import json
        import platform
        import mlx.core as mx
        versions={name:importlib.metadata.version(name) for name in ('mlx','mlx-audio','mlx-speech')}
        print(json.dumps({'scope':'capability_only','frozen':bool(getattr(sys,'frozen',False)),
                          'architecture':platform.machine(),'metal_available':bool(mx.metal.is_available()),
                          'versions':versions,'models_executed':False,'native_capture':False}),flush=True)
        sys.exit(0)
    if len(sys.argv)>1 and sys.argv[1]=='--live-worker':
        import json
        from live_worker import run, emit
        try:run(json.loads(sys.argv[2]))
        except Exception as exc:
            import traceback
            traceback.print_exc();emit({'type':'error','error':str(exc)});sys.exit(1)
        sys.exit(0)
    if len(sys.argv)>1 and sys.argv[1]=='--worker':
        import json
        from inference_worker import run
        task,request_path,output_path=sys.argv[2:]
        try:result=run(task,json.loads(Path(request_path).read_text()))
        except Exception as exc:
            import traceback
            traceback.print_exc();result={'error':f'{type(exc).__name__}: {exc}'}
        Path(output_path).write_text(json.dumps(result,ensure_ascii=False))
        sys.exit(1 if 'error' in result else 0)
    from app import create_app
    from werkzeug.serving import make_server
    storage=Path(os.getenv('SPEAKERDESK_HOME',Path.home()/'Library/Application Support/Speakerdesk'))
    storage.mkdir(parents=True,exist_ok=True);storage.chmod(0o700)
    os.environ['SPEAKERDESK_MODELS']=str(storage/'models')
    os.environ['DIAR_PYTHON']=sys.executable
    os.environ['ASR_PYTHON']=sys.executable
    os.environ['HF_HOME']=str(storage/'cache')
    app=create_app(storage/'recordings')
    server=make_server('127.0.0.1',0,app,threaded=True)
    def listen_for_shutdown():
        # A closed parent pipe means the desktop app exited or crashed.
        try:
            for line in sys.stdin:
                if line.strip()=='shutdown':break
        finally:
            from pipeline import shutdown_workers
            app.extensions['speakerdesk']['meetings'].close()
            shutdown_workers()
            app.extensions['speakerdesk']['executor'].shutdown(wait=False,cancel_futures=True)
            server.shutdown()
    threading.Thread(target=listen_for_shutdown,daemon=True).start()
    print(f'SPEAKERDESK_URL=http://127.0.0.1:{server.server_port}',flush=True)
    server.serve_forever()
