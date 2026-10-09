"""Explicit frozen-runtime smoke: generated PCM, CPU DSP, no app or models."""
def run():
    import hashlib
    import importlib
    import importlib.abc
    import inspect
    import sys
    import threading
    from pathlib import Path

    refused = []
    class NoModels(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in {'mlx','mlx_audio','mlx_speech','onnxruntime','coremltools','torch'}:
                refused.append(fullname)
                raise RuntimeError('Neural imports forbidden in source smoke: '+fullname)
    guard = NoModels()
    sys.meta_path.insert(0,guard)
    try:
        names = ('capture_sources','source_runtime','source_voice','voice_source_audio','render_innovation','render_startup')
        root = Path(sys._MEIPASS)
        loaded = {}
        for name in names:
            module = importlib.import_module(name)
            assert Path(module.__file__).resolve().is_relative_to(root.resolve())
            loaded[name] = {'file': str(module.__file__), 'raw_source_sha256': hashlib.sha256((root/(name+'.py')).read_bytes()).hexdigest()}
        from alignment_provider import manifest
        timing = manifest()
        from asr_reuse_profile import current_profile, file_digest
        reuse_profile=current_profile()
        assert reuse_profile['namespace']=='source_bound_canonical_v1'
        assert reuse_profile['qualification']['current_namespace_execution_qualified'] is True
        for name,expected in reuse_profile['identity']['policy_source_sha256'].items():
            assert file_digest(root/name)==expected
        import os
        import tempfile
        from app import create_app
        with tempfile.TemporaryDirectory(prefix='speakerdesk-default-smoke-') as folder:
            saved={k:os.environ.get(k) for k in ('SPEAKERDESK_HOME','SPEAKERDESK_MODELS','SPEAKERDESK_VOICE_CONFIG')}
            os.environ.update(SPEAKERDESK_HOME=folder,SPEAKERDESK_MODELS=str(Path(folder)/'models'))
            os.environ.pop('SPEAKERDESK_VOICE_CONFIG',None)
            app=None
            try:
                app=create_app(Path(folder)/'recordings')
                ordinary=app.extensions['speakerdesk']['meetings']
                ordinary_gates={name:getattr(ordinary,name) for name in ('channel_transcription','source_innovation','source_startup_hold')}
                assert all(ordinary_gates.values())
                assert ordinary.capture is None and ordinary.worker is None and ordinary.jid is None
                client=app.test_client()
                assert client.get('/').status_code==200 and client.get('/api/meeting').status_code==200
            finally:
                if app is not None:
                    ordinary.close();app.extensions['speakerdesk']['close_voice']()
                    app.extensions['speakerdesk']['executor'].shutdown(wait=True,cancel_futures=True)
                for key,value in saved.items():
                    if value is None:os.environ.pop(key,None)
                    else:os.environ[key]=value
        from live_meeting import SourceMixer, MeetingManager
        from render_startup import StartupSourceSink
        from capture_sources import catalog,validate_catalog
        import numpy as np
        signature = inspect.signature(MeetingManager)
        gates = ('channel_transcription','source_innovation','source_startup_hold')
        defaults = {name: signature.parameters[name].default for name in gates}
        assert not any(defaults.values())
        manager = MeetingManager(lambda *a:None,lambda *a:None,lambda *a:None,lambda *a:None,
            threading.RLock(),lambda:False,updates=None,**{name:True for name in gates})
        assert all(getattr(manager,name) is True for name in gates)
        source_catalog = catalog('a'*32,innovation=True,startup=True)
        validate_catalog(source_catalog,'a'*32)
        assert source_catalog['derivation']['production_admissible'] is False
        far = np.random.default_rng(17).normal(0,.1,64000).astype(np.float32)
        mic = np.zeros_like(far)
        mic[1232:32000] = .18*far[:32000-1232]
        mic[32000:] = .18*far[32000-2400:-2400]
        def replay(near=False):
            raw = mic.copy()
            if near:
                raw[32000:34000] += np.random.default_rng(31).normal(0,.003,2000).astype(np.float32)
            out=[];mixes=[];native=[];decisions=[]
            def publish(mixed,tracks):
                out.append(tracks['microphone_clean'].copy());mixes.append(mixed.copy())
                native.append(tracks['microphone_native'].copy())
                decisions.extend(tracks['innovation_decisions'])
            mixer = SourceMixer(('microphone','system'),publish,innovation_factory=StartupSourceSink)
            try:
                for first in range(0,len(far),2000):
                    if first==32000:mixer.format_changed('microphone',first/16000)
                    end=first+2000
                    mixer.add('system',first/16000,np.repeat(far[first:end,None],2,axis=1).astype('<f4').tobytes(),channels=2)
                    mixer.add('microphone',first/16000,raw[first:end].astype('<f4').tobytes())
                    mixer.flush(end/16000)
                mixer.flush(len(far)/16000,final=True)
                selected=np.concatenate(out);mixed=np.concatenate(mixes);base=np.concatenate(native)
                assert len(selected)==len(raw)==mixer.cursor and mixer.late_samples==0
                np.testing.assert_array_equal(mixed,np.clip(far+selected,-1,1))
                if near:np.testing.assert_array_equal(selected[32000:34000],base[32000:34000])
                else:assert float(np.sqrt(np.mean(selected[32000:34400].astype(np.float64)**2)))<.6/32768
                return {'samples':len(selected),'late_samples':mixer.late_samples,'selected_sha256':hashlib.sha256(selected.tobytes()).hexdigest(),
                    'route_prefix_RMS_PCM16_units':float(np.sqrt(np.mean(selected[32000:34400].astype(np.float64)**2)))*32768,
                    'immediate_near_native_exact':bool(near),'mix_equation_exact':True,'decision_count':len(decisions)}
            finally:
                mixer.echo.close()
        cases={'microphone_reset_continuous_render':replay(),'immediate_quiet_near_after_reset':replay(True)}
        assert not refused
        return {'complete':True,'frozen':bool(getattr(sys,'frozen',False)),'loaded_modules':loaded,
            'timing_provider_identity_sha256':timing['provider_identity_sha256'],
            'timing_implementation_sha256':timing['provider_identity']['implementation_source_sha256'],
            'constructor_gates':defaults,'ordinary_factory_gates':ordinary_gates,
            'ordinary_factory_rendered':True,'reuse_namespace':reuse_profile['namespace'],
            'reuse_namespace_execution_qualified':True,'test_only_gates_enabled':True,'cases':cases,
            'neural_import_attempts':refused,'model_calls':0,'native_capture':False,'playback':False,
            'server_started':False,'native_UI_verified':False,'ordinary_recording_path_wired':True,
            'actual_recording_or_full_model_pipeline_executed':False}
    finally:
        sys.meta_path.remove(guard)
