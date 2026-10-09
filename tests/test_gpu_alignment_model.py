"""Portable data/backend regressions; peers are not acoustic hardware proof."""
import contextlib
import hashlib
import json
import threading
from pathlib import Path
import struct
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
import alignment_model as artifact
import mlx_ctc_forward as forward
from coarse_alignment import CoarseAlignment,CALIBRATION_ID,TOKEN_SHA256
from alignment_cache import AlignmentCache
from word_alignment import MODEL_SHA256

class ArtifactTests(unittest.TestCase):
    def test_manifest_all_parameters_sources_and_calibration_are_frozen_without_task_paths(self):
        m=artifact.manifest();self.assertEqual(len(m['parameters']),422);self.assertEqual(len(m['initializers']),714)
        self.assertEqual({s for p in m['parameters'] for s in p['sources']},set(m['initializers']))
        self.assertEqual(len({p['target'] for p in m['parameters']}),422)
        self.assertEqual(m['qualification']['frozen_plan_sha256'],'5ab4e661e62f757bc1498121b2c392b46216f4ad3db55f42e16448b43bf64396')
        self.assertFalse(m['qualification']['threshold_or_offset_fitted']);self.assertNotIn('/Users/',json.dumps(m))
    def test_changed_packaged_manifest_is_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'manifest.json').write_text('{}')
            with patch.object(artifact,'DATA',p):
                with self.assertRaisesRegex(ValueError,'tensor manifest differs'):artifact.manifest()
    def test_runtime_version_or_forward_source_drift_rejected(self):
        m=artifact.manifest()
        with patch('importlib.metadata.version',return_value='wrong'):
            with self.assertRaisesRegex(RuntimeError,'packaged runtime'):artifact.require_runtime(m)
        versions=m['runtime']
        with patch('importlib.metadata.version',side_effect=lambda n:versions[n]),patch('importlib.metadata.distribution',return_value=SimpleNamespace(locate_file=lambda n:n)),patch.object(artifact,'digest_file',return_value='wrong'):
            with self.assertRaisesRegex(RuntimeError,'forward component differs'):artifact.require_runtime(m)
    def test_exact_byte_map_dequantization_and_transpose_matches_independent_values(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);source=p/'source';source.write_bytes(bytes([2,3,4,5]));out=p/'weights.npz'
            expected=np.array([[.5,1.5],[1.,2.]],dtype=np.float32)
            metadata=dict(initializers={'q':dict(dtype='u1',shape=[2,2],offset=0,bytes=4),'scale':dict(dtype='<f4',shape=[],scalar_hex=struct.pack('<f',.5).hex()),'zero':dict(dtype='u1',shape=[],scalar_hex='01')},parameters=[dict(target='layer.weight',sources=['q','scale','zero'],transform='UINT8_dequantize_then_I-O_to_O-I',shape=[2,2],converted_tensor_sha256=hashlib.sha256(expected.tobytes()).hexdigest())])
            artifact._convert(source,out,metadata)
            with np.load(out,allow_pickle=False) as result:np.testing.assert_array_equal(result['layer.weight'],expected)
            out2=p/'repeat.npz';artifact._convert(source,out2,metadata);self.assertEqual(out.read_bytes(),out2.read_bytes())
            metadata['parameters'][0]['converted_tensor_sha256']='bad'
            with self.assertRaisesRegex(ValueError,'tensor identity differs'):artifact._convert(source,p/'bad.npz',metadata)
    def test_prepare_reuses_valid_weights_and_rejects_changed_source_tokens_or_weights(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);source=folder/'model.int8.onnx';source.write_bytes(b'original');(folder/'tokens.txt').write_bytes(b'tokens');target=folder/artifact.DERIVED_DIRECTORY;target.mkdir();weights=target/'weights.npz';weights.write_bytes(b'verified')
            m=dict(source_model_bytes=8,source_model_sha256=hashlib.sha256(b'original').hexdigest(),converted_weights_bytes=8)
            with patch.object(artifact,'manifest',return_value=m),patch.object(artifact,'MODEL_SHA256',hashlib.sha256(b'verified').hexdigest()),patch.object(artifact,'TOKEN_SHA256',hashlib.sha256(b'tokens').hexdigest()),patch.object(artifact,'_convert',side_effect=AssertionError('unexpected conversion')):
                self.assertEqual(artifact.prepare_model(folder)[0],target)
                source.write_bytes(b'bad-data')
                with self.assertRaisesRegex(ValueError,'source model or tokens'):artifact.prepare_model(folder)
                source.write_bytes(b'original');(folder/'tokens.txt').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'source model or tokens'):artifact.prepare_model(folder)
                (folder/'tokens.txt').write_bytes(b'tokens');weights.write_bytes(b'bad-data')
                with self.assertRaisesRegex(ValueError,'weights differ'):artifact.prepare_model(folder)
    def test_failed_conversion_is_atomic_and_does_not_leave_ready_model(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);(folder/'model.int8.onnx').write_bytes(b'source');(folder/'tokens.txt').write_bytes(b'tokens')
            m=dict(source_model_bytes=6,source_model_sha256=hashlib.sha256(b'source').hexdigest(),converted_weights_bytes=8)
            def fail(source,dest,metadata):dest.write_bytes(b'partial');raise ValueError('conversion failed')
            with patch.object(artifact,'manifest',return_value=m),patch.object(artifact,'TOKEN_SHA256',hashlib.sha256(b'tokens').hexdigest()),patch.object(artifact,'_convert',side_effect=fail):
                with self.assertRaisesRegex(ValueError,'conversion failed'):artifact.prepare_model(folder)
            target=folder/artifact.DERIVED_DIRECTORY
            self.assertFalse((target/'weights.npz').exists());self.assertEqual(list(target.glob('.prepare-*')),[])
    def test_original_only_model_is_not_reported_GPU_prepared(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);(folder/'model.int8.onnx').write_bytes(b'legacy');self.assertFalse(artifact.prepared(folder))

class BackendTests(unittest.TestCase):
    def modules(self,mx):
        # Source CI has no Metal package. Stub the parent as well as its core
        # so these guards test the fake backend on every platform.
        parent=ModuleType('mlx');parent.__path__=[];parent.core=mx
        return {'mlx':parent,'mlx.core':mx}
    def mx(self,available=True,device='gpu',stream_device='gpu'):
        calls=[]
        m=SimpleNamespace(gpu='gpu',float32=np.float32,metal=SimpleNamespace(is_available=lambda:available),default_device=lambda:SimpleNamespace(type=device),default_stream=lambda d:SimpleNamespace(device=SimpleNamespace(type=stream_device)),stream=lambda d:contextlib.nullcontext(),array=lambda x,**k:np.array(x,**k),mean=np.mean,var=np.var,sqrt=np.sqrt,logsumexp=lambda x,**k:np.log(np.exp(x).sum(**k)),eval=lambda *a:calls.append('eval'),synchronize=lambda *a:calls.append('sync'))
        return m,calls
    def test_Metal_unavailable_rejects_before_any_model_or_data_preparation(self):
        mx,_=self.mx(available=False)
        with patch.dict(sys.modules,self.modules(mx)),patch.object(forward,'prepare_model',side_effect=AssertionError('unexpected prepare')):
            with self.assertRaisesRegex(RuntimeError,'No CPU fallback'):forward.load_gpu_model(directory='configured-models')
    def test_CPU_default_or_stream_rejected_before_neural_call(self):
        for device,stream in [('cpu','gpu'),('gpu','cpu')]:
            mx,_=self.mx(device=device,stream_device=stream)
            with patch.dict(sys.modules,self.modules(mx)):
                with self.assertRaisesRegex(RuntimeError,'refusing neural inference'):forward.forward_scores(lambda *a:self.fail('model called'),np.ones(8,dtype=np.float32))
    def test_forward_materializes_GPU_scores_before_execution_provenance(self):
        mx,calls=self.mx();receipt={}
        def model(audio):self.assertEqual(receipt,{});calls.append('model');return np.zeros((1,2,4),dtype=np.float32)
        with patch.dict(sys.modules,self.modules(mx)):scores=forward.forward_scores(model,np.arange(16,dtype=np.float32),execution=receipt)
        self.assertEqual(calls,['model','eval','sync']);self.assertEqual(receipt['backend'],'mlx_metal_gpu');self.assertTrue(receipt['evaluated_and_GPU_synchronized']);np.testing.assert_allclose(scores,np.full((2,4),-np.log(4)),rtol=1e-6)
    def test_provider_does_not_cache_or_publish_unverified_execution(self):
        p=CoarseAlignment.__new__(CoarseAlignment);p.model=object();p.actual_gpu_forwards=0;p.cache=AlignmentCache((MODEL_SHA256,TOKEN_SHA256,CALIBRATION_ID,'ctc_emission_cell_envelope',320,0,-20,'mlx-0.32.2-cached-float32-gpu','ctc-segmentation-1.7.4'));p.vocabulary={}
        with patch('mlx_ctc_forward.forward_scores',return_value=np.zeros((2,4),dtype=np.float32)),patch('coarse_alignment.align_ctc_scores',side_effect=AssertionError('segmentation called')):
            with self.assertRaisesRegex(RuntimeError,'execution evidence missing'):p.align(np.ones(8,dtype=np.float32),'hello',start_sample=0,language='en')
        self.assertEqual(p.cache.metrics()['cache_entries'],0)
    def test_package_contains_map_forward_modules_and_metal_distribution_metadata(self):
        s=(Path(__file__).resolve().parents[1]/'packaging/runtime.spec').read_text()
        for marker in ["speakerdesk/alignment-mlx","'alignment_model','mlx_ctc_forward'","mlx_ctc_components.models.mms.mms","mlx_ctc_components.models.wav2vec.wav2vec","'mlx','mlx-metal','mlx-audio'"]:self.assertIn(marker,s)

if __name__=='__main__':unittest.main()
