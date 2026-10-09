"""CPU source identity controls; no new neural accuracy or calibration claim."""
import builtins
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'speakerdesk'))
import alignment_provider as provider


class SourceIdentityTests(unittest.TestCase):
    def test_current_channel_sources_bind_and_old_behavior_receipt_declines(self):
        attempts=[];original=builtins.__import__
        def deny(name,*a,**kw):
            if name.split('.')[0] in ('mlx','mlx_audio','mlx_speech','torch','onnxruntime','coremltools'):
                attempts.append(name);raise RuntimeError('Neural import refused')
            return original(name,*a,**kw)
        with patch('builtins.__import__',side_effect=deny):metadata=provider.manifest()
        self.assertEqual(attempts,[])
        root=Path(provider.__file__).parent
        identity=metadata['provider_identity']
        for name,pin in identity['implementation_source_sha256'].items():
            self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),pin)
        receipt=dict(model_sha256=provider.MODEL_SHA256,manifest_sha256=provider.MANIFEST_SHA256,
            runtime=metadata['runtime'],provider_id=identity['provider_id'],
            provider_identity_sha256='1f4c78c71ad068e6d1f263aa79ca8071712209667c78487692f54323a7eed3c1',
            execution=dict(backend='mlx_metal_gpu',evaluated_and_GPU_synchronized=True))
        with self.assertRaisesRegex(ValueError,'new GPU compatibility check'):provider.validate_receipt(receipt,metadata)

    def test_each_source_drift_still_declines_before_tensor_manifest(self):
        original=Path(provider.__file__).parent
        identity=json.loads((original/'alignment-mlx/provider-identity.json').read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'alignment-mlx').mkdir()
            shutil.copyfile(original/'alignment-mlx/provider-identity.json',root/'alignment-mlx/provider-identity.json')
            for name in identity['implementation_source_sha256']:shutil.copyfile(original/name,root/name)
            with patch.object(sys,'_MEIPASS',str(root),create=True):provider.manifest()
            for name in identity['implementation_source_sha256']:
                with self.subTest(source=name):
                    p=root/name;p.write_bytes(p.read_bytes()+b'\n# CPU drift sentinel\n')
                    with patch.object(sys,'_MEIPASS',str(root),create=True),patch.object(provider,'_tensor_manifest',
                            side_effect=AssertionError('Source drift must decline first')):
                        with self.assertRaisesRegex(ValueError,'implementation differs: '+name):provider.manifest()
                    shutil.copyfile(original/name,p)


if __name__=='__main__':unittest.main()
