"""Package metadata must describe the actual version, OS floor and exact bytes."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import tempfile
import unittest
from unittest.mock import patch

class ArtifactManifestTests(unittest.TestCase):
    def test_manifest_uses_authoritative_os_floor_and_hashes_each_deliverable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'scripts').mkdir();(root/'desktop/src-tauri').mkdir(parents=True);(root/'release').mkdir()
            script=root/'scripts/artifact_manifest.py'
            shutil.copyfile(Path(__file__).resolve().parents[1]/'scripts/artifact_manifest.py',script)
            config={'version':'0.3.0','bundle':{'macOS':{'minimumSystemVersion':'15.0'}}}
            (root/'desktop/src-tauri/tauri.conf.json').write_text(json.dumps(config))
            blobs={'.dmg':b'Synthetic DMG bytes','.app.zip':b'Synthetic app ZIP bytes'}
            for suffix,data in blobs.items():(root/'release'/('Speakerdesk_0.3.0_AppleSilicon'+suffix)).write_bytes(data)
            with patch.dict(os.environ,{'GITHUB_SHA':'a'*40,'SPEAKERDESK_NOTARIZED':'0'}):runpy.run_path(str(script))
            manifest=json.loads((root/'release/artifact-manifest.json').read_text())
            self.assertEqual(manifest['minimum_os'],'15.0');self.assertEqual(manifest['source_commit'],'a'*40)
            self.assertFalse(manifest['notarized']);self.assertFalse(manifest['public_ready'])
            for entry,(suffix,data) in zip(manifest['files'],blobs.items()):
                self.assertEqual(entry['bytes'],len(data));self.assertEqual(entry['sha256'],hashlib.sha256(data).hexdigest())
                self.assertIn(entry['sha256'],(root/'release/SHA256SUMS').read_text())

if __name__=='__main__':unittest.main()
