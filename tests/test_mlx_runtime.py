"""Pinned runtime preparation stays outside Cargo's pruned build outputs."""
import hashlib
import json
import os
import re
import subprocess
import tempfile
import tomllib
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path('.cache/native/mlx-0.32.2')


class MlxRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.wheel = self.root / 'fixture.whl'
        with zipfile.ZipFile(self.wheel, 'w') as archive:
            for name in ['lib/libmlx.dylib', 'lib/libjaccl.dylib', 'lib/mlx.metallib',
                         'include/mlx/mlx.h', 'share/cmake/MLX/MLXConfig.cmake']:
                archive.writestr('mlx/' + name, 'Independent fixture ' + name)
            archive.writestr('mlx_metal-0.32.2.dist-info/licenses/LICENSE', 'Original MLX license')
        self.digest = hashlib.sha256(self.wheel.read_bytes()).hexdigest()
        self.script = self.root / 'scripts/fetch_mlx_runtime.sh'
        self.script.parent.mkdir()
        # Only the archive pin changes in this isolated copy; real shasum and unzip run.
        self.script.write_text(re.sub(r'^sha=[0-9a-f]+$', 'sha=' + self.digest,
                                     (ROOT / 'scripts/fetch_mlx_runtime.sh').read_text(), flags=re.M))
        tools = self.root / 'tools'
        tools.mkdir()
        curl = tools / 'curl'
        curl.write_text('#!/usr/bin/env python3\n'
                        'import os, shutil, sys\n'
                        'from pathlib import Path\n'
                        'with Path(os.environ["MLX_TEST_REQUESTS"]).open("a") as output:\n'
                        '    output.write("download\\n")\n'
                        'shutil.copyfile(os.environ["MLX_TEST_WHEEL"], sys.argv[sys.argv.index("-o") + 1])\n')
        curl.chmod(0o755)
        self.requests = self.root / 'requests'
        self.environment = dict(os.environ, PATH=str(tools) + os.pathsep + os.environ['PATH'],
                                MLX_TEST_WHEEL=str(self.wheel), MLX_TEST_REQUESTS=str(self.requests))

    def fetch(self):
        return subprocess.run(['bash', str(self.script)], cwd=self.root,
                              env=self.environment, capture_output=True, text=True)

    def test_pruned_cargo_runtime_does_not_block_preparation_or_valid_reuse(self):
        # rust-cache 2.9.2 leaves non-Cargo directories after deleting their files.
        old = self.root / 'target/native/mlx-0.32.2'
        (old / 'lib').mkdir(parents=True)
        (old / 'CACHEDIR.TAG').write_text('Retained Cargo cache marker')
        prepared = self.fetch()
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        destination = self.root / RUNTIME
        self.assertEqual(Path(prepared.stdout.strip()), destination)
        self.assertEqual((destination / '.archive-sha256').read_text().strip(), self.digest)
        self.assertEqual((destination / 'LICENSE').read_text(), 'Original MLX license')
        cargo = tomllib.loads((ROOT / '.cargo/config.toml').read_text())['env']['MLX_PREBUILT_ROOT']
        self.assertTrue(cargo['relative'])
        self.assertEqual(self.root / cargo['value'], destination)
        bundle = json.loads((ROOT / 'desktop/src-tauri/tauri.conf.json').read_text())['bundle']
        tauri = self.root / 'desktop/src-tauri'
        for source in [bundle['macOS']['frameworks'][0],
                       next(source for source, target in bundle['resources'].items() if target == 'mlx/mlx.metallib')]:
            path = (tauri / source).resolve()
            self.assertTrue(path.is_relative_to(destination))
            self.assertTrue(path.is_file())
        self.wheel.unlink()  # Valid reuse must not try to download the archive again.
        reused = self.fetch()
        self.assertEqual(reused.returncode, 0, reused.stderr)
        self.assertEqual(reused.stdout, prepared.stdout)
        self.assertEqual(self.requests.read_text(), 'download\n')
        self.assertEqual((old / 'CACHEDIR.TAG').read_text(), 'Retained Cargo cache marker')
        self.assertEqual(list((old / 'lib').iterdir()), [])

    def test_incomplete_prepared_runtime_is_rejected_without_replacing_it(self):
        prepared = self.fetch()
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        destination = Path(prepared.stdout.strip())
        (destination / 'lib/mlx.metallib').unlink()
        rejected = self.fetch()
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn('Incomplete or different runtime', rejected.stderr)
        self.assertEqual((destination / '.archive-sha256').read_text().strip(), self.digest)
        self.assertTrue((destination / 'lib/libmlx.dylib').is_file())
        self.assertFalse((destination / 'lib/mlx.metallib').exists())
        self.assertEqual(self.requests.read_text(), 'download\n')
