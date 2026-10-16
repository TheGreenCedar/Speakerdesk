"""Small behavior checks for streaming coordinates, state bounds and result validation."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from compare_probe import compare
from export import ROOT, sha256
from qualify import compress, frame_windows
from verify_probe import WINDOWS_ML_SHA256


class Contract(unittest.TestCase):
    def test_stft_boundaries_use_original_sample_and_previous_sample(self):
        pcm = np.arange(1200, dtype=np.float32) / 1200
        complete = frame_windows(pcm, 0, len(pcm), 4, 8)
        retained = frame_windows(pcm[383:], 383, len(pcm), 4, 8)
        np.testing.assert_array_equal(complete, retained)
        self.assertEqual(complete[0, 0, 0], pcm[384] - np.float32(.97) * pcm[383])
        self.assertEqual(complete[0, -1, -1], 0)
        first = frame_windows(pcm, 0, len(pcm), 0, 1)
        self.assertTrue(np.all(first[0, 0, :256] == 0))
        self.assertEqual(first[0, 0, 256], pcm[0])

    def test_cache_compression_retains_silence_and_speaker_major_order(self):
        embeddings = np.broadcast_to(np.arange(300, dtype=np.float32)[None, :, None], (1, 300, 512)).copy()
        predictions = np.zeros((1, 300, 8), np.float32)
        silence = np.full(512, -1, np.float32)
        empty, empty_predictions = compress(embeddings, predictions, silence)
        self.assertEqual(empty.shape, (1, 264, 512))
        self.assertTrue(np.all(empty == -1))
        self.assertTrue(np.all(empty_predictions == 0))
        predictions[:, :, 0] = .9
        result, selected_predictions = compress(embeddings, predictions, silence)
        indices = result[0, :, 0]
        active = indices[indices >= 0].astype(int)
        self.assertEqual(len(active), 256)
        self.assertTrue(np.all(np.diff(active) > 0))
        self.assertTrue(set(range(264, 300)).issubset(active))
        self.assertTrue(np.all(selected_predictions[0, indices >= 0, 0] == np.float32(.9)))
        self.assertTrue(np.all(selected_predictions[0, indices < 0] == 0))

    def test_probe_comparison_rejects_changed_speaker_decision_and_corrupt_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "expected.f32"
            np.array([.4999], "<f4").tofile(reference)
            (root / "expected.json").write_text(json.dumps({"high": {
                "shape": [1], "file": reference.name, "sha256": sha256(reference)}}))
            rows = []
            for run in [0, 1]:
                path = root / f"result-{run}.f32"
                np.array([.5001], "<f4").tofile(path)
                rows.append({"run": run, "name": "high", "dtype": "f32", "shape": [1],
                             "file": path.name, "sha256": sha256(path)})
            (root / "model.onnx").write_bytes(b"graph identity control")
            (root / "case.json").write_text(json.dumps({"model": "model.onnx", "repeat": 2,
                "inputs": [{"name": "x", "dtype": "f32", "shape": [1], "path": reference.name}]}))
            (root / "profile.json").write_text(json.dumps([
                {"args": {"provider": "CPUExecutionProvider"}}]))
            record = {"provider_requested": "cpu", "runtime_sha256": WINDOWS_ML_SHA256,
                "loaded_runtime_libraries": [{"name": "onnxruntime.dll", "sha256": WINDOWS_ML_SHA256}],
                "case_sha256": sha256(root / "case.json"), "model_sha256": sha256(root / "model.onnx"),
                "inputs": [{"name": "x", "dtype": "f32", "shape": [1], "sha256": sha256(reference)}],
                "profile": "profile.json", "executed_provider_events": {"CPUExecutionProvider": 1},
                "outputs": rows}
            (root / "result.json").write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "changed_activity_decisions"):
                compare(root, root, .001, "cpu")
            # A numerically useful CPU control cannot stand in for a GPU run.
            with self.assertRaisesRegex(ValueError, "different provider"):
                compare(root, root, .001, "dml")
            record["provider_requested"] = "dml"
            pin = json.loads((ROOT / "crates/models/src/windows-runtime.json").read_text())
            record["loaded_runtime_libraries"].append({"name": "DirectML.dll", "sha256": pin["files"]["DirectML.dll"]["sha256"]})
            (root / "result.json").write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "without executed"):
                compare(root, root, .001, "dml")
            record["provider_requested"] = "cpu"
            record["loaded_runtime_libraries"][0]["sha256"] = "0" * 64
            (root / "result.json").write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "actual loaded runtime libraries differ"):
                compare(root, root, .001, "cpu")
            record["loaded_runtime_libraries"][0]["sha256"] = WINDOWS_ML_SHA256
            (root / "result.json").write_text(json.dumps(record))
            (root / rows[0]["file"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                compare(root, root, .001, "cpu")


if __name__ == "__main__":
    unittest.main()
