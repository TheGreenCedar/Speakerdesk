"""Behavior checks for Whisper framing and original-reference acceptance."""
import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto

from compare_probe import compare
from export import language_tokens, sha256
from qualify import frame_audio, route
from precision_candidate import operator_model, precise_attention
from gelu_boundary import operator_model as gelu_operator, preserve_gelu_boundaries
from position_constant import load_positions
from verify_probe import WINDOWS_ML_SHA256


class Contract(unittest.TestCase):
    def test_positional_parameter_rejects_a_finite_same_shape_substitution(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "positions.npy"
            np.save(path, np.zeros((1500, 384), np.float16))
            with self.assertRaisesRegex(ValueError, "pinned MLX source table"):
                load_positions(path)

    def test_compiled_half_gelu_preserves_reciprocal_and_each_rounding_boundary(self):
        x = np.arange(-384, 385, dtype=np.float32) / 64
        half = lambda value: value.astype(np.float16).astype(np.float32)
        divided = half(x * np.float32(.70703125))
        erf = half(np.asarray([math.erf(float(value)) for value in divided]))
        expected = half(half(x * half(1 + erf)) / 2)
        original = gelu_operator(list(x.shape))
        before = original.SerializeToString()
        corrected, _ = preserve_gelu_boundaries(original, expected_count=1)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        outputs = []
        for model in [original, corrected]:
            session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
            outputs.append(session.run(["y"], {"x": x})[0])
        self.assertGreater(float(np.max(np.abs(outputs[0] - expected))), .001)
        np.testing.assert_array_equal(outputs[1], expected)
        self.assertEqual(original.SerializeToString(), before)

    def test_precise_attention_keeps_half_result_and_existing_float_softmax(self):
        original = operator_model([1, 6, 3, 1500])
        before = original.SerializeToString()
        candidate, _ = precise_attention(original, expected_count=1)
        self.assertEqual(original.SerializeToString(), before)
        inferred = onnx.shape_inference.infer_shapes(candidate)
        types = {v.name: v.type.tensor_type.elem_type for v in inferred.graph.value_info}
        softmaxes = [node for node in candidate.graph.node if node.op_type == "Softmax"]
        # Both attention variants accumulate as Float32; public results still round to Half.
        self.assertTrue(all(types[node.input[0]] == TensorProto.FLOAT for node in softmaxes))
        self.assertEqual(types["native_half"], TensorProto.FLOAT16)
        self.assertEqual(types["precise_half"], TensorProto.FLOAT16)
        original_float = next(n for n in original.graph.node if n.output[0] == "float_result")
        candidate_float = next(n for n in candidate.graph.node if n.output[0] == "float_result")
        self.assertEqual(original_float.SerializeToString(), candidate_float.SerializeToString())
        with self.assertRaisesRegex(ValueError, "unexpected Half attention softmax inventory"):
            precise_attention(original)

    def test_center_reflection_padding_trim_and_source_isolation(self):
        pcm = np.arange(1000, dtype=np.float32) / 1000
        frames = frame_audio(pcm)
        self.assertEqual(frames.shape, (1, 3000, 400))
        np.testing.assert_array_equal(frames[0, 0, :200], pcm[1:201][::-1])
        np.testing.assert_array_equal(frames[0, 0, 200:], pcm[:200])
        self.assertTrue(np.all(frames[0, 10:] == 0))
        before = frames.copy()
        self.assertTrue(np.all(frame_audio(np.zeros(512, np.float32)) == 0))
        np.testing.assert_array_equal(frames, before)
        long = np.zeros(480001, np.float32)
        long[-1] = 1
        self.assertTrue(np.all(frame_audio(long) == 0))
        for bad in [[], [np.nan], [np.inf], [[0.]]]:
            with self.assertRaisesRegex(ValueError, "finite nonempty"):
                frame_audio(bad)

    def test_routing_check_uses_both_original_thresholds_and_stable_ties(self):
        codes = ["en", "fr", "de"]
        for scores, confident in [([.90, .05, .05], True), ([.899, .051, .05], False),
                                  ([.95, .751, 0], False), ([.95, .75, 0], False)]:
            # The final decimal boundary is evaluated as f64 by the existing policy.
            self.assertEqual(route(scores, codes)["confident"], confident)
        self.assertEqual(route([.4, .4, .2], codes)["code"], "en")
        self.assertFalse(route(np.array([.90, .05, .05], np.float32), codes)["confident"])

    def test_language_ids_keep_metadata_order_and_reject_shifted_sot(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tokens.json"
            metadata = json.loads((Path(__file__).resolve().parents[3] /
                                  "tests/metadata/whisper-tiny/added_tokens.json").read_text())
            path.write_text(json.dumps(metadata))
            codes, ids, sot = language_tokens(path)
            self.assertEqual(len(codes), 99)
            self.assertEqual(ids[codes.index("en")], 50259)
            self.assertEqual(sot, 50258)
            metadata["<|startoftranscript|>"] += 1
            path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "incompatible"):
                language_tokens(path)

    def test_same_top_language_does_not_pass_a_changed_probability_vector(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = np.zeros(99, np.float32)
            expected[:2] = [.6, .4]
            actual = expected.copy()
            actual[:2] += [.001, -.001]
            expected.tofile(root / "expected.f32")
            codes = ["en", "fr"] + [f"lang{i}" for i in range(97)]
            (root / "expected.json").write_text(json.dumps({
                "name": "probabilities", "shape": [99], "file": "expected.f32",
                "sha256": sha256(root / "expected.f32"),
                "independent": {"codes": codes, "probabilities": expected.tolist()}}))
            (root / "model.onnx").write_bytes(b"controlled numeric-rejection fixture")
            np.zeros(1, "<f4").tofile(root / "input.f32")
            (root / "case.json").write_text(json.dumps({"model": "model.onnx", "repeat": 2, "inputs": [
                {"name": "mel", "dtype": "f32", "shape": [1], "path": "input.f32"}]}))
            (root / "profile.json").write_text(json.dumps([{"args": {"provider": "CPUExecutionProvider"}}]))
            rows = []
            for run in [0, 1]:
                path = root / f"run-{run}.f32"
                actual.tofile(path)
                rows.append({"run": run, "name": "probabilities", "dtype": "f32", "shape": [99],
                             "file": path.name, "sha256": sha256(path)})
            (root / "result.json").write_text(json.dumps({"outputs": rows, "provider_requested": "cpu",
                "runtime_sha256": WINDOWS_ML_SHA256, "execution_verified": True,
                "case_sha256": sha256(root / "case.json"), "model_sha256": sha256(root / "model.onnx"),
                "inputs": [{"name": "mel", "dtype": "f32", "shape": [1], "sha256": sha256(root / "input.f32")}],
                "profile": "C:\\profile.json", "executed_provider_events": {"CPUExecutionProvider": 1},
                "loaded_runtime_libraries": [{"name": "onnxruntime.dll", "sha256": WINDOWS_ML_SHA256}]}))
            with contextlib.redirect_stdout(io.StringIO()):
                result = compare(root, root, "cpu")
            self.assertFalse(result["qualified_under_original_1e_4_bound"])
            self.assertGreater(result["reports"][0]["independent"]["max_probability_error"], 1e-4)
            with self.assertRaisesRegex(ValueError, "different provider"):
                compare(root, root, "dml")


if __name__ == "__main__":
    unittest.main()
