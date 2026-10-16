"""No inference: preserve graph bytes and catch diagnostic admission/interpretation mistakes."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper

from compare_taps import embedding_report, read_outputs
from prepare_taps import STAGE0_NORM_TAPS, STAGE_TAPS, TAPS, add_taps, prepare, sha256


class Taps(unittest.TestCase):
    def test_stage_preset_exposes_only_stage_returns_and_final_mixture(self):
        nodes = [helper.make_node("Identity", ["audio"], ["embedding"], name="embedding")]
        nodes.extend(helper.make_node("Cast", ["audio"], [f"stage_{index}"],
                                      name=node_name, to=TensorProto.FLOAT16)
                     for index, node_name in enumerate(STAGE_TAPS.values()))
        model = helper.make_model(helper.make_graph(nodes, "stage_fixture", [
            helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 12])], [
            helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, 12])]),
            opset_imports=[helper.make_opsetid("", 17)])
        taps = add_taps(model, backbone_stages=True)
        self.assertEqual([tap["name"] for tap in taps], [*STAGE_TAPS])
        self.assertEqual([value.name for value in model.graph.output], ["embedding", *STAGE_TAPS])
        self.assertTrue(all(tap["source_dtype"] == TensorProto.FLOAT16 for tap in taps))
        self.assertTrue(all(value.type.tensor_type.elem_type == TensorProto.FLOAT
                            for value in model.graph.output))
        onnx.checker.check_model(model)

    def test_normalization_preset_brackets_the_same_group_norm_and_excludes_other_taps(self):
        nodes = [helper.make_node("Identity", ["audio"], ["embedding"], name="embedding")]
        nodes.extend(helper.make_node("Cast", ["audio"], [f"norm_{index}"],
                                      name=node_name, to=TensorProto.FLOAT16)
                     for index, node_name in enumerate(STAGE0_NORM_TAPS.values()))
        model = helper.make_model(helper.make_graph(nodes, "norm_fixture", [
            helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 12])], [
            helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, 12])]),
            opset_imports=[helper.make_opsetid("", 17)])
        with self.assertRaisesRegex(ValueError, "choose one"):
            add_taps(model, backbone_stages=True, stage0_norm=True)
        taps = add_taps(model, stage0_norm=True)
        self.assertEqual([tap["name"] for tap in taps], [*STAGE0_NORM_TAPS])
        self.assertEqual([value.name for value in model.graph.output], ["embedding", *STAGE0_NORM_TAPS])
        self.assertTrue(all(value.type.tensor_type.elem_type == TensorProto.FLOAT
                            for value in model.graph.output))
        onnx.checker.check_model(model)

    def test_preparation_preserves_original_graph_and_external_initializers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original_case = root / "original"
            original_case.mkdir()
            (root / "redimnet.weights").write_bytes(np.array([1], "<f4").tobytes())
            tensor = onnx.TensorProto(name="external", data_type=TensorProto.FLOAT, dims=[1],
                                      data_location=TensorProto.EXTERNAL)
            for key, value in [("location", "redimnet.weights"), ("offset", "0"), ("length", "4")]:
                tensor.external_data.add(key=key, value=value)
            nodes = [helper.make_node("Identity", ["audio"], ["embedding"], name="embedding")]
            for index, node_name in enumerate(TAPS.values()):
                nodes.append(helper.make_node("Cast", ["audio"], [f"value_{index}"],
                                               name=node_name, to=TensorProto.FLOAT16 if index == 1 else TensorProto.FLOAT))
            model = helper.make_model(helper.make_graph(nodes, "fixture", [
                helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, 96000])], [
                helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, 96000])], [tensor]),
                opset_imports=[helper.make_opsetid("", 17)])
            original = model.SerializeToString()
            (root / "redimnet.onnx").write_bytes(original)
            np.zeros(96000, "<f4").tofile(original_case / "audio.f32")
            (original_case / "expected.json").write_text("{}")
            (original_case / "case.json").write_text(json.dumps(dict(model="../redimnet.onnx", repeat=3,
                model_files={"../redimnet.weights": sha256(root / "redimnet.weights")},
                inputs=[dict(name="audio", dtype="f32", shape=[1, 96000], path="audio.f32")])))
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(root / "redimnet.onnx", original_case / "case.json",
                        root / "taps.onnx", root / "taps")
            tapped = onnx.load(root / "taps.onnx", load_external_data=False)
            self.assertEqual((root / "redimnet.onnx").read_bytes(), original)
            self.assertEqual(tapped.graph.initializer[0].SerializeToString(), tensor.SerializeToString())
            self.assertEqual([n.SerializeToString() for n in tapped.graph.node[:len(nodes)]],
                             [n.SerializeToString() for n in nodes])
            self.assertEqual([v.name for v in tapped.graph.output], ["embedding", *TAPS])
            self.assertTrue(all(v.type.tensor_type.elem_type == TensorProto.FLOAT
                                for v in tapped.graph.output))
            onnx.checker.check_model(str(root / "taps.onnx"))
            with self.assertRaisesRegex(ValueError, "must be new"):
                prepare(root / "redimnet.onnx", original_case / "case.json",
                        root / "taps.onnx", root / "taps")

    def test_embedding_change_reports_failure_disappearing_without_qualifying_it(self):
        reference = np.eye(1, 192, dtype=np.float32).reshape(-1)
        baseline = np.roll(reference, 1)[None]
        result = embedding_report(reference[None], baseline, reference)
        self.assertTrue(result["numerical_perturbation"])
        self.assertTrue(result["gate_changed"])
        self.assertFalse(result["original_gate_passed"])
        self.assertTrue(result["diagnostic_gate_passed"])

    def test_reader_rejects_missing_duplicate_and_corrupt_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "output.f32"
            np.ones(2, "<f4").tofile(path)
            row = dict(run=0, name="tap", dtype="f32", shape=[2], file=path.name, sha256=sha256(path))
            for rows in [[], [row, row]]:
                with self.assertRaisesRegex(ValueError, "missing or duplicate"):
                    read_outputs({"outputs": rows}, root, ["tap"], 1)
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "hash differs"):
                read_outputs({"outputs": [row]}, root, ["tap"], 1)


if __name__ == "__main__":
    unittest.main()
