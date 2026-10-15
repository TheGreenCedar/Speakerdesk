"""Artifact transformation conservation and causal-class guards; no inference."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from export import ROOT, SOURCE_FILES, SOURCE_REVISION, sha256
from normalization import apply_stage, inventory
from normalize_artifact import transform
from test_global_norm_candidate import fixture


def artifact(root):
    names = ["encoder"] + [name for index in range(24) for name in [f"flash-{index:02d}", f"fsmn-{index:02d}"]] + ["finish"]
    graphs = []
    for name in names:
        if name in {"encoder", "finish"}:
            model = fixture(2, "/norm" if name == "encoder" else "/global_norm")
        else:
            model = helper.make_model(helper.make_graph([
                helper.make_node("Identity", ["x"], ["x_out"], name="unchanged")], name, [
                helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 512])], [
                helper.make_tensor_value_info("x_out", TensorProto.FLOAT, [1, 2, 512])]),
                opset_imports=[helper.make_opsetid("", 17)])
        path = root / (name + ".onnx")
        path.write_bytes(model.SerializeToString())
        graphs.append(dict(name=name, model=path.name, sha256=sha256(path), operators=[n.op_type for n in model.graph.node]))
    (root / "mossformer.weights").write_bytes(b"retained parameters")
    program = dict(checkpoint=json.loads((ROOT / "crates/host/src/separation-artifact.json").read_text()),
        source_revision=SOURCE_REVISION, source_sha256=SOURCE_FILES, dtype="float32", latent_channels=512,
        sample_bounds=[16, 392000], graphs=graphs, model_files={"mossformer.weights": sha256(root / "mossformer.weights")},
        stages_sha256=sha256(Path(__file__).with_name("stages.py")), source_adaptations=[], exporter_sha256="a" * 64)
    (root / "program.json").write_text(json.dumps(program))
    return program


class NormalizeArtifact(unittest.TestCase):
    def test_encoder_norm_source_shape_is_read_when_encoded_is_also_a_graph_output(self):
        model = fixture(2, "/norm")
        model.graph.input[0].name = "raw"
        model.graph.node.insert(0, helper.make_node("Identity", ["raw"], ["flat"], name="encoder"))
        model.graph.output.append(helper.make_tensor_value_info("flat", TensorProto.FLOAT, [1, 512, 2]))
        record = inventory(model)[0]
        self.assertEqual(record["source_shape"], [1, 512, 2])
        self.assertTrue(record["same_global_groupnorm_class"])
        _, changed = apply_stage(model, "encoder")
        self.assertEqual(len(changed), 1)

    def test_only_proven_global_class_is_transformed_and_other_stage_names_are_rejected(self):
        model = fixture(2, "/norm")
        self.assertTrue(inventory(model)[0]["same_global_groupnorm_class"])
        with self.assertRaisesRegex(ValueError, "inventory differs"):
            apply_stage(model, "fsmn-00")
        before, changed = apply_stage(model, "encoder")
        self.assertEqual(len(before), 1)
        self.assertEqual(len(changed), 1)
        self.assertEqual(changed[0]["removed_node"]["name"], "/norm/InstanceNormalization")
        per_channel = helper.make_model(helper.make_graph([
            helper.make_node("InstanceNormalization", ["x", "weight", "bias"], ["y"],
                             name="/dense/norm1/InstanceNormalization", epsilon=1e-5)], "fsmn_norm", [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 256, 2, 1])], [
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 256, 2, 1])], [
            numpy_helper.from_array(np.linspace(.1, 2, 256, dtype=np.float32), "weight"),
            numpy_helper.from_array(np.zeros(256, np.float32), "bias")]),
            opset_imports=[helper.make_opsetid("", 17)])
        serialized = per_channel.SerializeToString()
        norms, changed = apply_stage(per_channel, "fsmn-00")
        self.assertFalse(norms[0]["same_global_groupnorm_class"])
        self.assertEqual(changed, [])
        self.assertEqual(per_channel.SerializeToString(), serialized)

    def test_frozen_transform_copies_48_graphs_weights_and_source_program_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "original"
            source.mkdir()
            original = artifact(source)
            output = Path(temporary) / "normalized"
            with contextlib.redirect_stdout(io.StringIO()):
                program = transform(source, output)
            self.assertEqual(program["frozen_transform"]["changed_stages"], ["encoder", "finish"])
            self.assertEqual((source / "mossformer.weights").read_bytes(), (output / "mossformer.weights").read_bytes())
            self.assertEqual((source / "program.json").read_bytes(), (output / "source-program.json").read_bytes())
            self.assertEqual(program["exporter_sha256"], original["exporter_sha256"])
            for graph in original["graphs"]:
                old, new = source / graph["model"], output / graph["model"]
                if graph["name"] not in {"encoder", "finish"}:
                    self.assertEqual(old.read_bytes(), new.read_bytes())
                    continue
                before = onnx.load(old, load_external_data=False)
                after = onnx.load(new, load_external_data=False)
                self.assertEqual([t.SerializeToString() for t in before.graph.initializer],
                    [t.SerializeToString() for t in after.graph.initializer[:len(before.graph.initializer)]])
            with self.assertRaisesRegex(ValueError, "must be new"):
                transform(source, output)

    def test_modified_source_graph_or_weights_are_rejected_before_output_is_created(self):
        for name in ["fsmn-03.onnx", "mossformer.weights"]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                source = Path(temporary) / "original"
                source.mkdir()
                artifact(source)
                (source / name).write_bytes(b"modified")
                output = Path(temporary) / "normalized"
                with self.assertRaisesRegex(ValueError, "differs|weights differ"):
                    transform(source, output)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
