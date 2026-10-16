import copy
import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from attention import EQUATIONS, lower_attention


def fixture(batch=1):
    inputs = {
        "projection": [batch, "frames", 128], "gamma": [4, 128], "frequency": [16],
        **{f"position{i}": ["frames"] for i in range(4)},
        **{name: [batch, "groups", 256, width] for name, width in [
            ("quad_q", 128), ("quad_k", 128), ("linear_q", 128), ("linear_k", 128),
            ("value", 1024), ("gate", 1024)]},
    }
    arguments = [
        ("projection", "gamma"),
        *[(f"position{i}", "frequency") for i in range(4)],
        ("quad_q", "quad_k"), ("scores", "value"), ("scores", "gate"),
        ("linear_k", "value"), ("linear_k", "gate"),
        ("linear_q", "context"), ("linear_q", "gated_context"),
    ]
    outputs = ["offset", *[f"rotary{i}" for i in range(4)], "scores", "quad_value", "quad_gate",
               "context", "gated_context", "linear_value", "linear_gate"]
    nodes = [helper.make_node("Einsum", list(args), [output], name=name, equation=equation)
             for (name, equation), args, output in zip(EQUATIONS.items(), arguments, outputs)]
    # An unrelated operation must survive the shared transform exactly.
    nodes.append(helper.make_node("Identity", ["projection"], ["unchanged"], name="unrelated"))
    shapes = [[batch, "frames", 4, 128], *[["frames", 16] for _ in range(4)],
              [batch, "groups", 256, 256], *[[batch, "groups", 256, 1024] for _ in range(2)],
              *[[batch, 128, 1024] for _ in range(2)],
              *[[batch, "groups", 256, 1024] for _ in range(2)], [batch, "frames", 128]]
    return helper.make_model(helper.make_graph(nodes, "attention", [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name, shape in inputs.items()
    ], [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
        for name, shape in zip([*outputs, "unchanged"], shapes)]),
        opset_imports=[helper.make_opsetid("", 17)], ir_version=10)


def reference(inputs):
    # NumPy Float64 evaluates the source equations independently of the ONNX lowering.
    values = {name: value.astype(np.float64) for name, value in inputs.items()}
    values["offset"] = np.einsum("btd,hd->bthd", values["projection"], values["gamma"])
    for i in range(4):
        values[f"rotary{i}"] = np.einsum("t,f->tf", values[f"position{i}"], values["frequency"])
    values["scores"] = np.einsum("bgid,bgjd->bgij", values["quad_q"], values["quad_k"])
    for name, source in [("value", "value"), ("gate", "gate")]:
        values[f"quad_{name}"] = np.einsum("bgij,bgjd->bgid", values["scores"], values[source])
        context = "context" if name == "value" else "gated_context"
        values[context] = np.einsum("bgnd,bgne->bde", values["linear_k"], values[source], optimize=True)
        values[f"linear_{name}"] = np.einsum("bgnd,bde->bgne", values["linear_q"], values[context])
    values["unchanged"] = values["projection"]
    return values


class Attention(unittest.TestCase):
    def test_all_source_equations_match_independent_float64_across_padding_boundaries(self):
        original = fixture()
        lowered = copy.deepcopy(original)
        records = lower_attention(lowered)
        self.assertEqual(len(records), 10)
        self.assertEqual([n.name for n in lowered.graph.node if n.op_type == "Einsum"],
                         ["/block/Einsum_8", "/block/Einsum_10"])
        self.assertEqual(original.graph.node[-1].SerializeToString(), lowered.graph.node[-1].SerializeToString())
        onnx.checker.check_model(lowered, full_check=True)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sessions = [ort.InferenceSession(model.SerializeToString(), options,
                    providers=["CPUExecutionProvider"]) for model in [original, lowered]]
        rng = np.random.default_rng(42)
        for frames in [1, 255, 256, 257]:
            with self.subTest(frames=frames):
                groups = (frames + 255) // 256
                inputs = {value.name: rng.normal(0, .1, [
                    frames if d.dim_param == "frames" else groups if d.dim_param == "groups" else d.dim_value
                    for d in value.type.tensor_type.shape.dim]).astype(np.float32)
                    for value in original.graph.input}
                for name in ["quad_q", "quad_k", "linear_q", "linear_k", "value", "gate"]:
                    # Source grouping pads only the final group; padded keys/values contribute zero.
                    inputs[name].reshape(1, groups * 256, -1)[:, frames:] = 0
                expected = reference(inputs)
                for session in sessions:
                    for output, actual in zip(session.get_outputs(), session.run(None, inputs)):
                        np.testing.assert_allclose(actual, expected[output.name], rtol=1e-5, atol=2e-6,
                                                   err_msg=f"{output.name}, frames={frames}")

    def test_incompatible_inventory_shapes_and_types_are_rejected_before_mutation(self):
        mutations = [
            lambda model: setattr(model.graph.node[0].attribute[0], "s", b"...d,hd->...h"),
            lambda model: model.graph.node.pop(0),
            lambda model: model.graph.node.append(copy.deepcopy(model.graph.node[0])),
            lambda model: setattr(model.graph.input[0].type.tensor_type.shape.dim[0], "dim_value", 2),
            lambda model: setattr(model.graph.input[1].type.tensor_type.shape.dim[0], "dim_value", 3),
            lambda model: setattr(model.graph.input[2].type.tensor_type.shape.dim[0], "dim_value", 8),
            lambda model: setattr(model.graph.input[-1].type.tensor_type.shape.dim[2], "dim_value", 255),
            lambda model: setattr(model.graph.input[1].type.tensor_type, "elem_type", TensorProto.FLOAT16),
            lambda model: model.graph.input[0].type.tensor_type.shape.dim.pop(),
        ]
        for mutate in mutations:
            model = fixture()
            mutate(model)
            before = model.SerializeToString()
            with self.subTest(mutation=mutations.index(mutate)):
                with self.assertRaises((ValueError, onnx.shape_inference.InferenceError)):
                    lower_attention(model)
                self.assertEqual(model.SerializeToString(), before)

    def test_lowering_cannot_be_applied_twice(self):
        model = fixture()
        lower_attention(model)
        before = model.SerializeToString()
        with self.assertRaises(ValueError):
            lower_attention(model)
        self.assertEqual(model.SerializeToString(), before)


if __name__ == "__main__":
    unittest.main()
