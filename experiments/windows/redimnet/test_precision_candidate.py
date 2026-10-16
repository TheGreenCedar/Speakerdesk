"""Small arithmetic/scope checks only; no model or provider inference."""
import copy
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from precision_candidate import STAGE_AFFINES, promote_affines


def fixture():
    prefix = STAGE_AFFINES[0]
    shape = [1, 4608, 2]
    gamma = np.linspace(-0.35, 0.7, 4608, dtype=np.float16)
    beta = np.linspace(0.33, -0.4, 4608, dtype=np.float16)
    tensors = [numpy_helper.from_array(value, name) for name, value in [
        ("group_shape", np.array([0, 64, -1], np.int64)),
        ("unit_scale", np.ones(64, np.float16)),
        ("unit_bias", np.zeros(64, np.float16)),
        ("axes", np.array([1], np.int64)),
        ("stage.10.weight", gamma), ("stage.10.bias", beta)]]

    def node(op, inputs, output, suffix, **attrs):
        return helper.make_node(op, inputs, [output], name=prefix + "/" + suffix, **attrs)

    nodes = [node("Reshape", ["input", "group_shape"], "grouped", "Reshape"),
             node("InstanceNormalization", ["grouped", "unit_scale", "unit_bias"],
                  "unit", "InstanceNormalization", epsilon=1e-5),
             node("Shape", ["input"], "original_shape", "Shape"),
             node("Reshape", ["unit", "original_shape"], "restored", "Reshape_1"),
             node("Unsqueeze", ["stage.10.weight", "axes"], "gamma", "Unsqueeze"),
             node("Mul", ["restored", "gamma"], "scaled", "Mul"),
             node("Unsqueeze", ["stage.10.bias", "axes"], "beta", "Unsqueeze_1"),
             node("Add", ["scaled", "beta"], "affine", "Add"),
             helper.make_node("Identity", ["affine"], ["embedding"], name="downstream")]
    model = helper.make_model(helper.make_graph(nodes, "affine_control", [
        helper.make_tensor_value_info("input", TensorProto.FLOAT16, shape)], [
        helper.make_tensor_value_info("embedding", TensorProto.FLOAT16, shape),
        helper.make_tensor_value_info("unit", TensorProto.FLOAT16, [1, 64, 144])], tensors),
        opset_imports=[helper.make_opsetid("", 17)])
    return model, gamma, beta


class PrecisionCandidate(unittest.TestCase):
    def test_affine_uses_same_learned_bits_with_fp32_arithmetic_and_original_fp16_boundary(self):
        original, gamma, beta = fixture()
        candidate = copy.deepcopy(original)
        promote_affines(candidate, stage0_only=True)
        onnx.checker.check_model(candidate)
        checked = onnx.shape_inference.infer_shapes(candidate, check_type=True, strict_mode=True)
        types = {value.name: value.type.tensor_type.elem_type for value in checked.graph.value_info}
        self.assertEqual(types["affine"], TensorProto.FLOAT16)
        self.assertEqual(types["scaled__affine_f32"], TensorProto.FLOAT)
        data = np.random.default_rng(814).normal(size=(1, 4608, 2)).astype(np.float16)
        old_embedding, old_unit = ReferenceEvaluator(original).run(None, {"input": data})
        embedding, unit = ReferenceEvaluator(candidate).run(None, {"input": data})
        expected = (old_unit.reshape(data.shape).astype(np.float32) *
                    gamma.astype(np.float32)[None, :, None] +
                    beta.astype(np.float32)[None, :, None]).astype(np.float16)
        np.testing.assert_array_equal(unit, old_unit)
        np.testing.assert_array_equal(embedding, expected)
        # A control that accidentally leaves the affine FP16 must fail this input.
        self.assertTrue(np.any(old_embedding != expected))
        self.assertEqual([t.SerializeToString() for t in candidate.graph.initializer],
                         [t.SerializeToString() for t in original.graph.initializer])
        changed = {STAGE_AFFINES[0] + "/" + op for op in ("Reshape_1", "Mul", "Add")}
        after = {node.name: node for node in candidate.graph.node}
        for node in original.graph.node:
            if node.name not in changed:
                self.assertEqual(after[node.name].SerializeToString(), node.SerializeToString())

    def test_rejects_other_affine_wiring_and_shared_intermediate_before_mutation(self):
        for broken in ("different_input", "shared_intermediate"):
            with self.subTest(broken=broken):
                model, _, _ = fixture()
                if broken == "different_input":
                    next(node for node in model.graph.node if node.op_type == "Mul").input[0] = "input"
                else:
                    model.graph.node.append(helper.make_node("Identity", ["scaled"], ["other"], name="other"))
                before = model.SerializeToString()
                with self.assertRaisesRegex(ValueError, "unexpected GroupNorm affine wiring|another consumer"):
                    promote_affines(model, stage0_only=True)
                self.assertEqual(model.SerializeToString(), before)


if __name__ == "__main__":
    unittest.main()
