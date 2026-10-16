"""Small dynamic reduction controls; no learned-network or native execution."""
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator

from global_norm_candidate import replace_global_norm


def fixture(frames, prefix="/global_norm"):
    nodes = [helper.make_node("Constant", [], ["group_shape"], name=prefix + "/Constant",
                             value=numpy_helper.from_array(np.array([0, 1, -1], np.int64))),
             helper.make_node("Reshape", ["flat", "group_shape"], ["grouped"], name=prefix + "/Reshape"),
             helper.make_node("Constant", [], ["one"], name=prefix + "/Constant_1",
                             value=numpy_helper.from_array(np.ones(1, np.float32))),
             helper.make_node("Constant", [], ["zero"], name=prefix + "/Constant_2",
                             value=numpy_helper.from_array(np.zeros(1, np.float32))),
             helper.make_node("InstanceNormalization", ["grouped", "one", "zero"], ["normalized"],
                              name=prefix + "/InstanceNormalization", epsilon=1e-8),
             helper.make_node("Mul", ["normalized", "learned"], ["result"], name="learned_affine")]
    external = onnx.TensorProto(name="unused_external", data_type=TensorProto.FLOAT,
        dims=[1], data_location=TensorProto.EXTERNAL)
    for key, value in [("location", "mossformer.weights"), ("offset", "2048"), ("length", "4")]:
        external.external_data.add(key=key, value=value)
    model = helper.make_model(helper.make_graph(nodes, "reduction_control", [
        helper.make_tensor_value_info("flat", TensorProto.FLOAT, [1, 512, frames])], [
        helper.make_tensor_value_info("result", TensorProto.FLOAT, [1, 1, 512 * frames])],
        [numpy_helper.from_array(np.ones(1, np.float32), "learned"), external]),
        opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    return model


class GlobalNormCandidate(unittest.TestCase):
    def test_centered_variance_covers_empty_and_nonempty_partial_padding(self):
        rng = np.random.default_rng(17)
        for frames in [1, 255, 256, 257, 489]:
            with self.subTest(frames=frames):
                model = fixture(frames)
                # The unused external descriptor is tested separately; no file is
                # required to execute the independent small arithmetic fixture.
                del model.graph.initializer[-1]
                replace_global_norm(model)
                onnx.checker.check_model(model)
                data = (rng.normal(size=(1, 512, frames)) * 3 + .4).astype(np.float32)
                actual = ReferenceEvaluator(model).run(None, {"flat": data})[0]
                values = data.astype(np.float64)
                centered = values - values.mean()
                expected = (centered / np.sqrt(np.mean(centered * centered) + np.float32(1e-8))).reshape(actual.shape)
                # Roundoff budget for the bounded F32 reductions and elementwise
                # arithmetic; this is not a replacement acoustic-model gate.
                np.testing.assert_allclose(actual, expected, rtol=32 * np.finfo(np.float32).eps,
                                           atol=32 * np.finfo(np.float32).eps)
                self.assertEqual(actual.dtype, np.float32)

    def test_only_the_identified_unit_normalization_is_replaced(self):
        model = fixture(257)
        old_nodes = {node.name: node.SerializeToString() for node in model.graph.node}
        old_tensors = [tensor.SerializeToString() for tensor in model.graph.initializer]
        descriptor = replace_global_norm(model)
        self.assertEqual(descriptor["removed_node"]["name"], "/global_norm/InstanceNormalization")
        self.assertEqual(descriptor["reduction_lengths"], [512, 256, 192])
        self.assertEqual(old_tensors, [tensor.SerializeToString() for tensor in model.graph.initializer[:2]])
        for node in model.graph.node:
            if node.name in old_nodes and node.name != "/global_norm/InstanceNormalization":
                self.assertEqual(node.SerializeToString(), old_nodes[node.name])
        self.assertTrue(all(t.data_type != TensorProto.DOUBLE for t in model.graph.initializer))
        self.assertEqual(list(model.graph.output)[0].name, "result")
        with self.assertRaisesRegex(ValueError, "frozen global unit normalization"):
            replace_global_norm(model)


if __name__ == "__main__":
    unittest.main()
