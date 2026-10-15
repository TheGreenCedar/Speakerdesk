"""Hostile numerical regression for the isolated Half rounding control."""
import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from rounding_control import differing, expectations, hostile_values, native_values, operator_model
from rounding_candidate import preserve_half_conversions


class Rounding(unittest.TestCase):
    def test_conversion_transform_preserves_float_passthrough_and_half_arithmetic(self):
        x = native_values(hostile_values())
        nodes = [helper.make_node("Cast", ["x"], ["half"], to=TensorProto.FLOAT16),
                 helper.make_node("Add", ["half", "quarter"], ["sum"]),
                 helper.make_node("Mul", ["sum", "quarter"], ["product"]),
                 helper.make_node("Cast", ["product"], ["already_half"], to=TensorProto.FLOAT16),
                 helper.make_node("Cast", ["already_half"], ["half_result"], to=TensorProto.FLOAT),
                 helper.make_node("Identity", ["x"], ["full_result"])]
        source = helper.make_model(helper.make_graph(nodes, "mixed-precision-boundary",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [len(x)])],
            [helper.make_tensor_value_info(name, TensorProto.FLOAT, [len(x)])
             for name in ["half_result", "full_result"]],
            [numpy_helper.from_array(np.asarray(.25, np.float16), "quarter")]),
            opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
        corrected, changes = preserve_half_conversions(source, expected_count=1)
        self.assertEqual(changes["implicit_half_outputs_unchanged"], {"Add": 1, "Mul": 1})
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        # The CPU's Half promotion may erase the original conversion. Compare
        # its unchanged downstream arithmetic with an already-rounded input.
        original = ort.InferenceSession(source.SerializeToString(), options,
                   providers=["CPUExecutionProvider"]).run(None, {"x": expectations(x)[0]})
        actual = ort.InferenceSession(corrected.SerializeToString(), options,
                   providers=["CPUExecutionProvider"]).run(None, {"x": x})
        self.assertEqual(np.count_nonzero(differing(actual[0], original[0])), 0)
        self.assertEqual(np.count_nonzero(differing(actual[1], x)), 0)

    def test_all_half_bins_ties_neighbors_and_special_values_match_ieee_cast(self):
        x = hostile_values()
        nearest, truncated = expectations(x)
        native = native_values(x)
        self.assertTrue(np.isfinite(native).all())
        self.assertTrue(np.isfinite(expectations(native)[0]).all())
        self.assertTrue(np.any(np.signbit(native[native == 0])))
        self.assertTrue(np.any(~np.signbit(native[native == 0])))
        self.assertGreater(np.count_nonzero(differing(nearest, truncated)), 100000)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(operator_model().SerializeToString(), options,
                                       providers=["CPUExecutionProvider"])
        for name, actual in zip(["native", "grid", "corrected"], session.run(None, {"x": x})):
            self.assertEqual(np.count_nonzero(differing(actual, nearest)), 0, name)


if __name__ == "__main__":
    unittest.main()
