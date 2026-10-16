"""Lookup replacement preserves surrounding behavior and rejects leaked stages."""
import unittest

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from export import value
from gelu_boundary import operator_model, preserve_gelu_boundaries
from gelu_table_candidate import preserve_compiled_gelu
from rounding_candidate import preserve_half_conversions


def source_model(size):
    source, _ = preserve_gelu_boundaries(operator_model([size]), expected_count=1)
    source, _ = preserve_half_conversions(source, expected_count=6)
    return source


class GeluTableCandidate(unittest.TestCase):
    def test_supplied_full_domain_function_preserves_half_boundary_and_other_output(self):
        half = np.arange(65536, dtype=np.uint16).view(np.float16)
        inputs = half[np.isfinite(half)].astype(np.float32)
        with np.errstate(all="ignore"):
            table = (-half).astype(np.float32)
        source = source_model(len(inputs))
        scale = np.asarray(.70001, np.float32)
        source.graph.initializer.append(numpy_helper.from_array(scale, "outside_scale"))
        source.graph.node.append(helper.make_node("Mul", ["x", "outside_scale"], ["passthrough"]))
        source.graph.output.append(value("passthrough", [len(inputs)]))
        original = source.SerializeToString()
        actual_model, changes = preserve_compiled_gelu(source, table, expected_count=1)
        self.assertEqual(source.SerializeToString(), original)
        self.assertEqual(len(changes), 1)
        original_parameters = {item.name: item.SerializeToString() for item in source.graph.initializer}
        actual_parameters = {item.name: item.SerializeToString() for item in actual_model.graph.initializer}
        self.assertTrue(all(actual_parameters[name] == data for name, data in original_parameters.items()))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.log_severity_level = 3
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        actual, other = ort.InferenceSession(actual_model.SerializeToString(), options,
            providers=["CPUExecutionProvider"]).run(None, {"x": inputs})
        np.testing.assert_array_equal(actual.view(np.uint32), (-inputs).view(np.uint32))
        np.testing.assert_array_equal(other, inputs * scale)

    def test_observable_gelu_intermediate_cannot_be_discarded(self):
        source = source_model(8)
        source.graph.output.append(value("divided", [8], TensorProto.FLOAT))
        with self.assertRaisesRegex(ValueError, "observable intermediate"):
            preserve_compiled_gelu(source, np.zeros(65536, np.float32), expected_count=1)


if __name__ == "__main__":
    unittest.main()
