"""The lookup index must cover every Half bin, sign and special pattern."""
import unittest

import numpy as np
import onnxruntime as ort

from export import Graph, value
from gelu_table import half_index


class HalfIndex(unittest.TestCase):
    def test_complete_binary16_domain_including_signed_zero_and_power_boundaries(self):
        bits = np.arange(65536, dtype=np.uint16)
        with np.errstate(all="ignore"):
            inputs = bits.view(np.float16).astype(np.float32)
        expected = bits.astype(np.float32)
        expected[(bits & 0x7fff) > 0x7c00] = 0x7e00
        g = Graph({})
        g.op("Identity", half_index(g, "x"), name="index")
        model = g.model("complete-binary16-index", [value("x", [len(bits)])], [value("index", [len(bits)])])
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        actual, = ort.InferenceSession(model.SerializeToString(), options,
            providers=["CPUExecutionProvider"]).run(None, {"x": inputs})
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual[0], 0)
        self.assertEqual(actual[0x8000], 0x8000)


if __name__ == "__main__":
    unittest.main()
