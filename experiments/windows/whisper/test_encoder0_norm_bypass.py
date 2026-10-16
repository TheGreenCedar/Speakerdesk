"""Paired diagnostic preserves the baseline and intervenes at one state only."""
import unittest

import numpy as np
import onnxruntime as ort
from onnx import TensorProto

from encoder0_norm_bypass import prefix_model
from export import Graph, value


def session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


class EncoderNormIntervention(unittest.TestCase):
    def test_baseline_is_preserved_and_bypass_changes_only_selected_norm(self):
        shape = [1, 3, 4]
        g = Graph({})
        g.op("Cast", "original_input", name="Cast_42", to=TensorProto.FLOAT16)
        full = g.op("Cast", "Cast_42", to=TensorProto.FLOAT)
        normalized = g.op("Add", full, g.const(np.asarray(.5, np.float32)))
        g.op("Cast", normalized, name="LayerNormalization_43", to=TensorProto.FLOAT16)
        g.op("Cast", "LayerNormalization_43", name="Cast_44", to=TensorProto.FLOAT)
        residual = g.op("Add", full, "Cast_44")
        g.op("Cast", residual, name="Cast_116", to=TensorProto.FLOAT16)
        g.op("Cast", "Cast_116", name="original_output", to=TensorProto.FLOAT)
        source = g.model("small-intervention-boundary", [value("original_input", shape)],
                         [value("original_output", shape)])
        before = source.SerializeToString()
        prefix, _ = prefix_model(source)
        self.assertEqual(source.SerializeToString(), before)
        x = (np.arange(12, dtype=np.float32).reshape(shape) / 8 - .5)
        injected = np.full(shape, -.25, np.float32)
        original, = session(source).run(None, {"original_input": x})
        runtime = session(prefix)
        normal, baseline = runtime.run(None, {"positions": x, "source_first_norm": injected,
                                              "bypass": np.asarray(0, np.int64)})
        np.testing.assert_array_equal(baseline.view(np.uint32), original.view(np.uint32))
        np.testing.assert_array_equal(normal, x + .5)
        normal, bypass = runtime.run(None, {"positions": x, "source_first_norm": injected,
                                            "bypass": np.asarray(1, np.int64)})
        np.testing.assert_array_equal(normal.view(np.uint32), injected.view(np.uint32))
        np.testing.assert_array_equal(bypass, x + injected)
        self.assertGreater(np.count_nonzero(bypass != baseline), 0)


if __name__ == "__main__":
    unittest.main()
