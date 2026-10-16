"""A Half addition must round before its promoted Float32 consumer."""
import unittest

import numpy as np
import onnxruntime as ort

from implicit_control import bias_model


class ImplicitBoundary(unittest.TestCase):
    def test_half_add_rounds_before_float_consumer_for_ties_subnormals_and_signs(self):
        half = np.arange(0x7c00, dtype=np.uint16).view(np.float16)
        half = half[half <= 16]
        x = np.concatenate([half, -half]).astype(np.float32).reshape(-1, 1)
        bias = np.asarray([.09375], np.float16)
        expected = (x + bias.astype(np.float32)).astype(np.float16).astype(np.float32)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(bias_model(bias, rows=len(x)).SerializeToString(), options,
                                       providers=["CPUExecutionProvider"])
        native, corrected = session.run(None, {"x": x})
        # Half inputs plus a Half bias are exactly representable in Float32 in
        # this range, so NumPy's binary16 cast is an independent rounding oracle.
        self.assertGreater(np.count_nonzero(native != expected), 1000)
        np.testing.assert_array_equal(corrected, expected)
        self.assertTrue(np.isfinite(corrected).all())


if __name__ == "__main__":
    unittest.main()
