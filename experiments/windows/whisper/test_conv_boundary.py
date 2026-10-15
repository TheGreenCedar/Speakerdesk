"""Convolution rounds before bias; padded edges retain the same boundary."""
import unittest

import numpy as np
import onnxruntime as ort

from conv_control import operator_model


class ConvolutionBoundary(unittest.TestCase):
    def test_float_convolution_retains_half_result_before_bias_at_edges_and_interior(self):
        rng = np.random.default_rng(0x434F4E56)
        x = (rng.integers(-512, 513, (1, 17, 3)) / 128).astype(np.float16).astype(np.float32)
        weight = (rng.integers(-512, 513, (2, 3, 3)) / 128).astype(np.float16)
        bias = np.asarray([.09375, -.68115234375], np.float16)
        exact = np.zeros((1, 17, 2), np.float64)
        for frame in range(17):
            for channel in range(2):
                for kernel in range(3):
                    source = frame + kernel - 1
                    if 0 <= source < 17:
                        exact[0, frame, channel] += np.dot(
                            x[0, source].astype(np.float64), weight[channel, :, kernel].astype(np.float64))
        rounded = exact.astype(np.float16).astype(np.float32)
        expected = (rounded.astype(np.float64) + bias).astype(np.float16).astype(np.float32)
        fused = (exact + bias).astype(np.float16).astype(np.float32)
        self.assertGreater(np.count_nonzero(fused != expected), 0)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(operator_model(weight, bias, frames=17).SerializeToString(), options,
                                       providers=["CPUExecutionProvider"])
        actual = dict(zip([item.name for item in session.get_outputs()], session.run(None, {"mel": x})))
        np.testing.assert_array_equal(actual["f32_convolution"], exact.astype(np.float32))
        np.testing.assert_array_equal(actual["corrected_convolution"], rounded)
        np.testing.assert_array_equal(actual["corrected_bias"], expected)


if __name__ == "__main__":
    unittest.main()
