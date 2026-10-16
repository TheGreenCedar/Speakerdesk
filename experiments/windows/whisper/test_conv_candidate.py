"""Both source convolution paths retain contraction and bias rounding."""
import unittest

import numpy as np
import onnxruntime as ort
from onnx import TensorProto

from conv_candidate import preserve_convolution_boundaries
from export import Graph, value


class ConvolutionCandidate(unittest.TestCase):
    def test_two_convolutions_keep_separate_rounds_padding_and_stride(self):
        rng = np.random.default_rng(0x53545249)
        mel = (rng.integers(-32, 33, (1, 17, 3)) / 8).astype(np.float32)
        stages = [(rng.integers(-32, 33, (2, 3, 3)) / 8,
                   [.09375, -.68115234375], 1),
                  (rng.integers(-32, 33, (3, 2, 3)) / 8,
                   [.333251953125, -.09375, .68115234375], 2)]
        graph = Graph({})
        x = graph.op("Cast", "mel", to=TensorProto.FLOAT16)
        expected = mel
        for index, (weight, bias, stride) in enumerate(stages):
            weight, bias = np.asarray(weight, np.float16), np.asarray(bias, np.float16)
            x = graph.op("Transpose", x, perm=[0, 2, 1])
            x = graph.op("Conv", x, graph.const(weight, "weight_" + str(index)),
                         pads=[1, 1], strides=[stride])
            x = graph.op("Cast", graph.op("Cast", x, to=TensorProto.FLOAT16), to=TensorProto.FLOAT16)
            x = graph.op("Transpose", x, perm=[0, 2, 1])
            x = graph.op("Add", x, graph.const(bias, "bias_" + str(index)))
            x = graph.op("Cast", graph.op("Cast", x, to=TensorProto.FLOAT16), to=TensorProto.FLOAT16)
            frames = (expected.shape[1] + stride - 1) // stride
            contraction = np.zeros((1, frames, len(bias)), np.float64)
            for frame in range(frames):
                for kernel in range(3):
                    source = frame * stride + kernel - 1
                    if 0 <= source < expected.shape[1]:
                        contraction[0, frame] += weight[:, :, kernel].astype(np.float64) @ expected[0, source].astype(np.float64)
            expected = (contraction.astype(np.float16).astype(np.float64) + bias).astype(np.float16).astype(np.float32)
        graph.op("Cast", x, name="rounded", to=TensorProto.FLOAT)
        graph.op("Mul", "mel", graph.const(np.asarray(1.12345, np.float32)), name="passthrough")
        source = graph.model("two-source-convolutions", [value("mel", [1, 17, 3])],
                             [value("rounded", [1, 9, 3]), value("passthrough", [1, 17, 3])])
        corrected, _ = preserve_convolution_boundaries(source)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(corrected.SerializeToString(), options, providers=["CPUExecutionProvider"])
        rounded, passthrough = session.run(None, {"mel": mel})
        np.testing.assert_array_equal(rounded, expected)
        np.testing.assert_array_equal(passthrough, mel * np.float32(1.12345))


if __name__ == "__main__":
    unittest.main()
