"""Source placement and exact constant-row normalization boundaries."""
import unittest

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from export import Graph, value
from materialization_candidate import preserve_source_materializations, source_norm
from materialization_control import norm


def session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


class MaterializationCandidate(unittest.TestCase):
    def test_encoder_decoder_and_unbiased_dot_placement_preserves_parameters_and_other_output(self):
        rng = np.random.default_rng(0x504C4143)
        x = (rng.integers(-32, 33, (32, 384)) / 32).astype(np.float32)
        w = (rng.integers(-32, 33, (384, 384)) / 32).astype(np.float16)
        b = np.linspace(-.9, .9, 384).astype(np.float16)
        g = Graph({})
        encoded = g.op("MatMul", g.op("Cast", "x", to=TensorProto.FLOAT16),
                       g.const(w, "encoder.test.weight"))
        g.op("Identity", g.op("Add", encoded, g.const(b, "encoder.test.bias")), name="encoded")
        decoded = g.op("MatMul", g.op("Cast", "decoder", to=TensorProto.FLOAT16),
                       g.const(w, "decoder.test.weight"))
        g.op("Identity", g.op("Add", decoded, g.const(b, "decoder.test.bias")), name="decoded")
        g.op("Identity", g.op("MatMul", g.op("Cast", "x", to=TensorProto.FLOAT16),
                              g.const(w, "attention_weight")), name="attention")
        # Real source input names must not collide with names generated while
        # expanding the much larger LayerNorm expression.
        n = g.op("Cast", "normal", name="Cast_92", to=TensorProto.FLOAT16)
        g.op("LayerNormalization", n, g.const(np.ones(384, np.float16), "norm_weight"),
             g.const(np.zeros(384, np.float16), "norm_bias"), name="normalized", epsilon=1e-5)
        g.op("Mul", "x", g.const(np.asarray(.70001, np.float32)), name="passthrough")
        source = g.model("source-placement", [value("x", [32, 384]), value("decoder", [1, 384]), value("normal", [2, 384])],
            [value("encoded", [32, 384], TensorProto.FLOAT16), value("decoded", [1, 384], TensorProto.FLOAT16),
             value("attention", [32, 384], TensorProto.FLOAT16), value("normalized", [2, 384], TensorProto.FLOAT16),
             value("passthrough", [32, 384])])
        original = source.SerializeToString()
        corrected, _ = preserve_source_materializations(source,
            {"matmul": 3, "encoder_biased": 1, "decoder_biased": 1, "layer_norm": 1})
        self.assertEqual(source.SerializeToString(), original)
        normal = np.asarray([np.full(384, 4), np.tile([4 + 2**-7, 4 - 2**-7], 192)], np.float32)
        actual = session(corrected).run(None, {"x": x, "decoder": x[:1].copy(), "normal": normal})
        exact = (x.astype(np.float64) @ w.astype(np.float64)).astype(np.float32)
        np.testing.assert_array_equal(actual[0], (exact + b.astype(np.float32)).astype(np.float16))
        np.testing.assert_array_equal(actual[1], (exact[:1].astype(np.float16).astype(np.float32) + b).astype(np.float16))
        np.testing.assert_array_equal(actual[2], exact.astype(np.float16))
        np.testing.assert_array_equal(actual[3][0], np.zeros(384, np.float16))
        self.assertGreater(np.max(abs(actual[3][1])), .5)
        np.testing.assert_array_equal(actual[4], x * np.float32(.70001))

    def test_constant_row_is_zero_even_when_backend_mean_reciprocal_is_one_ulp_low(self):
        x = np.asarray([np.full(384, 4), np.tile([4 + 2**-7, 4 - 2**-7], 192)], np.float32)
        def model(lowering):
            g = Graph({})
            result = lowering(g, "x", g.const(np.ones(384, np.float32)),
                              g.const(np.full(384, .125, np.float32)))
            g.op("Identity", result, name="y")
            m = g.model("mean-reciprocal-boundary", [value("x", [2, 384])], [value("y", [2, 384])])
            # Emulate the observed DML constant-row failure without a production
            # test hook: only the mean's backend reduction is changed here.
            index = next(i for i, node in enumerate(m.graph.node) if node.op_type == "ReduceMean")
            original = m.graph.node[index]
            sum_name = "backend_sum"
            replacement = [helper.make_node("ReduceSum", [*original.input, "backend_axes"], [sum_name], keepdims=1),
                           helper.make_node("Mul", [sum_name, "backend_reciprocal"], list(original.output))]
            del m.graph.node[index]
            for offset, node in enumerate(replacement):
                m.graph.node.insert(index + offset, node)
            m.graph.initializer.append(numpy_helper.from_array(np.asarray([-1], np.int64), "backend_axes"))
            m.graph.initializer.append(numpy_helper.from_array(np.asarray(
                np.nextafter(np.float32(1 / 384), np.float32(0))), "backend_reciprocal"))
            return m
        before = session(model(norm)).run(None, {"x": x})[0]
        after = session(model(lambda g, x, w, b: source_norm(g, x, w, b, 1e-5))).run(None, {"x": x})[0]
        self.assertGreater(np.max(abs(before[0] - np.float32(.125))), 0)
        np.testing.assert_array_equal(after[0], np.full(384, .125, np.float32))
        self.assertGreater(np.max(abs(after[1] - np.float32(.125))), .5)


if __name__ == "__main__":
    unittest.main()
