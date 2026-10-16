"""Synthetic arithmetic only; no learned-network or native provider execution."""
import unittest

import numpy as np
from onnx import numpy_helper
from onnx.reference import ReferenceEvaluator

from affine_operator import RAW_SHAPE, RESTORED_SHAPE, arithmetic_oracle, build_model


class AffineOperator(unittest.TestCase):
    def test_original_half_parameters_promote_exactly_and_outputs_obey_arithmetic_bounds(self):
        gamma = np.linspace(-0.35, 0.7, 4608).astype(np.float16)
        beta = np.linspace(0.33, -0.4, 4608).astype(np.float16)
        raw = np.random.default_rng(15).normal(size=RAW_SHAPE).astype(np.float32)
        model = build_model(gamma, beta)
        parameters = {tensor.name: numpy_helper.to_array(tensor) for tensor in model.graph.initializer}
        np.testing.assert_array_equal(parameters["gamma"].reshape(-1), gamma.astype(np.float32))
        np.testing.assert_array_equal(parameters["beta"].reshape(-1), beta.astype(np.float32))
        oracle, bounds = arithmetic_oracle(raw, parameters["gamma"], parameters["beta"])
        full, half = ReferenceEvaluator(model).run(None, {"raw_norm": raw})
        self.assertEqual(list(full.shape), RESTORED_SHAPE)
        np.testing.assert_array_equal(half, full.astype(np.float16).astype(np.float32))
        self.assertTrue(np.all(np.abs(full.astype(np.float64) - oracle) <= bounds[0]))
        self.assertTrue(np.all(np.abs(half.astype(np.float64) - oracle) <= bounds[1]))
        # The observed whole-graph .53 gain cannot be hidden by these rounding bounds.
        self.assertGreater(np.count_nonzero(np.abs(half * .53 - oracle) > bounds[1]), raw.size * .99)

    def test_rejects_non_source_parameter_precision_shape_and_nonfinite_values(self):
        valid = np.ones(4608, np.float16)
        invalid = [valid.astype(np.float32), valid[:-1], np.full(4608, np.nan, np.float16)]
        for value in invalid:
            with self.subTest(shape=value.shape, dtype=value.dtype):
                with self.assertRaisesRegex(ValueError, "original FP16"):
                    build_model(value, valid)


if __name__ == "__main__":
    unittest.main()
