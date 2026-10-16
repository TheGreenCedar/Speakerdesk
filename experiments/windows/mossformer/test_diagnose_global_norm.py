"""Independent synthetic controls for reduction attribution; no model inference."""
import unittest

import numpy as np

from diagnose_global_norm import own_input_oracle


class GlobalNormOracle(unittest.TestCase):
    def test_recovers_wrong_scalar_variance_without_confusing_it_with_affine_or_input_drift(self):
        x = np.array([[[1., 7.], [2., -4.], [3., 2.], [-1., 5.]]], np.float32)
        gamma = np.array([.2, 1.4], np.float32)
        beta = np.array([-.1, .3], np.float32)
        mean, variance = float(x.astype(np.float64).mean()), float(x.astype(np.float64).var())
        for variance_scale, mean_shift in [(1., 0.), (1.02, .01)]:
            with self.subTest(variance_scale=variance_scale):
                output = (((x.astype(np.float64) - mean - mean_shift) /
                    np.sqrt(variance * variance_scale + 1e-8)) * gamma + beta).astype(np.float32).transpose(0, 2, 1)
                result = own_input_oracle(x, output, gamma, beta, 1e-8)
                fitted = result["fitted_scalar_normalization"]
                self.assertAlmostEqual(fitted["implied_variance"], variance * variance_scale, places=5)
                self.assertAlmostEqual(fitted["implied_mean"], mean + mean_shift, places=6)
                self.assertLess(fitted["rms_residual"], 1e-7)
                error = result["output_vs_own_input_f64"]["normalized_rms_error"]
                if variance_scale == 1.:
                    self.assertLess(error, 1e-7)
                else:
                    self.assertGreater(error, .005)

    def test_rejects_wrong_layout_or_nonfinite_captures(self):
        x = np.arange(8, dtype=np.float32).reshape(1, 4, 2)
        gamma, beta = np.ones(2, np.float32), np.zeros(2, np.float32)
        for source, output in [(x, x), (np.full_like(x, np.nan), x.transpose(0, 2, 1)),
                               (x, np.full((1, 2, 4), np.inf, np.float32))]:
            with self.assertRaises(ValueError):
                own_input_oracle(source, output, gamma, beta, 1e-8)


if __name__ == "__main__":
    unittest.main()
