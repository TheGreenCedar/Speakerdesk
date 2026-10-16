"""Protect diagnostic bit errors, complete branches, and exact repetition."""
import copy
import unittest

import numpy as np

import gelu_control as control


class GeluControl(unittest.TestCase):
    def test_bit_and_ulp_errors_do_not_hide_signed_zero_or_one_ulp(self):
        expected = np.asarray([-1, -0.0, 0, 1], np.float32)
        actual = expected.copy()
        actual[1] = 0
        actual[3] = np.nextafter(np.float32(1), np.float32(2))
        report = control.metrics(actual, expected)
        self.assertEqual(report["unequal_bits"], 2)
        self.assertEqual(report["maximum_ulp_distance"], 1)
        self.assertEqual(report["max_absolute_error"], 2 ** -23)

    def test_complete_repeating_branches_and_native_erf_reassembly(self):
        x = np.asarray([0, 1], np.float32)
        expected = np.asarray([0, 0.5], np.float32)
        run = dict(fused_gelu=expected.copy(), bias_gelu=expected.copy(), scaled=x / control.SQRT2,
                   native_erf=np.asarray([-1, 0], np.float32), one_plus_erf=x.copy(),
                   product=x.copy(), explicit_gelu=expected.copy())
        run["fused_gelu"][1] = np.nextafter(np.float32(0.5), np.float32(1))
        runs = [run, copy.deepcopy(run)]
        report = control.summarize(x, expected, runs)[0]
        self.assertEqual(report["versus_retained_source"]["fused_gelu"]["maximum_ulp_distance"], 1)
        self.assertEqual(report["native_erf_reassembled_vs_source"]["unequal_bits"], 0)
        self.assertEqual(report["explicit_vs_reassembled"]["unequal_bits"], 0)
        self.assertEqual(report["scaled_vs_ieee_division"]["unequal_bits"], 0)
        for defect in ("missing", "extra", "nonfinite", "repeat"):
            bad = copy.deepcopy(runs)
            if defect == "missing":
                del bad[1]["native_erf"]
            elif defect == "extra":
                bad[1]["other"] = x.copy()
            elif defect == "nonfinite":
                bad[1]["native_erf"][0] = np.nan
            else:
                bad[1]["native_erf"][0] = 0
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                control.summarize(x, expected, bad)


if __name__ == "__main__":
    unittest.main()
