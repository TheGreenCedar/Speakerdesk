"""Independent byte-order, affine-offset and native-result rejection controls."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("quantized_probe", Path(__file__).with_name("quantized_probe.py"))
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class QuantizedProbeTests(unittest.TestCase):
    def test_low_nibble_first_affine_groups_retain_noninteger_zero_points(self):
        # Same packing as two original MLX U32 words, repeated to fill each group.
        packed = struct.pack("<8I", *([0x76543210] * 8)) + struct.pack("<8I", *([0xFEDCBA98] * 8))
        expected = [x * 0.25 - 0.125 for x in list(range(8)) * 8]
        expected += [x * -0.5 - 3.25 for x in list(range(8, 16)) * 8]
        self.assertEqual(probe.reconstruct(packed, [0.25, -0.5], [-0.125, -3.25], 1, 128), expected)
        for invalid in ((packed[:-1], [0.25, -0.5], [-0.125, -3.25], 1, 128),
                        (packed, [0.25], [-0.125, -3.25], 1, 128),
                        (packed, [0.25, -0.5], [-0.125, -3.25], 1, 127)):
            with self.assertRaises(ValueError):
                probe.reconstruct(*invalid)

    def test_native_results_reject_wrong_reconstruction_drift_repeats_and_missing_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case, result = root / "case", root / "result"
            case.mkdir()
            result.mkdir()
            references = [dict(name="weights", shape=[1, 2], file="weights.f32", exact=True),
                          dict(name="product", shape=[1, 2], file="product.f32", exact=False)]
            (case / "weights.f32").write_bytes(struct.pack("<2f", -0.125, 3.625))
            (case / "product.f32").write_bytes(struct.pack("<2f", 2, -1))
            probe.write_json(case / "reference.json", dict(outputs=references, atol=0.002, rtol=0.002))
            outputs = []
            for run in range(2):
                for reference in references:
                    path = result / f"{run}-{reference['file']}"
                    path.write_bytes((case / reference["file"]).read_bytes())
                    outputs.append(dict(run=run, name=reference["name"], shape=reference["shape"],
                                        dtype="f32", file=path.name, sha256=probe.verifier.sha256(path)))
            record = dict(outputs=outputs)
            self.assertEqual(probe.compare_outputs(case, result, record), dict(weights=0, product=0))

            for position, values, reason in [
                (0, [0, 3.625], "reconstructed Float32 bits"),
                (1, [2.1, -1], "matrix product exceeds"),
                (1, [float("nan"), -1], "nonfinite"),
                (3, [2.001, -1], "repeats differ"),
            ]:
                changed = copy.deepcopy(record)
                output = changed["outputs"][position]
                path = result / output["file"]
                previous = path.read_bytes()
                path.write_bytes(struct.pack("<2f", *values))
                output["sha256"] = probe.verifier.sha256(path)
                with self.assertRaisesRegex(ValueError, reason):
                    probe.compare_outputs(case, result, changed)
                path.write_bytes(previous)
            for changed in (dict(outputs=outputs[:-1]), dict(outputs=outputs[:-1] + [outputs[0]])):
                with self.assertRaises(ValueError):
                    probe.compare_outputs(case, result, changed)
            changed = copy.deepcopy(record)
            changed["outputs"][0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "hash differs"):
                probe.compare_outputs(case, result, changed)


if __name__ == "__main__":
    unittest.main()
