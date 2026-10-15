"""Reject incomplete, substituted or non-repeating native normalization evidence."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

import norm_control as control


class NormControl(unittest.TestCase):
    def test_native_outputs_require_hashes_shapes_complete_branches_and_repetition(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rows = []
            for index in (0, 1):
                for name, shape in control.OUTPUT_SHAPES.items():
                    data = np.zeros(shape, "<f4").tobytes()
                    filename = f"{index}-{name}.f32"
                    (directory / filename).write_bytes(data)
                    rows.append(dict(run=index, name=name, dtype="f32", shape=shape,
                                     file=filename, sha256=control.digest(data)))
            record = dict(outputs=rows)
            self.assertEqual(len(control.read_outputs(record, directory)), 2)
            for defect in ("missing", "duplicate", "hash", "shape", "repeat", "nonfinite"):
                bad = copy.deepcopy(record)
                row = bad["outputs"][-1]
                if defect == "missing":
                    bad["outputs"].pop()
                elif defect == "duplicate":
                    bad["outputs"].append(copy.deepcopy(row))
                elif defect == "hash":
                    row["sha256"] = "00" * 32
                elif defect == "shape":
                    row["shape"] = [512]
                else:
                    values = np.zeros(row["shape"], "<f4")
                    values.flat[0] = np.nan if defect == "nonfinite" else 1
                    data = values.tobytes()
                    row["file"] = "changed.f32"
                    (directory / row["file"]).write_bytes(data)
                    row["sha256"] = control.digest(data)
                with self.subTest(defect=defect), self.assertRaises(ValueError):
                    control.read_outputs(bad, directory)


if __name__ == "__main__":
    unittest.main()
