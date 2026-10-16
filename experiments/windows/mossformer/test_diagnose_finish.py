import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from diagnose_finish import attribution, bind_inputs, mapped_float
from export import sha256
from qualify import waveform_error


class FinishDiagnostic(unittest.TestCase):
    def test_attribution_does_not_label_captured_state_as_an_independent_forward(self):
        independent = np.ones((1, 16), np.float32)
        changed = independent + np.float32(.01)
        zero = waveform_error(independent, independent)
        drift = waveform_error(changed, independent)
        self.assertEqual(attribution([drift] * 2, [zero] * 2, [drift] * 2),
                         "finish_path_sufficient_for_original_bound_failure")
        self.assertEqual(attribution([zero] * 2, [drift] * 2, [drift] * 2),
                         "captured_state_or_source_runtime_difference_remains")

    def test_captured_state_is_bound_to_original_audio_and_every_tensor_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for name, shape in {"audio": [1, 24], "encoded": [1, 512, 2],
                                "residual": [1, 512, 2], "x": [1, 2, 512]}.items():
                np.zeros(shape, "<f4").tofile(root / f"{name}.f32")
                rows.append({"name": name, "shape": shape, "dtype": "f32", "path": f"{name}.f32"})
            case = {"inputs": rows}
            path = root / "case.json"
            path.write_text(json.dumps(case))
            reference = {"cases": [{"samples": 24, "input": {"sha256": sha256(root / "audio.f32")}}]}
            self.assertEqual(bind_inputs(root, reference)[1]["x"].shape, (1, 2, 512))
            reference["cases"][0]["input"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "exactly one independent"):
                bind_inputs(root, reference)
            rows[-1]["shape"] = [1, 512, 2]
            path.write_text(json.dumps(case))
            with self.assertRaisesRegex(ValueError, "type/shape"):
                bind_inputs(root, reference)
            with self.assertRaisesRegex(ValueError, "identity differs"):
                mapped_float(root / "audio.f32", [1, 24], "0" * 64)


if __name__ == "__main__":
    unittest.main()
