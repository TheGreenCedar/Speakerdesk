import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from export import externalize
from qualify import waveform_error


class Contract(unittest.TestCase):
    def test_original_absolute_relative_shape_and_channel_bounds(self):
        for expected, actual, accepted in [
                ([1., -1.], [1., -1.], True),
                ([100., 100.], [100.0021, 100.0021], False),
                ([.01, .01], [.01003, .01003], False),
                ([1., -1.], [-1., 1.], False),
                ([0., 0.], [0., 0.], True)]:
            self.assertEqual(waveform_error(np.array(actual), np.array(expected))["within_original_bounds"], accepted)
        for actual in [np.zeros(3), np.array([np.nan, 1.]), np.array([np.inf, 1.])]:
            with self.assertRaisesRegex(ValueError, "shape or values"):
                waveform_error(actual, np.zeros(2))

    @unittest.skipUnless(importlib.util.find_spec("onnx"), "requires existing ONNX preparation environment")
    def test_two_graphs_share_one_external_tensor_without_losing_bytes(self):
        import onnx
        from onnx import TensorProto, helper, numpy_helper
        values = np.arange(2048, dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary:
            root, offsets = Path(temporary), {}
            with (root / "mossformer.weights").open("wb") as weights:
                for index in range(2):
                    tensor = numpy_helper.from_array(values, "weights")
                    graph = helper.make_graph([helper.make_node("Identity", ["weights"], ["output"])],
                                              "external", [], [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2048])], [tensor])
                    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
                    externalize(model, weights, offsets)
                    weights.flush()
                    path = root / f"graph-{index}.onnx"
                    onnx.save(model, path)
                    onnx.checker.check_model(str(path), full_check=True)
                    restored = onnx.load(path)
                    np.testing.assert_array_equal(numpy_helper.to_array(restored.graph.initializer[0]), values)
            self.assertEqual((root / "mossformer.weights").stat().st_size, values.nbytes)
            self.assertEqual(len(offsets), 1)


if __name__ == "__main__":
    unittest.main()
