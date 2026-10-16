"""Small graph-preservation, hook-observation and error-reporting controls."""
from collections import Counter
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper

from finish_taps import TAPS, add_outputs, capture_hooks, tensor_error


class FinishTaps(unittest.TestCase):
    def test_outputs_preserve_every_source_operation_and_external_tensor_descriptor(self):
        nodes = [helper.make_node("Identity", ["input"], ["source0"], name="source0"),
                 helper.make_node("Identity", ["input"], ["source1"], name="source1")]
        nodes.extend(helper.make_node("Identity", ["input"], ["boundary" + str(index)], name=name)
                     for index, name in enumerate(TAPS.values()))
        tensor = onnx.TensorProto(name="external", data_type=TensorProto.FLOAT, dims=[1],
                                  data_location=TensorProto.EXTERNAL)
        for key, value in [("location", "mossformer.weights"), ("offset", "4096"), ("length", "4")]:
            tensor.external_data.add(key=key, value=value)
        model = helper.make_model(helper.make_graph(nodes, "fixture", [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, None)], [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, None) for name in ["source0", "source1"]],
            [tensor]))
        before = [node.SerializeToString() for node in model.graph.node]
        taps = add_outputs(model, 31)
        self.assertEqual([node.SerializeToString() for node in model.graph.node], before)
        self.assertEqual(model.graph.initializer[0].SerializeToString(), tensor.SerializeToString())
        self.assertEqual([row["name"] for row in taps], list(TAPS))
        self.assertEqual(taps[-1]["shape"], [1, 24])  # Seven padded samples stay outside the decoder tap.
        self.assertEqual([value.name for value in model.graph.output[:2]], ["source0", "source1"])
        with self.assertRaisesRegex(ValueError, "two-source finish"):
            add_outputs(model, 31)

    def test_hooks_capture_both_decoder_calls_without_changing_finish_outputs(self):
        import torch
        from torch import nn

        class SmallFinish(nn.Module):
            def __init__(self):
                super().__init__()
                for name in ["intra_norm", "global_norm", "projection", "mask_projection", "decoder"]:
                    setattr(self, name, nn.Linear(2, 2))

            def forward(self, x):
                x = self.mask_projection(self.projection(self.global_norm(self.intra_norm(x))))
                return self.decoder(x + 1), self.decoder(x - 1)

        torch.manual_seed(17)
        finish = SmallFinish().eval()
        data = torch.tensor([[.2, -.3]])
        with torch.inference_mode():
            original = finish(data)
            masked = finish.mask_projection(finish.projection(finish.global_norm(finish.intra_norm(data))))
            captured = {}
            handles, calls = capture_hooks(finish, lambda name, value: captured.setdefault(name, value.clone()))
            # The production save callback returns None; hooks must ignore any callback return value.
            actual = finish(data)
            for handle in handles:
                handle.remove()
            for got, wanted in zip(actual, original):
                torch.testing.assert_close(got, wanted, rtol=0, atol=0)
            torch.testing.assert_close(captured["decoder_input0"], masked + 1, rtol=0, atol=0)
            torch.testing.assert_close(captured["decoder_input1"], masked - 1, rtol=0, atol=0)
            for index in range(2):
                torch.testing.assert_close(captured[f"decoder_output{index}"], original[index], rtol=0, atol=0)
        self.assertEqual(set(captured), set(TAPS))
        self.assertEqual(calls, Counter(intra_norm=1, global_norm=1, projection=1,
            mask_projection=1, decoder_input=2, decoder_output=2))

    def test_error_reports_full_tensor_difference_without_assigning_a_new_gate(self):
        expected = np.array([1., 2., -3., 4.], np.float32)
        actual = expected + np.array([.125, 0, -.25, 0], np.float32)
        report = tensor_error(actual, expected)
        self.assertEqual(report["max_abs_error"], .25)
        self.assertAlmostEqual(report["normalized_rms_error"], np.sqrt((.125**2 + .25**2) / 30))
        self.assertNotIn("passed", report)
        for invalid in [np.ones(3, np.float32), np.array([1., 2., np.nan, 4.])]:
            with self.assertRaises(ValueError):
                tensor_error(invalid, expected)


if __name__ == "__main__":
    unittest.main()
