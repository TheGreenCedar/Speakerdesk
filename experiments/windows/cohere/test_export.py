"""Small executable guards for checkpoint conversion and dynamic KV graph export."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnxruntime as ort
import torch
from transformers import CohereAsrConfig, CohereAsrForConditionalGeneration

SPEC = importlib.util.spec_from_file_location("cohere_export", Path(__file__).with_name("export.py"))
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)


class ConversionTests(unittest.TestCase):
    def test_affine_nibbles_preserve_noninteger_offsets_and_group_boundaries(self):
        packed = torch.tensor([[0x76543210] * 8 + [0xFEDCBA98] * 8], dtype=torch.uint32)
        scales = torch.tensor([[0.25, 0.5]], dtype=torch.bfloat16)
        biases = torch.tensor([[-0.125, -3.25]], dtype=torch.bfloat16)
        expected = [x * 0.25 - 0.125 for x in list(range(8)) * 8]
        expected += [x * 0.5 - 3.25 for x in list(range(8, 16)) * 8]
        np.testing.assert_array_equal(export.dequantize(packed, scales, biases).numpy(), [expected])

    @torch.inference_mode()
    def test_exported_decoder_cache_matches_full_sequence_at_two_continuation_lengths(self):
        torch.manual_seed(714)
        config = CohereAsrConfig(
            vocab_size=64, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
            intermediate_size=32, max_position_embeddings=64,
            encoder_config=dict(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                                intermediate_size=32, num_mel_bins=8,
                                subsampling_conv_channels=4, conv_kernel_size=3,
                                scale_input=False),
        )
        config._attn_implementation = "eager"
        config.encoder_config._attn_implementation = "eager"
        model = CohereAsrForConditionalGeneration(config).eval()
        decoder = export.Decoder(model)
        encoded = torch.randn(1, 5, 16)
        mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.int64)
        prefix = [4, 8, 9]
        initial_args = export.decoder_inputs(prefix, 0, encoded, mask)
        initial = decoder(*initial_args)
        example_args = export.decoder_inputs([11], 3, encoded, mask, initial[1:])
        names = ["tokens", "positions", "encoded", "self_mask", "cross_mask",
                 "past_self_k_0", "past_self_v_0", "past_cross_k_0", "past_cross_v_0"]
        outputs = ["logits", "present_self_k_0", "present_self_v_0",
                   "present_cross_k_0", "present_cross_v_0"]
        dynamic = {"tokens": {1: "tokens"}, "positions": {1: "tokens"},
                   "encoded": {1: "encoded_frames"},
                   "self_mask": {2: "tokens", 3: "total_tokens"},
                   "cross_mask": {2: "tokens", 3: "encoded_frames"}}
        for name in names[5:] + outputs[1:]:
            dynamic[name] = {2: "encoded_frames" if "cross" in name else
                             ("past_tokens" if name.startswith("past") else "total_tokens")}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            kept = export.export_graph(decoder, example_args, path / "decoder.onnx", names,
                                       outputs, dynamic, (path / "weights.bin", {}))
            session = ort.InferenceSession(str(path / "decoder.onnx"), providers=["CPUExecutionProvider"])
            for prefix in ([4, 8], [4, 8, 9]):
                initial = decoder(*export.decoder_inputs(prefix, 0, encoded, mask))
                for suffix in ([11], [11, 12]):
                    values = export.decoder_inputs(suffix, len(prefix), encoded, mask, initial[1:])
                    actual = session.run(None, {name: values[names.index(name)].numpy() for name in kept})
                    full = decoder(*export.decoder_inputs(prefix + suffix, 0, encoded, mask))
                    for result, expected in zip(actual, full):
                        np.testing.assert_allclose(result, expected.numpy(), rtol=2e-5, atol=2e-6)

            encoder = export.Encoder(model)
            example_features = torch.randn(1, 17, 8)
            example_mask = torch.tensor([[1] * 13 + [0] * 4], dtype=torch.int64)
            kept = export.export_graph(
                encoder, (example_features, example_mask), path / "encoder.onnx",
                ["features", "attention_mask"], ["encoded", "encoder_mask"],
                {"features": {1: "frames"}, "attention_mask": {1: "frames"},
                 "encoded": {1: "encoded_frames"}, "encoder_mask": {1: "encoded_frames"}},
                (path / "encoder.weights", {}))
            session = ort.InferenceSession(str(path / "encoder.onnx"), providers=["CPUExecutionProvider"])
            for frames, valid in [(17, 13), (25, 17)]:
                features = torch.randn(1, frames, 8)
                attention_mask = (torch.arange(frames)[None, :] < valid).to(torch.int64)
                expected = encoder(features, attention_mask)
                actual = session.run(None, dict(features=features.numpy(), attention_mask=attention_mask.numpy()))
                self.assertEqual(actual[1].sum(), (valid + 7) // 8)
                for value, reference in zip(actual, expected):
                    np.testing.assert_allclose(value, reference.numpy(), rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
