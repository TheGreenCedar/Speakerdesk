import os
from pathlib import Path
import unittest
import warnings


@unittest.skipUnless(os.environ.get("SPEAKERDESK_MOSSFORMER_SOURCE_PARENT"), "requires retained pinned source and cached Torch")
class Stages(unittest.TestCase):
    def test_flash_export_preserves_keyword_only_none_mask(self):
        import torch
        from export import load_source
        from stages import Flash, adapt_for_export
        _, block = load_source(Path(os.environ["SPEAKERDESK_MOSSFORMER_SOURCE_PARENT"]))
        torch.manual_seed(17)
        raw = block.FLASH_ShareA_FFConvM(dim=8, query_key_dim=8, expansion_factor=4).eval()
        adapt_for_export(raw, block)
        x = torch.linspace(-.1, .1, 16).reshape(1, 2, 8)
        # Reproduce the actual legacy exporter argument expansion without a model export.
        arguments = torch.onnx.utils._decide_input_format(raw, (x,))
        with self.assertRaisesRegex(TypeError, "positional arguments"):
            raw(*arguments)
        wrapped = Flash(raw)
        arguments = torch.onnx.utils._decide_input_format(wrapped, (x,))
        with torch.inference_mode():
            torch.testing.assert_close(wrapped(*arguments), raw(x, mask=None), rtol=0, atol=0)

    def test_stage_composition_matches_maintained_whole_forward_at_group_and_tail_edges(self):
        import torch
        from export import load_source
        from stages import adapt_for_export, program
        source, block = load_source(Path(os.environ["SPEAKERDESK_MOSSFORMER_SOURCE_PARENT"]))
        torch.manual_seed(17)
        torch.set_num_threads(1)
        model = source.MossFormer(num_blocks=1).eval()
        adapt_for_export(model, block)
        steps = program(model)
        with torch.inference_mode():
            for samples in [24, 25, 31, 2056, 2063, 2064]:
                audio = torch.linspace(-.1, .1, samples)[None]
                expected = model(audio)
                encoded, residual, x = steps[0][1](audio)
                for _, stage in steps[1:-1]:
                    x = stage(x)
                actual = steps[-1][1](audio, encoded, residual, x)
                for got, wanted in zip(actual, expected):
                    self.assertEqual(got.shape, (1, samples))
                    torch.testing.assert_close(got, wanted, rtol=0, atol=0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
                traced_flash = torch.jit.trace(steps[1][1], torch.zeros(1, 257, 512), check_trace=False)
                traced_fsmn = torch.jit.trace(steps[2][1], torch.zeros(1, 257, 512), check_trace=False)
            for frames in [2, 255, 256, 257, 258]:
                x = torch.linspace(-.1, .1, frames * 512).reshape(1, frames, 512)
                for stage, traced in [(steps[1][1], traced_flash), (steps[2][1], traced_fsmn)]:
                    torch.testing.assert_close(traced(x), stage(x), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
