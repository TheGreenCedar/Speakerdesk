"""Small executable guards for ReDimNet waveform preparation and mixed-precision ONNX export."""
import argparse
import importlib.util
from pathlib import Path
import tempfile

import numpy as np
import onnxruntime as ort
import torch

SPEC = importlib.util.spec_from_file_location("redimnet_export", Path(__file__).with_name("export.py"))
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)


def check_preparation():
    clip = np.arange(40000, dtype=np.float32)
    repeated = export.prepared_audio(clip)
    np.testing.assert_array_equal(repeated[:40000], clip)
    np.testing.assert_array_equal(repeated[40000:80000], clip)
    np.testing.assert_array_equal(repeated[80000:], clip[:16000])
    for length in (96000, 96001, 160000):
        source = np.arange(length, dtype=np.float32)
        start = (length - 96000) // 2
        np.testing.assert_array_equal(export.prepared_audio(source), source[start:start + 96000])
    for invalid in (np.zeros(31999), np.zeros(160001), np.full(32000, np.nan)):
        try:
            export.prepared_audio(invalid)
        except AssertionError:
            continue
        raise AssertionError("invalid waveform was accepted")


@torch.inference_mode()
def check_graph(source):
    torch.set_num_threads(4)
    torch.manual_seed(608)
    model_type = export.source_class(source)
    model = model_type(F=8, C=4, causal="none", out_channels=8,
        return_2d_output=True, embed_dim=192, agg_gnorm=True,
        stages_setup=[((1, 1), 1, 1, [(3, 3)], None)], feat_type="tf",
        spec_params=dict(do_preemph=True, do_spec_aug=False, norm_signal=True)).eval()
    model = export.Embedding(model)
    example = torch.randn(1, 96000) * 0.2
    expected = model(example).numpy()
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        export.export_graph(model, example, directory)
        session = ort.InferenceSession(str(directory / "redimnet.onnx"), providers=["CPUExecutionProvider"])
        rows = []
        for gain in (1.0, 0.0001):
            audio = example.numpy() * gain
            eager = export.normalized(model(torch.from_numpy(audio)).numpy())
            actual = export.normalized(session.run(None, {"audio": audio})[0])
            cosine = float(np.dot(actual, eager))
            assert cosine >= export.IDENTITY_REUSE_COSINE, cosine
            assert float(np.dot(actual, export.normalized(expected))) >= export.IDENTITY_REUSE_COSINE
            rows.append(dict(gain=gain, onnx_eager_cosine=cosine))
    print(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    check_preparation()
    check_graph(args.source)
    print("Waveform boundary and mixed-precision portable graph checks passed.")
