"""Compare all pinned Whisper language scores and prepare native ORT graph cases."""
import argparse
import json
from pathlib import Path
import time
import wave

import numpy as np
import onnxruntime as ort

from export import ROOT, frontend_buffers, sha256


def frame_audio(pcm):
    pcm = np.asarray(pcm, np.float32)
    if pcm.ndim != 1 or not len(pcm) or not np.isfinite(pcm).all():
        raise ValueError("Whisper requires finite nonempty 16 kHz mono PCM")
    audio = np.zeros(480000, np.float32)
    audio[:min(len(pcm), len(audio))] = pcm[:len(audio)]
    centered = np.pad(audio, (200, 200), mode="reflect")
    return np.lib.stride_tricks.sliding_window_view(centered, 400)[::160][:3000].copy()[None]


def numpy_mel(windows):
    window, filters = frontend_buffers()
    power = np.abs(np.fft.rfft(windows * window)) ** np.float32(2)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        mel = power.astype(np.float32) @ filters
    log = np.log10(np.maximum(mel, np.float32(1e-10)))
    result = (np.maximum(log, log.max() - np.float32(8)) + np.float32(4)) / np.float32(4)
    if not np.isfinite(result).all():
        raise ValueError("invalid Whisper features")
    return result.astype(np.float16).astype(np.float32)


def route(probabilities, codes):
    order = np.argsort(-np.asarray(probabilities), kind="stable")
    best, second = float(probabilities[order[0]]), float(probabilities[order[1]])
    return {"code": codes[order[0]], "probability": float(best), "margin": float(best - second),
            "probability_threshold_distance": float(best) - .90,
            "margin_threshold_distance": float(best - second) - .20,
            "confident": bool(best >= .90 and best - second >= .20)}


def score_report(actual, expected, codes):
    actual, expected = np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    if actual.shape != (99,) or expected.shape != actual.shape or not np.isfinite(actual).all():
        raise ValueError("invalid 99-language probability vector")
    if np.any((actual < 0) | (actual > 1)) or abs(float(actual.sum()) - 1) > 1e-5:
        raise ValueError("invalid language probability normalization")
    return {"max_probability_error": float(np.max(np.abs(actual - expected))),
            "actual_route": route(actual, codes), "expected_route": route(expected, codes)}


def save_case(artifact, name, model, input_name, inputs, output_name, outputs, independent=None):
    directory = artifact / "cases" / name
    directory.mkdir(parents=True, exist_ok=True)
    inputs.astype("<f4").tofile(directory / "input.f32")
    outputs.astype("<f4").tofile(directory / "expected.f32")
    case = {"model": "../../" + model, "repeat": 2, "inputs": [{
        "name": input_name, "dtype": "f32", "shape": list(inputs.shape), "path": "input.f32"}]}
    (directory / "case.json").write_text(json.dumps(case, indent=2) + "\n")
    (directory / "expected.json").write_text(json.dumps({
        "name": output_name, "shape": list(outputs.shape), "file": "expected.f32",
        "sha256": sha256(directory / "expected.f32"), "independent": independent}, indent=2) + "\n")
    return str(directory.relative_to(artifact) / "case.json")


def qualify(artifact, output):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    network = ort.InferenceSession(str(artifact / "whisper.onnx"), options, providers=["CPUExecutionProvider"])
    frontend = ort.InferenceSession(str(artifact / "frontend.onnx"), options, providers=["CPUExecutionProvider"])
    codes = json.loads((artifact / "languages.json").read_text())["codes"]
    reference_path = ROOT / "crates/models/tests/data/whisper-reference.json"
    reference = json.loads(reference_path.read_text())
    wav = ROOT / "crates/models/tests/data/two-voice.wav"
    if sha256(wav) != reference["wav_sha256"] or reference["source_revision"] != "feb25a37b07923bae556e59111995071d66afa0d":
        raise ValueError("independent reference identity mismatch")
    with wave.open(str(wav)) as reader:
        if (reader.getnchannels(), reader.getsampwidth(), reader.getframerate()) != (1, 2, 16000):
            raise ValueError("expected mono 16 kHz PCM16")
        pcm = np.frombuffer(reader.readframes(reader.getnframes()), "<i2").astype(np.float32) / 32768
    reports, cases = [], []
    audio_cases = [pcm, pcm[:48000], pcm[16000:16549], np.zeros(512, np.float32)]
    for audio, expected in zip(audio_cases, reference["cases"]):
        if set(expected["probabilities"]) != set(codes) or len(audio) != expected["samples"]:
            raise ValueError("reference languages or sample count mismatch")
        started = time.perf_counter()
        windows = frame_audio(audio)
        mel = numpy_mel(windows)
        onnx_mel = frontend.run(["mel"], {"audio_windows": windows})[0]
        wanted = [expected["probabilities"][code] for code in codes]
        scores = network.run(["probabilities"], {"mel": mel})[0]
        complete = network.run(["probabilities"], {"mel": onnx_mel})[0]
        repeated = network.run(["probabilities"], {"mel": mel})[0]
        mel_error = max(abs(float(mel[0, point["frame"], point["band"]]) - point["value"])
                        for point in expected["mel"])
        onnx_mel_error = max(abs(float(onnx_mel[0, point["frame"], point["band"]]) - point["value"])
                             for point in expected["mel"])
        report = {"name": expected["name"], "samples": len(audio), "seconds": time.perf_counter() - started,
                  "numpy_frontend": score_report(scores, wanted, codes), "onnx_frontend": score_report(complete, wanted, codes),
                  "numpy_sampled_mel_error": mel_error, "onnx_sampled_mel_error": onnx_mel_error,
                  "repeat_error": float(np.max(np.abs(repeated - scores)))}
        reports.append(report)
        print(json.dumps(report), flush=True)
        cases.append(save_case(artifact, "neural-" + expected["name"], "whisper.onnx", "mel", onnx_mel,
                               "probabilities", complete, {"codes": codes, "probabilities": wanted,
                               "reference_sha256": sha256(reference_path), "case": expected["name"]}))
        cases.append(save_case(artifact, "frontend-" + expected["name"], "frontend.onnx", "audio_windows", windows,
                               "mel", onnx_mel, {"sampled_mel": expected["mel"],
                               "reference_sha256": sha256(reference_path), "case": expected["name"]}))
    result = {"runtime": ort.__version__, "provider": "CPUExecutionProvider", "cases": reports,
              "reference_sha256": sha256(reference_path), "wav_sha256": sha256(wav),
              "models": {name: sha256(artifact / name) for name in ["whisper.onnx", "frontend.onnx"]}}
    output.write_text(json.dumps(result, indent=2) + "\n")
    (artifact / "cases.json").write_text(json.dumps({"cases": cases, "models": result["models"]}, indent=2) + "\n")
    for report in reports:
        comparison = report["onnx_frontend"]
        if comparison["max_probability_error"] >= 1e-4 or any(
                comparison["actual_route"][key] != comparison["expected_route"][key] for key in ["code", "confident"]):
            raise ValueError(("independent scores changed", report["name"], comparison))
        if report["repeat_error"] != 0 or max(report["numpy_sampled_mel_error"], report["onnx_sampled_mel_error"]) >= 1e-4:
            raise ValueError(("frontend or repeat changed", report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    qualify(args.artifact, args.output)
