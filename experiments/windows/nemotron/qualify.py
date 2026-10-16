"""Independent-reference and streaming qualification; Python is build tooling only."""
import argparse
import json
from pathlib import Path
import time
import wave

import numpy as np
import onnxruntime as ort

from export import ROOT, sha256


def read_pcm(path):
    with wave.open(str(path)) as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            raise ValueError("expected mono 16 kHz PCM16")
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768


def frame_windows(pcm, sample_offset, total_samples, start, count):
    positions = (start + np.arange(count))[:, None] * 160 + np.arange(512)[None] - 256
    local = positions - sample_offset

    def gather(indices):
        values = pcm[np.clip(indices, 0, len(pcm) - 1)]
        return np.where((indices >= 0) & (indices < len(pcm)), values, np.float32(0))

    frames = gather(local) - np.float32(.97) * gather(local - 1)
    frames = np.where((positions >= 0) & (positions < total_samples), frames, np.float32(0))
    return frames[None]


def features(pcm, sample_offset, total_samples, start, count, window, filterbank):
    frames = frame_windows(pcm, sample_offset, total_samples, start, count)[0]
    power = np.abs(np.fft.rfft(frames * np.pad(window, (56, 56)))) ** np.float32(2)
    # Apple Accelerate sets spurious FP flags for some finite SGEMM shapes in NumPy 2.2.6.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        mel = np.log(power.astype(np.float32) @ filterbank[0].T + np.float32(2**-24))
    if not np.isfinite(mel).all():
        raise ValueError("invalid Nemotron features")
    return np.where((start + np.arange(count))[:, None] < total_samples // 160, mel, np.float32(0))[None]


def compress(embeddings, predictions, silence):
    other = np.log(np.maximum(1 - predictions, np.float32(.25)))
    scores = np.log(np.maximum(predictions, np.float32(.25))) - other + other.sum(2, keepdims=True) - np.float32(np.log(.5))
    speech = predictions > .5
    scores = np.where(speech, scores, -np.inf)
    positive = scores > 0
    scores = np.where(~positive & speech & (positive.sum(1, keepdims=True) >= 16), -np.inf, scores)
    scores[:, 264:] += np.float32(.05)
    for count, scale in [(24, 2), (48, 1)]:
        for speaker in range(8):
            indices = np.argpartition(-scores[0, :, speaker], count - 1)[:count]
            finite = np.isfinite(scores[0, indices, speaker])
            scores[0, indices[finite], speaker] += np.float32(-scale * np.log(.5))
    frames = scores.shape[1]
    scores = np.concatenate([scores, np.full((1, 1, 8), np.inf, np.float32)], 1)
    flat = scores.transpose(0, 2, 1).reshape(-1)
    selected = np.argpartition(-flat, 263)[:264]
    selected = np.sort(np.where(flat[selected] > -np.inf, selected, 99999))
    indices = selected % (frames + 1)
    disabled = (selected == 99999) | (indices >= frames)
    indices = np.where(disabled, 0, indices)
    return (np.where(disabled[None, :, None], silence[None, None], embeddings[:, indices]),
            np.where(disabled[None, :, None], np.float32(0), predictions[:, indices]))


class Session:
    def __init__(self, run, artifact, preset, save=None, frontend=None):
        self.run, self.save, self.frontend = run, save, frontend
        self.central, self.right, self.limit, self.period = {
            "offline": (2720, 320, 40, 300), "low": (72, 32, 264, 222)}[preset]
        self.window, self.filterbank = np.load(artifact / "window.npy"), np.load(artifact / "fb.npy")
        self.silence = np.load(artifact / "learnable_sil_emb.npy")
        self.cache = np.zeros((1, 0, 512), np.float32)
        self.fifo = self.cache.copy()
        self.cache_predictions = np.zeros((1, 0, 8), np.float32)
        self.compressed = False
        self.pcm = np.zeros(0, np.float32)
        self.received = self.frames = self.sample_offset = self.steps = 0
        self.finished = False

    def feed(self, pcm, last=False):
        if self.finished or not np.isfinite(pcm).all():
            raise ValueError("invalid or finished Nemotron session")
        self.finished = True
        output = []
        for start in range(0, len(pcm), 16000):
            chunk = pcm[start:start + 16000]
            self.pcm = np.concatenate([self.pcm, chunk])
            self.received += len(chunk)
            output.extend(self.process(False))
        if last:
            output.extend(self.process(True))
        self.finished = last
        if last:
            self.pcm = np.zeros(0, np.float32)
        return np.asarray(output, np.float32).reshape(-1, 8)

    def process(self, last):
        output = []
        while True:
            available = self.received // 160 - self.frames
            needed = (self.frames + self.central + self.right - 1) * 160 + 256
            if not available or (not last and self.received < needed):
                return output
            central = min(available, self.central)
            count = min(((self.received // 160 + 1 + 15) // 16) * 16 - self.frames,
                        self.central + self.right) if last else self.central + self.right
            mel = features(self.pcm, self.sample_offset, self.received, self.frames, count,
                           self.window, self.filterbank)
            if self.frontend:
                mel = self.frontend({"audio_windows": frame_windows(self.pcm, self.sample_offset, self.received,
                                                                    self.frames, count),
                                     "valid_frames": np.array([min(available, count)], np.int64)})
            inputs = {"features": mel, "cache": self.cache, "fifo": self.fifo,
                      "valid_frames": np.array([min(available, count)], np.int64)}
            chunk, high, low = self.run(inputs)
            if self.save:
                self.save(self.steps, inputs, {"chunk": chunk, "high": high, "low": low})
            if not np.isfinite(high).all() or np.any((high < 0) | (high > 1)):
                raise ValueError("invalid Nemotron probabilities")
            prefix = self.cache.shape[1] + self.fifo.shape[1]
            output.extend(high[0, prefix * 8:prefix * 8 + central])
            n = (central + 7) // 8
            cache_len = self.cache.shape[1]
            self.fifo = np.concatenate([self.fifo, chunk[:, :n]], 1)
            fifo_predictions = np.concatenate([low[:, cache_len:prefix], low[:, prefix:prefix + n]], 1)
            if self.fifo.shape[1] > self.limit:
                pop = min(self.fifo.shape[1], max(self.period, self.fifo.shape[1] - self.limit))
                self.cache = np.concatenate([self.cache, self.fifo[:, :pop]], 1)
                previous = self.cache_predictions if self.compressed else low[:, :cache_len]
                self.cache_predictions = np.concatenate([previous, fifo_predictions[:, :pop]], 1)
                self.fifo = self.fifo[:, pop:]
                if self.cache.shape[1] > 264:
                    self.cache, self.cache_predictions = compress(self.cache, self.cache_predictions, self.silence)
                    self.compressed = True
            self.frames += central
            self.steps += 1
            keep = max(0, self.frames * 160 - 257)
            self.pcm = self.pcm[keep - self.sample_offset:]
            self.sample_offset = keep


def compare(actual, expected):
    expected = np.asarray(expected, np.float32)
    if actual.shape != expected.shape:
        raise ValueError(("frame count", actual.shape, expected.shape))
    return {"frames": len(actual), "max_probability_error": float(np.max(np.abs(actual - expected), initial=0)),
            "mean_probability_error": float(np.mean(np.abs(actual - expected))) if actual.size else 0.,
            "changed_activity_decisions": int(np.count_nonzero((actual > .5) != (expected > .5)))}


def qualify(artifact, low_reference, output):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(artifact / "nemotron.onnx"), options, providers=["CPUExecutionProvider"])
    run = lambda inputs: session.run(["chunk", "high", "low"], inputs)
    frontend_session = ort.InferenceSession(str(artifact / "frontend.onnx"), options, providers=["CPUExecutionProvider"])
    frontend = lambda inputs: frontend_session.run(["features"], inputs)[0]
    wav = ROOT / "crates/models/tests/data/two-voice.wav"
    pcm = read_pcm(wav)
    references = [("offline", ROOT / "crates/models/tests/data/nemotron-reference.json")]
    if low_reference:
        references.append(("low", low_reference))
    reports = []
    for preset, path in references:
        reference = json.loads(path.read_text())
        if sha256(wav) != reference["wav_sha256"] or reference["source_revision"] != "feb25a37b07923bae556e59111995071d66afa0d":
            raise ValueError("independent reference identity mismatch")
        started = time.perf_counter()
        one = Session(run, artifact, preset)
        actual = one.feed(pcm, True)
        report = {"preset": preset, "reference_sha256": sha256(path), "seconds": time.perf_counter() - started,
                  **compare(actual, reference["probabilities"])}
        split = Session(run, artifact, preset)
        chunks, cursor, index = [], 0, 0
        sizes = [1, 159, 160, 511, 512, 16000, 17]
        while cursor < len(pcm):
            end = min(len(pcm), cursor + sizes[index % len(sizes)])
            chunks.append(split.feed(pcm[cursor:end], False))
            cursor, index = end, index + 1
        chunks.append(split.feed(np.zeros(0, np.float32), True))
        report["packetization"] = compare(np.concatenate(chunks), actual)
        report["reset"] = compare(Session(run, artifact, preset).feed(pcm, True), actual)
        complete = Session(run, artifact, preset, frontend=frontend).feed(pcm, True)
        report["onnx_frontend"] = compare(complete, reference["probabilities"])
        for comparison in [report, report["onnx_frontend"]]:
            if comparison["max_probability_error"] > .025 or comparison["changed_activity_decisions"]:
                raise ValueError(("independent reference changed", comparison))
        for comparison in [report["packetization"], report["reset"]]:
            if comparison["max_probability_error"] != 0:
                raise ValueError(("session state changed", comparison))
        reports.append(report)
        print(json.dumps(report), flush=True)
    result = {"model_sha256": sha256(artifact / "nemotron.onnx"), "runtime": ort.__version__,
              "provider": "CPUExecutionProvider", "cases": reports}
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--low-reference", type=Path)
    args = parser.parse_args()
    qualify(args.artifact, args.low_reference, args.output)
