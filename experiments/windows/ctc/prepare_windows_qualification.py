"""Seal the prospective CTC corpus without executing either acoustic model."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import unicodedata
import wave


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_wav(path, expected):
    data = path.read_bytes()
    if digest(data) != expected:
        raise ValueError(f"waveform identity differs: {path}")
    with wave.open(str(path)) as wav:
        if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (1, 2, 16000):
            raise ValueError(f"expected mono signed16 16kHz PCM: {path}")
        raw = wav.readframes(wav.getnframes())
    return [sample[0] / 32768 for sample in struct.iter_unpack("<h", raw)]


def columns(text, vocabulary, initial=()):
    # This English/French corpus needs only NFC, punctuation removal and simple
    # case aliases; the real core independently validates all requested columns.
    result = list(initial) or [0]
    for character in unicodedata.normalize("NFC", text):
        if unicodedata.category(character).startswith("P"):
            continue
        character = " " if character.isspace() else character
        if character not in vocabulary:
            character = character.lower()
        if character not in vocabulary:
            raise ValueError(f"unsupported qualification character: {character!r}")
        token = vocabulary[character]
        if token not in result:
            result.append(token)
    return result


def prepare(ordinary, fleurs, tokens, output):
    root = Path(__file__).resolve().parents[3]
    protocol_bytes = Path(__file__).with_name("windows-qualification-v1.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    source = ordinary / "ordinary-cases.json"
    if digest(source.read_bytes()) != protocol["selection"]["ordinary_manifest_sha256"]:
        raise ValueError("ordinary source manifest differs")
    voices = json.loads(source.read_bytes())
    pilot = json.loads((fleurs / "manifest.json").read_bytes())
    if pilot["selection"]["revision"] != protocol["selection"]["fleurs_revision"]:
        raise ValueError("FLEURS revision differs")
    token_bytes = tokens.read_bytes()
    if digest(token_bytes) != protocol["tokens_sha256"]:
        raise ValueError("vocabulary differs")
    vocabulary = {t: int(i) for t, i in (line.rsplit(" ", 1) for line in token_bytes.decode().splitlines())}
    reference_path = root / "crates/models/tests/data/ctc-reference.json"
    reference_bytes = reference_path.read_bytes()
    if digest(reference_bytes) != protocol["original_reference_sha256"]:
        raise ValueError("original reference differs")
    reference = json.loads(reference_bytes)
    waveform = root / "crates/models/tests/data/two-voice.wav"
    pcm = read_wav(waveform, reference["wav_sha256"])
    transcript = json.loads((waveform.parent / "cohere-reference.json").read_bytes())["text"]
    cases = []
    audio = {}

    def add(name, samples, text, language, provenance, initial=()):
        if not 400 <= len(samples) <= 392000:
            raise ValueError(f"invalid complete crop: {name}")
        raw = struct.pack(f"<{len(samples)}f", *samples)
        audio[name] = raw
        cases.append(dict(id=name, path=f"{name}.f32", samples=len(samples), pcm_sha256=digest(raw),
                          text=text, text_sha256=digest(text.encode()), language=language,
                          token_ids=columns(text, vocabulary, initial), source=provenance))

    for name, samples in [("whole", pcm), ("400", pcm[16000:16400]),
                          ("719", pcm[16000:16719]), ("720", pcm[16000:16720]),
                          ("partial_tail", pcm[16000:16549]), ("silence", [0.] * 720)]:
        add(f"original-{name}", samples, transcript if name == "whole" else "a", "en",
            {"original_case": name, "wav_sha256": reference["wav_sha256"]}, reference["token_ids"])
    for speaker in protocol["selection"]["known_speakers"]:
        selected = None
        for case in voices["cases"]:
            if str(case["speaker"]) != speaker or case["start_sample"] != 0 or case["end_sample"] != case["source_samples"]:
                continue
            samples = read_wav(ordinary / case["path"], case["wav_sha256"])
            if len(samples) != case["source_samples"]:
                raise ValueError("retained full utterance is truncated")
            if 400 <= len(samples) <= 392000:
                selected = (case, samples)
                break
        if selected is None:
            raise ValueError(f"no complete retained utterance for {speaker}")
        case, samples = selected
        add(f"libri-{case['id']}", samples, case["words"], "en",
            {k: case[k] for k in ["id", "speaker", "wav_sha256", "source_samples", "row_offset"]})
    english = None
    for name in protocol["selection"]["fleurs"]:
        case = next(c for c in pilot["cases"] if c["id"] == name)
        samples = read_wav(fleurs / Path(case["path"]).name, case["wav_sha256"])
        if len(samples) != case["samples"]:
            raise ValueError("FLEURS complete utterance is truncated")
        add(name, samples, case["words"], case["language"],
            {"id": name, "wav_sha256": case["wav_sha256"]})
        if name == "en_us-0":
            english = (case, samples)
    case, samples = english
    add("en-quiet", [x / 256 for x in samples], case["words"], "en", {"parent": "en_us-0", "gain": 1 / 256})
    add("en-clipped", [max(-1, min(1, x * 4)) for x in samples], case["words"], "en", {"parent": "en_us-0", "gain": 4, "clip": [-1, 1]})
    assert len(cases) == protocol["selection"]["total_cases"]
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in audio.items():
        (output / f"{name}.f32").write_bytes(raw)
    (output / "tokens.txt").write_bytes(token_bytes)
    manifest = dict(protocol_sha256=digest(protocol_bytes), preparation_sha256=digest(Path(__file__).read_bytes()),
                    tokens_sha256=digest(token_bytes), cases=cases)
    data = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()
    (output / "cases.json").write_bytes(data)
    print(json.dumps({"cases": len(cases), "manifest_sha256": digest(data), "pcm_bytes": sum(map(len, audio.values()))}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["ordinary", "fleurs", "tokens", "output"]:
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    prepare(args.ordinary, args.fleurs, args.tokens, args.output)
