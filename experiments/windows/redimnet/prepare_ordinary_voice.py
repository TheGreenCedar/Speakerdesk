"""Freeze a bounded natural-reading pilot; performs no model inference.

Uses retained public source rows, then adjacent rows only if the retained range
has too few duration-eligible utterances. Never selects using embedding scores.
Requires the existing build environment's soundfile package.
"""
import argparse
import datetime
import hashlib
import io
import json
import pathlib
import shutil
import urllib.parse
import urllib.request
import wave

import soundfile as sf

REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
BUDGET = 20 * 1024 * 1024
MODEL = {
    "model_id": "PalabraAI/redimnet2",
    "revision": "2a8d15f65b1dfb5d73fede2f11ee42bcccca3035:b6-vb2+vox2_v0-lm:sd-pcm-v1:windowsml-2.4.89-directml-v1:ort-opt-disabled",
    "artifact_sha256": "89c532e48539bebddc168e773cdcbf014b1941043c50d0a4ff7ccd1139efa5f1",
    "dimension": 192,
    "sample_rate": 16000,
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=pathlib.Path, help="retained identity-pilot directory")
    parser.add_argument("prepared", type=pathlib.Path, help="staged voice-natural-v1 directory")
    parser.add_argument("output", type=pathlib.Path, help="new ordinary qualification directory")
    args = parser.parse_args()
    args.output.mkdir()  # Refuse an existing selection, including a partial prior attempt.
    audio = args.output / "audio"
    audio.mkdir()
    receipts = args.output / "source-rows"
    receipts.mkdir()
    downloaded = 0
    source_pins = {}

    def fetch(url, limit):
        nonlocal downloaded
        with urllib.request.urlopen(url, timeout=30) as response:
            result = bytearray()
            while True:
                remaining = BUDGET - downloaded
                if remaining <= 0:
                    raise RuntimeError("public download budget exhausted")
                chunk = response.read(min(65536, limit + 1 - len(result), remaining))
                if not chunk:
                    break
                result.extend(chunk)
                downloaded += len(chunk)
                if len(result) > limit:
                    raise RuntimeError("public download budget exceeded")
        return bytes(result)

    def rows(offset, length, name):
        url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode({
            "dataset": "openslr/librispeech_asr", "config": "clean", "split": "test",
            "offset": offset, "length": length,
        })
        data = fetch(url, 1024 * 1024)
        (receipts / name).write_bytes(data)
        source_pins[f"source-rows/{name}"] = sha(data)
        return json.loads(data)["rows"]

    cases, trials = [], []
    seen = set()

    def add(case, role):
        case = dict(case)
        assert case["id"] not in seen
        seen.add(case["id"])
        original = args.prepared / case["path"]
        data = original.read_bytes()
        assert sha(data) == case["wav_sha256"]
        relative = f"audio/{case['wav_sha256']}.wav"
        shutil.copyfile(original, args.output / relative)
        case.update(path=relative, role=role)
        cases.append(case)
        return case["id"]

    staged = {}
    for split in ["development", "holdout"]:
        name = f"{split}-fixed-crops.json"
        data = (args.prepared / name).read_bytes()
        source_pins[name] = sha(data)
        staged[split] = json.loads(data)
    selection = json.loads((args.source / "selection.json").read_bytes())
    assert selection["revision"] == REVISION
    known = list(selection["queries"])
    assert len(known) == 6
    excluded = {c["id"] for m in staged.values() for c in m["cases"]}
    for speaker in known:
        refs = [c for c in staged["development"]["cases"]
                if c["role"] == "reference" and str(c["speaker"]) == speaker]
        assert len(refs) == 2
        for case in refs:
            add(case, "reference")
        raw = (args.source / speaker / "rows.json").read_bytes()
        name = f"{speaker}-retained.json"
        (receipts / name).write_bytes(raw)
        source_pins[f"source-rows/{name}"] = sha(raw)
        entries = json.loads(raw)["rows"]
        query_row = selection["queries"][speaker]
        excluded.update(e["row"]["id"] for e in entries if e["row_idx"] == query_row)
        accepted = []
        position = 0
        extended = False
        while len(accepted) < 4:
            if position == len(entries):
                assert not extended, f"insufficient source rows for {speaker}"
                extended = True
                entries.extend(rows(max(e["row_idx"] for e in entries) + 1, 8,
                                    f"{speaker}-continuation.json"))
            entry = entries[position]
            position += 1
            row = entry["row"]
            if str(row["speaker_id"]) != speaker or row["id"] in excluded:
                continue
            url = row["audio"][0]["src"].split("?")[0]
            assert f"/--/{REVISION}/--/clean/test/{entry['row_idx']}/" in url
            local = args.source / speaker / (row["id"] + ".flac")
            if local.exists():
                flac = local.read_bytes()
            else:
                # A fresh row receipt renews the public asset URL without changing its identity.
                fresh = rows(entry["row_idx"], 1, f"row-{entry['row_idx']}.json")[0]
                assert fresh["row_idx"] == entry["row_idx"] and fresh["row"]["id"] == row["id"]
                renewed = fresh["row"]["audio"][0]["src"]
                assert renewed.split("?")[0] == url
                flac = fetch(renewed, 2 * 1024 * 1024)
            assert flac[:4] == b"fLaC" and int.from_bytes(flac[5:8], "big") == 34
            pcm, rate = sf.read(io.BytesIO(flac), dtype="int16")
            assert rate == 16000 and pcm.ndim == 1
            assert hashlib.md5(pcm.astype("<i2").tobytes()).digest() == flac[26:42]
            if len(pcm) < 32000:
                continue
            end = 96000 if len(pcm) > 160000 else len(pcm)
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(16000)
                wav.writeframes(pcm.astype("<i2").tobytes())
            data = buffer.getvalue()
            digest = sha(data)
            relative = f"audio/{digest}.wav"
            (args.output / relative).write_bytes(data)
            assert row["id"] not in seen
            seen.add(row["id"])
            cases.append(dict(id=row["id"], speaker=int(speaker), role="ordinary_query",
                              path=relative, start_sample=0, end_sample=end,
                              source_samples=len(pcm), words=row["text"],
                              row_offset=entry["row_idx"], flac_sha256=sha(flac),
                              wav_sha256=digest, source_url_without_expiring_query=url))
            accepted.append(row["id"])
        for pair in range(2):
            trials.append(dict(id=f"known-{speaker}-{pair}", speaker=int(speaker),
                               expected=speaker, cases=accepted[pair * 2:pair * 2 + 2]))
    unknown = {}
    for case in staged["holdout"]["cases"]:
        if case["role"] == "reference":
            speaker = str(case["speaker"])
            assert speaker not in known
            unknown.setdefault(speaker, []).append(add(case, "unknown_query"))
    assert len(unknown) == 4
    for speaker, ids in unknown.items():
        assert len(ids) == 2
        trials.append(dict(id=f"unknown-{speaker}", speaker=int(speaker), expected=None, cases=ids))
    assert len(cases) == 44 and len(trials) == 16
    manifest = dict(dataset="openslr/librispeech_asr", revision=REVISION,
                    frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    selection="Six existing development speakers; first four unused original utterances in retained dataset row order, then adjacent continuation if needed; exclude existing reference/query IDs and clips under 2 seconds; first 6 seconds only if over 10 seconds. Pair consecutive selected queries. Four disjoint holdout speakers contribute their existing reference pairs as unknown queries, never enrolled.",
                    source_pins=source_pins, downloaded_bytes=downloaded,
                    known_speakers=known, unknown_speakers=list(unknown), cases=cases, trials=trials)
    write_json(args.output / "ordinary-cases.json", manifest)
    policy = dict(model=MODEL, manifest_sha256=sha((args.output / "ordinary-cases.json").read_bytes()),
                  threshold=0.7264125328398843, margin=0.10003429964182987, minimum_clips=2,
                  genuine_trials=12, unknown_trials=4,
                  acceptance="All 12 known query pairs correctly accepted; all four unknown pairs rejected against only the six known profiles. No fitting, retries or reselection from results.",
                  limits="Small natural-reading pilot; not population accuracy or old-profile compatibility. Frozen before any Windows embedding/model run; production activation separate.")
    write_json(args.output / "ordinary-policy.json", policy)
    print(json.dumps({"manifest_sha256": policy["manifest_sha256"],
                      "policy_sha256": sha((args.output / "ordinary-policy.json").read_bytes()),
                      "downloaded_bytes": downloaded, "cases": len(cases), "trials": len(trials)}))


if __name__ == "__main__":
    main()
