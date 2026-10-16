"""Select complete pinned FLEURS utterances with bounded stdlib archive streaming."""
import hashlib
import json
from pathlib import PurePosixPath
import re
import tarfile


def selected_audio(config, revision, downloads, metadata_limit, audio_limit):
    if not re.fullmatch(r"[a-z_0-9]+", config) or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("invalid pinned dataset path")
    base = f"https://huggingface.co/datasets/google/fleurs/resolve/{revision}/data/{config}/"
    labels_blob = downloads.get(base + "dev.tsv", metadata_limit)
    labels = {}
    for line in labels_blob.decode("utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) != 7 or fields[1] in labels:
            raise ValueError("pinned label schema/filename differs")
        labels[fields[1]] = fields
    tree_url = f"https://huggingface.co/api/datasets/google/fleurs/tree/{revision}/data/{config}/audio?recursive=false&expand=false&limit=1000"
    tree = downloads.get(tree_url, metadata_limit)
    expected_path = f"data/{config}/audio/dev.tar.gz"
    entries = [item for item in json.loads(tree) if item["path"] == expected_path]
    if len(entries) != 1 or entries[0]["size"] != entries[0]["lfs"]["size"]:
        raise ValueError("pinned archive inventory differs")
    pin = entries[0]
    if not re.fullmatch(r"[a-f0-9]{64}", pin["lfs"]["oid"]):
        raise ValueError("invalid pinned archive identity")
    rows = []
    before = downloads.bytes
    with downloads.stream(base + "audio/dev.tar.gz") as response:
        with tarfile.open(fileobj=response, mode="r|gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".wav"):
                    continue
                filename = PurePosixPath(member.name).name
                if not re.fullmatch(r"[0-9]+\.wav", filename) or filename not in labels:
                    raise ValueError("archive audio is missing from pinned labels")
                if not 0 < member.size <= audio_limit:
                    raise ValueError("complete selected audio exceeds frozen per-asset limit")
                data = archive.extractfile(member).read(member.size + 1)
                if len(data) != member.size:
                    raise ValueError("truncated selected archive member")
                fields = labels[filename]
                rows.append({"member": member.name, "filename": filename, "id": int(fields[0]),
                             "num_samples": int(fields[5]), "raw_transcription": fields[2], "data": data})
                if len(rows) == 2:
                    break
    prefix_bytes = downloads.bytes - before
    if len(rows) != 2 or prefix_bytes >= pin["size"]:
        raise ValueError("archive did not yield two bounded complete utterances")
    record = {"kind": "pinned archive first two regular WAV members", "archive_url": base + "audio/dev.tar.gz",
              "archive_lfs_sha256_from_pinned_tree": pin["lfs"]["oid"], "archive_size": pin["size"],
              "archive_prefix_bytes_read": prefix_bytes, "whole_archive_downloaded": False,
              "tsv_sha256": hashlib.sha256(labels_blob).hexdigest(), "tree_sha256": hashlib.sha256(tree).hexdigest(),
              "selected": [{key: value for key, value in row.items() if key != "data"} for row in rows]}
    return rows, record
