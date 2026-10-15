"""Byte preservation, aliasing, hostile metadata and native small-model equivalence."""
import copy
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

import shard_encoder as shard


def pins(directory):
    return {name: {"size": (directory / name).stat().st_size,
                   "sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest()}
            for name in ("encoder.onnx", "encoder.weights")}


def fixture(directory):
    directory.mkdir()
    a = np.asarray([[1, 2], [3, 4]], "<f4")
    c, d = np.asarray([5, -3], "<f4"), np.asarray([1, 7], "<f4")
    (directory / "encoder.weights").write_bytes(a.tobytes() + b"PADDING!" + c.tobytes() + d.tobytes())
    values = []
    for name, array, offset in [("a", a, 0), ("b", a, 0), ("c", c, 24), ("d", d, 32)]:
        tensor = numpy_helper.from_array(array, name)
        tensor.doc_string = f"retain every field of {name}"
        onnx.external_data_helper.set_external_data(tensor, "encoder.weights", offset, array.nbytes)
        tensor.ClearField("raw_data")
        values.append(tensor)
    graph = helper.make_graph([
        helper.make_node("MatMul", ["x", "a"], ["xa"], name="first"),
        helper.make_node("MatMul", ["x", "b"], ["xb"], name="alias"),
        helper.make_node("Add", ["xa", "xb"], ["sum"]),
        helper.make_node("Add", ["sum", "c"], ["biased"]),
        helper.make_node("Add", ["biased", "d"], ["y"]),
    ], "external-alias-test", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])], values)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=7)
    model.doc_string = "All graph fields must survive repacking."
    model.metadata_props.add(key="provenance", value="synthetic independent arithmetic")
    onnx.save_model(model, directory / "encoder.onnx")
    return model


class BoundedReader:
    def __init__(self, handle, truncate=False):
        self.handle, self.truncate = handle, truncate

    def __getattr__(self, name):
        return getattr(self.handle, name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self.handle.__exit__(*args)

    def read(self, count):
        if not 0 < count <= 3:
            raise AssertionError("tensor copy is not bounded by the configured chunk size")
        return b"" if self.truncate else self.handle.read(count)


class ShardEncoder(unittest.TestCase):
    def test_graph_parse_uses_the_hashed_snapshot_despite_path_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, output = root / "source", root / "output"
            original = fixture(source)
            original_pins = pins(source)
            graph = source / "encoder.onnx"
            graph_bytes = graph.read_bytes()
            changed = copy.deepcopy(original)
            changed.doc_string = changed.doc_string.replace("All", "New")
            replacement = changed.SerializeToString()
            self.assertEqual(len(replacement), len(graph_bytes))
            sha256, file_digest = hashlib.sha256, hashlib.file_digest
            attempted = False

            def mutate():
                nonlocal attempted
                attempted = True
                graph.write_bytes(replacement)

            def digest_bytes(data=b""):
                result = sha256(data)
                if data == graph_bytes:
                    mutate()
                return result

            def digest_file(stream, algorithm):
                result = file_digest(stream, algorithm)
                if not isinstance(stream.name, int) and Path(stream.name) == graph:
                    mutate()
                return result

            with patch.object(shard.hashlib, "sha256", digest_bytes), patch.object(shard.hashlib, "file_digest", digest_file):
                shard.repack(source, output, original_pins, 24)
            self.assertTrue(attempted)
            actual = onnx.load(output / "encoder.onnx", load_external_data=False)
            for old, new in zip(original.graph.initializer, actual.graph.initializer):
                del new.external_data[:]
                new.external_data.extend(old.external_data)
            self.assertEqual(actual.SerializeToString(), original.SerializeToString())

    def test_same_length_source_mutation_cannot_publish_old_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, output = root / "source", root / "output"
            fixture(source)
            original_pins = pins(source)
            weights = source / "encoder.weights"
            replacement = bytearray(weights.read_bytes())
            replacement[0] ^= 1
            file_digest = hashlib.file_digest
            attempted, blocked = False, False

            def digest_then_mutate(stream, algorithm):
                nonlocal attempted, blocked
                result = file_digest(stream, algorithm)
                name = stream.name
                if not attempted and (isinstance(name, int) or Path(name) == weights):
                    attempted = True
                    try:
                        weights.write_bytes(replacement)
                    except PermissionError:
                        blocked = True
                return result

            with patch.object(shard.hashlib, "file_digest", digest_then_mutate):
                if os.name == "nt":
                    shard.repack(source, output, original_pins, 24)
                    self.assertTrue(blocked, "the retained Windows handle must block mutation")
                    self.assertEqual(pins(source), original_pins)
                else:
                    try:
                        shard.repack(source, output, original_pins, 24)
                    except ValueError as error:
                        self.assertIn("changed", str(error))
                    else:
                        self.fail("same-length mutation published a manifest claiming the old source hash")
                    self.assertFalse((output / "encoder-shards.json").exists())
            self.assertTrue(attempted)

    def test_boundaries_aliases_every_tensor_byte_and_ort_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source, output = root / "source", root / "output"
            original = fixture(source)
            original_pins = pins(source)
            open_weights = shard.open_source_weights
            def bounded_open(path):
                return BoundedReader(open_weights(path))
            with patch.object(shard, "COPY_BYTES", 3), patch.object(shard, "open_source_weights", bounded_open):
                manifest = shard.repack(source, output, original_pins, shard_bytes=24)
            self.assertEqual(pins(source), original_pins)
            self.assertEqual(manifest["tensors"], 4)
            self.assertEqual(manifest["unique_ranges"], 3)
            self.assertEqual([manifest["files"][name]["size"] for name in manifest["files"]
                              if name != "encoder.onnx"], [24, 8])
            repacked = onnx.load(output / "encoder.onnx", load_external_data=False)
            restored = copy.deepcopy(repacked)
            for old, new, restore in zip(original.graph.initializer, repacked.graph.initializer,
                                          restored.graph.initializer):
                before = {v.key: v.value for v in old.external_data}
                after = {v.key: v.value for v in new.external_data}
                expected = (source / before["location"]).read_bytes()[int(before["offset"]):int(before["offset"]) + int(before["length"])]
                actual = (output / after["location"]).read_bytes()[int(after["offset"]):int(after["offset"]) + int(after["length"])]
                self.assertEqual(actual, expected, old.name)
                del restore.external_data[:]
                restore.external_data.extend(old.external_data)
            self.assertEqual(restored.SerializeToString(), original.SerializeToString())
            self.assertEqual(repacked.graph.initializer[0].external_data,
                             repacked.graph.initializer[1].external_data)
            for name, pin in manifest["files"].items():
                self.assertEqual(shard.sha256(output / name), pin["sha256"])
                self.assertEqual((output / name).stat().st_size, pin["size"])
            for row in manifest["ranges"]:
                data = (output / row["location"]).read_bytes()[row["offset"]:row["offset"] + row["length"]]
                self.assertEqual(hashlib.sha256(data).hexdigest(), row["sha256"])
            onnx.checker.check_model(str(output / "encoder.onnx"))
            for directory in (source, output):
                session = ort.InferenceSession(str(directory / "encoder.onnx"), providers=["CPUExecutionProvider"])
                actual, = session.run(None, {"x": np.asarray([[2, -1]], np.float32)})
                np.testing.assert_array_equal(actual, [[4, 4]])

    def test_invalid_metadata_ranges_and_oversized_tensor_fail_before_output(self):
        changes = [
            ("location", "../encoder.weights"), ("location", "/encoder.weights"),
            ("location", r"C:\encoder.weights"), ("location", r"..\encoder.weights"),
            ("location", "sub/encoder.weights"), ("location", "encoder.weights:stream"),
            ("offset", "-1"), ("offset", "1.0"), ("offset", "32"),
            ("length", "0"), ("length", "15"), ("length", "+16"),
        ]
        for key, value in changes:
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                model = fixture(root / "source")
                next(v for v in model.graph.initializer[0].external_data if v.key == key).value = value
                onnx.save_model(model, root / "source/encoder.onnx")
                with self.assertRaises(ValueError):
                    shard.repack(root / "source", root / "output", pins(root / "source"), 24)
                self.assertFalse((root / "output").exists())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            fixture(root / "source")
            for limit in (8, 0, shard.MAX_SHARD_BYTES + 1):
                with self.subTest(limit=limit), self.assertRaises(ValueError):
                    shard.repack(root / "source", root / "output", pins(root / "source"), limit)
                self.assertFalse((root / "output").exists())
            path = root / "source/encoder.weights"
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "truncated"):
                shard.repack(root / "source", root / "output", pins(root / "source"), 24)
            self.assertFalse((root / "output").exists())

    def test_source_pins_fresh_output_and_midcopy_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            fixture(root / "source")
            original_pins = pins(root / "source")
            for name in original_pins:
                changed = copy.deepcopy(original_pins)
                changed[name]["sha256"] = "00" * 32
                with self.subTest(pin=name), self.assertRaisesRegex(ValueError, "hash"):
                    shard.repack(root / "source", root / "output", changed, 24)
                self.assertFalse((root / "output").exists())
            (root / "output").mkdir()
            (root / "output/marker").write_bytes(b"retain")
            with self.assertRaises(FileExistsError):
                shard.repack(root / "source", root / "output", original_pins, 24)
            self.assertEqual((root / "output/marker").read_bytes(), b"retain")
            with self.assertRaises(ValueError):
                shard.repack(root / "source", root / "source/nested", original_pins, 24)
            open_weights = shard.open_source_weights
            def truncated_open(path):
                return BoundedReader(open_weights(path), truncate=True)
            with patch.object(shard, "COPY_BYTES", 3), patch.object(shard, "open_source_weights", truncated_open):
                with self.assertRaisesRegex(ValueError, "truncated during copy"):
                    shard.repack(root / "source", root / "partial", original_pins, 24)
            self.assertFalse((root / "partial/encoder-shards.json").exists())
            self.assertEqual(pins(root / "source"), original_pins)


if __name__ == "__main__":
    unittest.main()
