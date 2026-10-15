"""Independent packed-weight correspondence, bounded storage and opset controls."""
from contextlib import ExitStack, nullcontext
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import compact_weights as compact


def checkpoint_bytes(tensors):
    offset, header, data = 0, {}, b""
    for name, dtype, shape, raw in tensors:
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[offset, offset + len(raw)])
        data += raw
        offset += len(raw)
    raw = json.dumps(header).encode()
    return struct.pack("<Q", len(raw)) + raw + data


def fixture():
    packed = struct.pack("<16I", *([0x76543210] * 8 + [0xFEDCBA98] * 8))
    raw = checkpoint_bytes([
        ("linear.weight", "U32", [1, 16], packed),
        ("linear.scales", "BF16", [1, 2], struct.pack("<2H", 0x3e80, 0xbf00)),
        ("linear.biases", "BF16", [1, 2], struct.pack("<2H", 0xbe00, 0xc050)),
        ("linear.bias", "F32", [1], struct.pack("<f", 2)),
    ])
    expected = [x * 0.25 - 0.125 for x in list(range(8)) * 8]
    expected += [x * -0.5 - 3.25 for x in list(range(8, 16)) * 8]
    return raw, expected


class CompactWeights(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("onnx"), "ONNX required for exact graph metadata control")
    def test_shape_fold_preserves_signed_int64_dynamic_float_and_unrelated_graph_bytes(self):
        import copy
        from collections import Counter
        from onnx import helper as h, TensorProto as T

        literals = [h.make_tensor("lo", T.INT64, [], [-(1 << 63)]),
                    h.make_tensor("hi", T.INT64, [], [(1 << 63) - 1]),
                    h.make_tensor("zero", T.INT64, [1], [0]),
                    h.make_tensor("negative", T.INT64, [1], [-1]),
                    h.make_tensor("float", T.FLOAT, [], [0.125])]
        nodes = [h.make_node("Unsqueeze", ["lo", "zero"], ["low"], name="low"),
                 h.make_node("Unsqueeze", ["hi", "negative"], ["high"], name="high"),
                 h.make_node("Unsqueeze", ["dynamic", "zero"], ["dyn"], name="dynamic"),
                 h.make_node("Unsqueeze", ["float", "zero"], ["flt"], name="float"),
                 h.make_node("Unsqueeze", ["lo", "axes"], ["dynamic_axes"], name="dynamic_axes")]
        model = h.make_model(h.make_graph(nodes, "shape-metadata", [
            h.make_tensor_value_info("dynamic", T.INT64, []),
            h.make_tensor_value_info("axes", T.INT64, [1])], [], literals),
            opset_imports=[h.make_opsetid("", 21)], ir_version=10)
        model.doc_string = "Preserve every unrelated field."
        before = model.SerializeToString(deterministic=True)
        result, proof = compact.finalize_metadata(model, fold_shapes=True)
        self.assertEqual(model.SerializeToString(deterministic=True), before)
        added = {v.name: v for v in list(result.graph.initializer)[len(literals):]}
        self.assertEqual(set(added), {"low", "high"})
        self.assertEqual(list(added["low"].int64_data), [-(1 << 63)])
        self.assertEqual(list(added["high"].int64_data), [(1 << 63) - 1])
        self.assertEqual([list(v.dims) for v in added.values()], [[1], [1]])
        self.assertEqual(Counter(v.SerializeToString() for v in result.graph.node),
                         Counter(v.SerializeToString() for v in nodes[2:]))
        self.assertEqual([v.SerializeToString() for v in result.graph.initializer[:len(literals)]],
                         [v.SerializeToString() for v in literals])
        metadata = []
        for value in (model, result):
            value = copy.deepcopy(value)
            value.graph.ClearField("node")
            value.graph.ClearField("initializer")
            metadata.append(value.SerializeToString())
        self.assertEqual(metadata[0], metadata[1])
        self.assertEqual([(r["axis"], r["value"]) for r in proof["folded_int64_unsqueezes"]],
                         [(0, -(1 << 63)), (-1, (1 << 63) - 1)])
        # Idempotence includes the stable complete input-independent ordering.
        again, second = compact.finalize_metadata(result, fold_shapes=True)
        self.assertEqual(result.SerializeToString(), again.SerializeToString())
        self.assertFalse(second["folded_int64_unsqueezes"])

    @unittest.skipUnless(importlib.util.find_spec("onnx"), "ONNX required for hostile graph metadata control")
    def test_shape_fold_rejects_invalid_constant_axes_and_duplicate_values(self):
        from onnx import helper as h, TensorProto as T
        for axes in ([1], [-2], [0, 0]):
            model = h.make_model(h.make_graph([
                h.make_node("Unsqueeze", ["scalar", "axes"], ["y"], name="shape")], "invalid", [], [], [
                    h.make_tensor("scalar", T.INT64, [], [3]),
                    h.make_tensor("axes", T.INT64, [len(axes)], axes)]))
            with self.subTest(axes=axes), self.assertRaisesRegex(ValueError, "axes"):
                compact.finalize_metadata(model, fold_shapes=True)
        model.graph.node.add(op_type="Identity", input=["scalar"], output=["y"], name="duplicate")
        model.graph.initializer[1].dims[:] = [1]
        model.graph.initializer[1].int64_data[:] = [0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compact.finalize_metadata(model, fold_shapes=True)

    def test_frozen_oracle_inventory_rejects_same_length_tampering_before_graph_work(self):
        names = ("encoder-case/case.json", "encoder-case/expected.json", "encoder-case/features.f32",
                 "encoder-case/expected-encoded.f32", "prompts.json", "provenance.json")
        # Without ONNX, ordinary reads exercise byte admission locally; Windows
        # controls use the existing real deny-write helper.
        portable = (patch.dict("sys.modules", {"shard_encoder": SimpleNamespace(open_source_weights=lambda p: p.open("rb"))})
                    if not importlib.util.find_spec("onnx") else nullcontext())
        with tempfile.TemporaryDirectory() as temporary, portable:
            source = Path(temporary)
            data = {name: bytes([index + 1]) * 8 for index, name in enumerate(names)}
            for name, value in data.items():
                (source / name).parent.mkdir(exist_ok=True)
                (source / name).write_bytes(value)
            pins = {name: (len(value), hashlib.sha256(value).hexdigest()) for name, value in data.items()}
            with ExitStack() as handles:
                self.assertEqual(set(compact.admit_retained(handles, source, pins)), set(names))
            for name in names:
                altered = bytearray(data[name])
                altered[0] ^= 1
                (source / name).write_bytes(altered)
                with self.subTest(name=name), ExitStack() as handles, self.assertRaisesRegex(ValueError, "size/hash differs"):
                    compact.admit_retained(handles, source, pins)
                (source / name).write_bytes(data[name])
            inventory = Path(compact.__file__).with_name("compact-retained-pins.json").read_bytes()
            self.assertEqual(len(compact.retained_pins()), 119)
            (source / "inventory.json").write_bytes(inventory.replace(b"7280", b"7281", 1))
            with self.assertRaisesRegex(ValueError, "inventory hash differs"):
                compact.retained_pins(source / "inventory.json")

    def test_header_rejects_duplicate_keys_overlaps_gaps_truncation_and_unknown_types(self):
        raw, _ = fixture()
        entries = compact.safetensors_header(io.BytesIO(raw), len(raw))
        self.assertEqual(entries["linear.weight"]["shape"], [1, 16])
        header = json.loads(raw[8:8 + struct.unpack("<Q", raw[:8])[0]])
        for key, change in [("dtype", "F16"), ("data_offsets", [1, 65]), ("shape", [-1, 16])]:
            changed = json.loads(json.dumps(header))
            changed["linear.weight"][key] = change
            payload = json.dumps(changed).encode()
            candidate = struct.pack("<Q", len(payload)) + payload + bytes(80)
            with self.subTest(change=change), self.assertRaises(ValueError):
                compact.safetensors_header(io.BytesIO(candidate), len(candidate))
        for bad in (b"", raw[:-1], struct.pack("<Q", 17 << 20),
                    struct.pack("<Q", 13) + b'{"a":1,"a":2}'):
            with self.assertRaises((ValueError, TypeError)):
                compact.safetensors_header(io.BytesIO(bad), len(bad))

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy required for bounded affine conversion")
    def test_every_reconstructed_byte_matches_independent_original_and_rejects_wrong_mapping(self):
        raw, expected = fixture()
        entries = compact.safetensors_header(io.BytesIO(raw), len(raw))
        source = io.BytesIO(raw)
        original = struct.pack("<128f", *expected)
        interval = dict(offset=0, length=len(original))
        compact.verify_affine(source, entries, "linear.weight", [1, 128], io.BytesIO(original), interval)
        with self.assertRaisesRegex(ValueError, "differs from frozen"):
            compact.verify_affine(source, entries, "linear.weight", [1, 128], io.BytesIO(bytes(len(original))), interval)
        with self.assertRaisesRegex(ValueError, "unexpected affine"):
            compact.affine_group(entries, "linear.weight", [2, 64])
        widened = b"".join(compact.bf16_chunks(source, entries["linear.scales"]))
        self.assertEqual(widened, struct.pack("<2f", 0.25, -0.5))

    def test_shared_ranges_preserve_bytes_shard_boundaries_and_reject_partial_copy(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as handles:
            root = Path(temporary)
            store = compact.Store(root, "decoder", handles)
            with patch.object(compact, "SHARD_BYTES", 5):
                first = store.add(("checkpoint", "a"), 4, [b"ab", b"cd"])
                self.assertEqual(store.add(("checkpoint", "a"), 4, []), first)
                second = store.add(("checkpoint", "b"), 3, [b"xyz"])
            handles.close()
            self.assertEqual(store.files, ["decoder.weights-000", "decoder.weights-001"])
            self.assertEqual((root / first["location"]).read_bytes(), b"abcd")
            self.assertEqual((root / second["location"]).read_bytes(), b"xyz")
            self.assertEqual(len(store.ranges), 2)
            with ExitStack() as outputs:
                broken = compact.Store(root, "broken", outputs)
                with self.assertRaisesRegex(ValueError, "truncated"):
                    broken.add("a", 4, [b"abc"])
                self.assertFalse(broken.ranges)

    def test_only_exact_known_embedding_output_aliases_can_close_source_coverage(self):
        tensors = [(prefix + suffix, dtype, shape, data) for prefix in ("decoder.embed_tokens", "proj_out")
                   for suffix, dtype, shape, data in [(".weight", "U32", [1, 8], bytes(range(32))),
                       (".scales", "BF16", [1, 1], b"\x80\x3e"), (".biases", "BF16", [1, 1], b"\0\xbe")]]
        raw = checkpoint_bytes(tensors)
        entries = compact.safetensors_header(io.BytesIO(raw), len(raw))
        used = {name for name in entries if name.startswith("decoder.")}
        aliases = compact.source_aliases(io.BytesIO(raw), entries, set(used))
        self.assertEqual(set(aliases), {"proj_out.weight", "proj_out.scales", "proj_out.biases"})
        changed = bytearray(raw)
        changed[entries["proj_out.weight"]["offset"]] ^= 1
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            compact.source_aliases(io.BytesIO(changed), entries, set(used))
        with self.assertRaisesRegex(ValueError, "incomplete checkpoint coverage"):
            compact.source_aliases(io.BytesIO(raw), entries, set())

    @unittest.skipUnless(importlib.util.find_spec("onnx"), "ONNX required for opset and graph control")
    def test_rewrite_preserves_reduce_mean_semantics_and_shared_decoder_parameter_bytes(self):
        import numpy as np
        import onnx
        from onnx import helper as h, TensorProto as T

        raw, expected = fixture()
        entries = compact.safetensors_header(io.BytesIO(raw), len(raw))
        weights = struct.pack("<128f", *expected) + struct.pack("<f", 2)
        initializers = []
        for name, shape, offset, length in [("linear.weight", [1, 128], 0, 512), ("linear.bias", [1], 512, 4)]:
            value = T(name=name, data_type=T.FLOAT, dims=shape)
            compact.bind(value, dict(location="decoder.weights", offset=offset, length=length))
            initializers.append(value)
        graph = h.make_graph([
            h.make_node("Transpose", ["linear.weight"], ["wt"], perm=[1, 0]),
            h.make_node("MatMul", ["x", "wt"], ["product"]),
            h.make_node("Add", ["product", "linear.bias"], ["biased"]),
            h.make_node("ReduceMean", ["biased"], ["y"], axes=[1], keepdims=1),
        ], "synthetic-original", [h.make_tensor_value_info("x", T.FLOAT, [1, 128])],
            [h.make_tensor_value_info("y", T.FLOAT, [1, 1])], initializers)
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)], ir_version=8)
        model.doc_string = "Keep unrelated model metadata."
        model.metadata_props.add(key="synthetic", value="independent arithmetic")
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as handles:
            root = Path(temporary)
            store = compact.Store(root, "decoder", handles)
            for name in ("init", "step"):
                rewritten, coverage = compact.rewrite(model, io.BytesIO(raw), entries, io.BytesIO(weights),
                    len(weights), "decoder.weights", store, lambda value: value)
                onnx.save(rewritten, root / (name + ".onnx"))
                self.assertEqual(model.ir_version, 8)
                self.assertEqual(rewritten.ir_version, 10)
                self.assertEqual(rewritten.doc_string, model.doc_string)
                self.assertEqual(rewritten.metadata_props, model.metadata_props)
                self.assertEqual([v["kind"] for v in coverage], ["original_affine_uint4", "unchanged_fp32"])
                reduce = next(node for node in rewritten.graph.node if node.op_type == "ReduceMean")
                self.assertEqual(len(reduce.input), 2)
                self.assertNotIn("axes", [a.name for a in reduce.attribute])
            handles.close()
            self.assertEqual(len(store.ranges), 4)
            self.assertEqual((root / store.files[0]).stat().st_size, 64 + 8 + 8 + 4)
            onnx.checker.check_model(str(root / "init.onnx"))
            if importlib.util.find_spec("onnxruntime"):
                import onnxruntime as ort
                x = np.asarray([list(range(128))], dtype=np.float32) / 128
                result, = ort.InferenceSession(str(root / "init.onnx"), providers=["CPUExecutionProvider"]).run(None, {"x": x})
                wanted = sum(float(x[0, i]) * expected[i] for i in range(128)) + 2
                np.testing.assert_allclose(result, [[wanted]], atol=2e-5, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()
