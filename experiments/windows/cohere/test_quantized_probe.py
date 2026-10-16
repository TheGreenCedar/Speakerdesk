"""Independent byte-order, affine-offset and native-result rejection controls."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

SPEC = importlib.util.spec_from_file_location("quantized_probe", Path(__file__).with_name("quantized_probe.py"))
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class QuantizedProbeTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("onnx") and importlib.util.find_spec("numpy"),
                         "requires the existing ONNX generation environment")
    def test_dynamic_chain_external_storage_and_single_scalar_reference(self):
        import onnx

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = root / "interleaved"
            probe.build_case(case, 64, 64, 3, 3, dependent_chain=True,
                             external_model=root / "interleaved.onnx")
            original = onnx.load(root / "interleaved.onnx", load_external_data=False)
            self.assertEqual(original.graph.input[0].type.tensor_type.shape.dim[0].dim_param, "frames")
            self.assertTrue(all(out.type.tensor_type.shape.dim[0].dim_param == "frames"
                                for out in original.graph.output))
            ordered = probe.independent_partition(list(original.graph.node),
                                                   [tensor.name for tensor in original.graph.initializer])
            sizes = {f"matrix{i}_transposed": 64 * 64 * 4 for i in range(3)}
            self.assertEqual(probe.weight_liveness(list(original.graph.node), sizes), 3 * 64 * 64 * 4)
            self.assertEqual(probe.weight_liveness(ordered, sizes), 64 * 64 * 4)
            before = sorted(node.SerializeToString() for node in original.graph.node)
            del original.graph.node[:]
            original.graph.node.extend(ordered)
            self.assertEqual(before, sorted(node.SerializeToString() for node in original.graph.node))
            onnx.save(original, root / "hoisted.onnx")
            for filename in ("interleaved.onnx", "hoisted.onnx"):
                onnx.checker.check_model(str(root / filename))
            tensors = {tensor.name: tensor for tensor in
                       onnx.load(root / "interleaved.onnx", load_external_data=True).graph.initializer}
            inputs = probe.read_floats(case / "input.f32")
            for i in range(3):
                prefix = f"matrix{i}"
                weights = probe.reconstruct(tensors[prefix + "_q4"].raw_data,
                    struct.unpack("<64f", tensors[prefix + "_scale"].raw_data),
                    struct.unpack("<64f", tensors[prefix + "_offset"].raw_data), 64, 64)
                expected = [probe.f32(sum(inputs[frame*64 + column] * weights[row*64 + column]
                                         for column in range(64)))
                            for frame in range(3) for row in range(64)]
                actual = probe.read_floats(case / f"{prefix}_product.f32")
                self.assertTrue(all(abs(a-b) <= 0.002 + 0.002*abs(b) for a, b in zip(actual, expected)))
                inputs = [probe.f32(value / 256) for value in expected]

    def test_complete_static_closure_controls_dependent_weight_lifetimes(self):
        nodes = []
        for i in range(3):
            nodes.extend([
                SimpleNamespace(name=f"q{i}", input=[f"packed{i}"], output=[f"weight{i}"]),
                # The original Transpose must move too; moving only new q nodes fails.
                SimpleNamespace(name=f"transpose{i}", input=[f"weight{i}"], output=[f"transposed{i}"]),
                SimpleNamespace(name=f"product{i}", input=["input" if i == 0 else f"product{i-1}",
                                                           f"transposed{i}"], output=[f"product{i}"]),
            ])
        sizes = {f"transposed{i}": 36 * 1024 * 1024 for i in range(3)}
        hoisted = probe.independent_partition(nodes, [f"packed{i}" for i in range(3)])
        self.assertEqual(probe.weight_liveness(nodes, sizes), 108 * 1024 * 1024)
        self.assertEqual(probe.weight_liveness(hoisted, sizes), 36 * 1024 * 1024)
        self.assertEqual(sorted(map(id, nodes)), sorted(map(id, hoisted)))
        for ordered in (nodes, hoisted):
            values = dict(input=2, packed0=3, packed1=5, packed2=7)
            for node in ordered:
                values[node.output[0]] = (values[node.input[0]] if len(node.input) == 1
                                          else values[node.input[0]] * values[node.input[1]])
            self.assertEqual(values["product2"], 210)

    def test_private_commit_contrast_rejects_loader_dominance_small_effect_and_guard_breach(self):
        mib = 1024 * 1024
        family = dict(private_commit_limit_bytes=2048*mib,
                      maximum_loaded_commit_difference_bytes=36*mib,
                      minimum_peak_commit_reduction_bytes=108*mib)
        reports = {name: dict(private_commit=dict(lifetime_peak_bytes=peak*mib),
                             before_run=dict(current_bytes=48*mib, lifetime_peak_bytes=64*mib),
                             run_complete_peak_bytes=peak*mib)
                   for name, peak in (("interleaved", 400), ("hoisted", 200))}
        self.assertEqual(probe.scheduling_commit_gate(family, reports)["status"], "passed")
        for attack, reason in (("baseline", "baselines differ"), ("loader", "loader peak dominates"),
                              ("effect", "less than"), ("guard", "exceeded")):
            changed = copy.deepcopy(reports)
            if attack == "baseline":
                changed["hoisted"]["before_run"]["current_bytes"] = 85*mib
            elif attack == "loader":
                changed["hoisted"]["run_complete_peak_bytes"] = 64*mib
            elif attack == "effect":
                changed["hoisted"]["private_commit"]["lifetime_peak_bytes"] = 293*mib
            else:
                changed["interleaved"]["private_commit"]["lifetime_peak_bytes"] = 2049*mib
            result = probe.scheduling_commit_gate(family, changed)
            self.assertEqual(result["status"], "not_demonstrated")
            self.assertTrue(any(reason in value for value in result["reasons"]), result)

    def test_per_case_private_peak_stops_even_when_current_commit_is_below_guard(self):
        limit = 2 * 1024 ** 3
        observations = [dict(current_bytes=48 * 1024 ** 2, lifetime_peak_bytes=peak,
                             source="GetProcessMemoryInfo/PROCESS_MEMORY_COUNTERS_EX")
                        for peak in (64 * 1024 ** 2, limit, limit, limit, limit)]
        probe.check_private_commit_observations(observations, limit)
        # A brief breach survives in the OS lifetime peak after current commit drops.
        for value in observations[1:]:
            value["lifetime_peak_bytes"] = limit + 1
        with self.assertRaisesRegex(ValueError, "per-case 2 GiB guard"):
            probe.check_private_commit_observations(observations, limit)

    def test_low_nibble_first_affine_groups_retain_noninteger_zero_points(self):
        # Same packing as two original MLX U32 words, repeated to fill each group.
        packed = struct.pack("<8I", *([0x76543210] * 8)) + struct.pack("<8I", *([0xFEDCBA98] * 8))
        expected = [x * 0.25 - 0.125 for x in list(range(8)) * 8]
        expected += [x * -0.5 - 3.25 for x in list(range(8, 16)) * 8]
        self.assertEqual(probe.reconstruct(packed, [0.25, -0.5], [-0.125, -3.25], 1, 128), expected)
        for invalid in ((packed[:-1], [0.25, -0.5], [-0.125, -3.25], 1, 128),
                        (packed, [0.25], [-0.125, -3.25], 1, 128),
                        (packed, [0.25, -0.5], [-0.125, -3.25], 1, 127)):
            with self.assertRaises(ValueError):
                probe.reconstruct(*invalid)

    def test_native_results_reject_wrong_reconstruction_drift_repeats_and_missing_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case, result = root / "case", root / "result"
            case.mkdir()
            result.mkdir()
            references = [dict(name="weights", shape=[1, 2], file="weights.f32", exact=True),
                          dict(name="product", shape=[1, 2], file="product.f32", exact=False)]
            (case / "weights.f32").write_bytes(struct.pack("<2f", -0.125, 3.625))
            (case / "product.f32").write_bytes(struct.pack("<2f", 2, -1))
            probe.write_json(case / "reference.json", dict(outputs=references, atol=0.002, rtol=0.002))
            outputs = []
            for run in range(2):
                for reference in references:
                    path = result / f"{run}-{reference['file']}"
                    path.write_bytes((case / reference["file"]).read_bytes())
                    outputs.append(dict(run=run, name=reference["name"], shape=reference["shape"],
                                        dtype="f32", file=path.name, sha256=probe.verifier.sha256(path)))
            record = dict(outputs=outputs)
            self.assertEqual(probe.compare_outputs(case, result, record), dict(weights=0, product=0))

            for position, values, reason in [
                (0, [0, 3.625], "reconstructed Float32 bits"),
                (1, [2.1, -1], "matrix product exceeds"),
                (1, [float("nan"), -1], "nonfinite"),
                (3, [2.001, -1], "repeats differ"),
            ]:
                changed = copy.deepcopy(record)
                output = changed["outputs"][position]
                path = result / output["file"]
                previous = path.read_bytes()
                path.write_bytes(struct.pack("<2f", *values))
                output["sha256"] = probe.verifier.sha256(path)
                with self.assertRaisesRegex(ValueError, reason):
                    probe.compare_outputs(case, result, changed)
                path.write_bytes(previous)
            for changed in (dict(outputs=outputs[:-1]), dict(outputs=outputs[:-1] + [outputs[0]])):
                with self.assertRaises(ValueError):
                    probe.compare_outputs(case, result, changed)
            changed = copy.deepcopy(record)
            changed["outputs"][0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "hash differs"):
                probe.compare_outputs(case, result, changed)


if __name__ == "__main__":
    unittest.main()
