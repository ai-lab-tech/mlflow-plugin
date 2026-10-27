"""Serving comparison and orchestration tests; no Docker process is launched."""

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "examples" / "serving_validation.py"
SPEC = importlib.util.spec_from_file_location("serving_validation_test", SOURCE)
serving = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(serving)


class ServingValidationTests(unittest.TestCase):
    def test_numeric_string_and_nonfinite_binary_comparison(self):
        expected = {
            "score": np.array([1.0, np.nan, np.inf], dtype=np.float32),
            "label": np.array(["Berlin"]),
        }
        actual = {
            "score": expected["score"].copy(),
            "label": np.array([b"Berlin"], dtype=object),
        }
        specs = [
            {"name": "score", "datatype": "FP32"},
            {"name": "label", "datatype": "BYTES"},
        ]
        serving.compare_outputs(actual, expected, specs)
        for replacement in [
            np.array([2.0, np.nan, np.inf], dtype=np.float32),
            expected["score"].astype(np.float64),
            np.array([1], dtype=np.float32),
        ]:
            with self.assertRaises(AssertionError):
                serving.compare_outputs(
                    {**actual, "score": replacement}, expected, specs
                )
        with self.assertRaises(AssertionError):
            serving.compare_outputs({"score": actual["score"]}, expected, specs)

    def run_scenario(
        self,
        fail_client=False,
        version="3.11",
        device="cpu",
        fail_gpu=False,
        fail_reference=False,
    ):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        row = {
            "case": "example",
            "serving_path": str(root / "model"),
            "fixture_path": str(root / "fixture.npz"),
        }
        options = SimpleNamespace(
            timeout=10, startup_timeout=5, inference_timeout=5, device=device
        )
        commands = []

        def run(command, **kwargs):
            commands.append(command)
            if fail_reference and "--native-gpu-reference" in command:
                raise subprocess.CalledProcessError(1, command)
            if fail_gpu and any(
                "torch.cuda.is_available()" in part for part in command
            ):
                raise subprocess.CalledProcessError(1, command)
            if fail_client and "/validation.py" in command:
                raise subprocess.CalledProcessError(1, command)
            return subprocess.CompletedProcess(command, 0)

        with (
            patch.object(serving.subprocess, "run", side_effect=run),
            patch.object(
                serving.subprocess,
                "check_output",
                side_effect=[
                    version,
                    "libpython3.11.so",
                    json.dumps(
                        {
                            "native_gpu_vs_cpu": {"status": "failed"},
                            "triton_vs_native_gpu": {"status": "passed"},
                            "triton_vs_cpu": {"status": "passed"},
                        }
                    ),
                ],
            ),
        ):
            result = serving.validate_serving(
                row,
                "runtime:test",
                "3.11",
                root / "requirements.txt",
                root / "result",
                options,
            )
        return result, commands, root / "result/result.json"

    def test_success_requires_real_client_command(self):
        result, commands, report_path = self.run_scenario()
        # Assert
        self.assertTrue(any(c[:3] == ["docker", "rm", "-f"] for c in commands))
        self.assertTrue(report_path.is_file())
        self.assertEqual(result["triton_validation"], "passed")
        self.assertTrue(any("/validation.py" in c for c in commands))
        self.assertTrue(any("tritonserver" in " ".join(c) for c in commands))

    def test_client_failure_is_not_a_pass(self):
        result, commands, report_path = self.run_scenario(fail_client=True)
        # Assert
        self.assertTrue(any(c[:3] == ["docker", "rm", "-f"] for c in commands))
        self.assertTrue(report_path.is_file())
        self.assertEqual(result["triton_validation"], "failed")
        self.assertEqual(result["triton_stage"], "inference_comparison")

    def test_abi_mismatch_blocks_install_and_inference(self):
        result, commands, report_path = self.run_scenario(version="3.14")
        # Assert
        self.assertTrue(any(c[:3] == ["docker", "rm", "-f"] for c in commands))
        self.assertTrue(report_path.is_file())
        self.assertEqual(result["triton_validation"], "failed")
        self.assertEqual(result["triton_stage"], "python_compatibility")
        self.assertFalse(any("/validation.py" in c for c in commands))

    def test_gpu_enables_docker_cuda_and_checks_availability(self):
        result, commands, report_path = self.run_scenario(device="gpu")
        # Assert
        self.assertTrue(any(c[:3] == ["docker", "rm", "-f"] for c in commands))
        self.assertTrue(report_path.is_file())
        self.assertEqual(result["triton_validation"], "passed")
        self.assertEqual(result["device"], "gpu")
        command = next(c for c in commands if c[:2] == ["docker", "run"])
        self.assertIn("--gpus", command)
        self.assertNotIn("CUDA_VISIBLE_DEVICES=-1", command)
        self.assertNotIn("NVIDIA_VISIBLE_DEVICES=void", command)
        self.assertTrue(
            any("torch.cuda.is_available()" in " ".join(c) for c in commands)
        )
        client = next(c for c in commands if "/validation.py" in c)
        self.assertEqual(client[client.index("--device") + 1], "gpu")
        install = next(c for c in commands if "pip" in c)
        self.assertNotIn("https://download.pytorch.org/whl/cpu", install)

    def test_gpu_failure_does_not_fall_back_to_cpu(self):
        result, commands, report_path = self.run_scenario(device="gpu", fail_gpu=True)
        # Assert
        self.assertTrue(any(c[:3] == ["docker", "rm", "-f"] for c in commands))
        self.assertTrue(report_path.is_file())
        self.assertEqual(result["triton_validation"], "failed")
        self.assertEqual(result["triton_stage"], "gpu_availability")
        self.assertFalse(any("/validation.py" in c for c in commands))

    def test_cpu_keeps_gpu_hidden(self):
        result, commands, report_path = self.run_scenario()
        # Assert
        self.assertTrue(any(c[:3] == ["docker", "rm", "-f"] for c in commands))
        self.assertTrue(report_path.is_file())
        command = next(c for c in commands if c[:2] == ["docker", "run"])
        self.assertEqual(result["device"], "cpu")
        self.assertNotIn("--gpus", command)
        self.assertIn("CUDA_VISIBLE_DEVICES=-1", command)

    def test_comparison_report_keeps_cpu_gpu_difference_visible(self):
        expected = {"x": np.array([0.0, 1.0], dtype=np.float32)}
        actual = {"x": np.array([6e-6, 1.0], dtype=np.float32)}
        report = serving.comparison_result(
            actual, expected, [{"name": "x", "datatype": "FP32"}]
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["outputs"]["x"]["mismatched_elements"], 1)
        self.assertEqual(report["atol"], 1e-6)
        self.assertGreater(
            report["outputs"]["x"]["max_absolute_finite_difference"], 5e-6
        )

    def test_gpu_reference_runs_before_triton_start_and_cpu_drift_stays_separate(self):
        result, commands, report_path = self.run_scenario(device="gpu")
        # Assert
        self.assertTrue(any(c[:3] == ["docker", "rm", "-f"] for c in commands))
        self.assertTrue(report_path.is_file())
        reference = next(
            i for i, c in enumerate(commands) if "--native-gpu-reference" in c
        )
        server = next(
            i
            for i, c in enumerate(commands)
            if any("exec /opt/tritonserver/bin/tritonserver" in part for part in c)
        )
        self.assertLess(reference, server)
        self.assertEqual(result["triton_validation"], "passed")
        self.assertEqual(result["native_gpu_vs_cpu"]["status"], "failed")
        self.assertEqual(result["comparison"], "saved_native_artifact_gpu")

    def test_failed_native_gpu_reference_blocks_server_start(self):
        result, commands, report_path = self.run_scenario(
            device="gpu", fail_reference=True
        )
        # Assert
        self.assertTrue(any(c[:3] == ["docker", "rm", "-f"] for c in commands))
        self.assertTrue(report_path.is_file())
        self.assertEqual(result["triton_validation"], "failed")
        self.assertEqual(result["triton_stage"], "native_gpu_reference")
        self.assertFalse(
            any(
                "exec /opt/tritonserver/bin/tritonserver" in " ".join(c)
                for c in commands
            )
        )
