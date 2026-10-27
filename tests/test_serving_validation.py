"""Serving comparison and orchestration tests; no Docker process is launched."""

import importlib.util
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
        self, fail_client=False, version="3.11", device="cpu", fail_gpu=False
    ):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
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
                    side_effect=[version, "libpython3.11.so"],
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
            self.assertTrue(any(c[:3] == ["docker", "rm", "-f"] for c in commands))
            self.assertTrue((root / "result/result.json").is_file())
            return result, commands

    def test_success_requires_real_client_command(self):
        result, commands = self.run_scenario()
        self.assertEqual(result["triton_validation"], "passed")
        self.assertTrue(any("/validation.py" in c for c in commands))
        self.assertTrue(any("tritonserver" in " ".join(c) for c in commands))

    def test_client_failure_is_not_a_pass(self):
        result, _ = self.run_scenario(fail_client=True)
        self.assertEqual(result["triton_validation"], "failed")
        self.assertEqual(result["triton_stage"], "inference_comparison")

    def test_abi_mismatch_blocks_install_and_inference(self):
        result, commands = self.run_scenario(version="3.14")
        self.assertEqual(result["triton_validation"], "failed")
        self.assertEqual(result["triton_stage"], "python_compatibility")
        self.assertFalse(any("/validation.py" in c for c in commands))

    def test_gpu_enables_docker_cuda_and_checks_availability(self):
        result, commands = self.run_scenario(device="gpu")
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
        result, commands = self.run_scenario(device="gpu", fail_gpu=True)
        self.assertEqual(result["triton_validation"], "failed")
        self.assertEqual(result["triton_stage"], "gpu_availability")
        self.assertFalse(any("/validation.py" in c for c in commands))

    def test_cpu_keeps_gpu_hidden(self):
        result, commands = self.run_scenario()
        command = next(c for c in commands if c[:2] == ["docker", "run"])
        self.assertEqual(result["device"], "cpu")
        self.assertNotIn("--gpus", command)
        self.assertIn("CUDA_VISIBLE_DEVICES=-1", command)
