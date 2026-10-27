"""Runner selection and disk-placement checks without Docker execution."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))
try:
    SPEC = importlib.util.spec_from_file_location("example_runner_test", EXAMPLES / "run_real_models.py")
    runner = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(runner)
finally:
    sys.path.remove(str(EXAMPLES))


class ExampleRunnerTests(unittest.TestCase):
    def test_selected_case_and_packaging_tmp_are_on_output_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            case = "sklearn_mixed_columns"
            (data / runner.CASES[case][1]).mkdir(parents=True)
            images = root / "images.json"
            images.write_text(json.dumps({"3.11": "runtime:test"}))
            output = root / "output"
            argv = [
                "runner",
                "--data-root",
                str(data),
                "--output",
                str(output),
                "--triton-images",
                str(images),
                "--case",
                case,
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(runner.shutil, "which", return_value="/usr/bin/docker"),
                patch.object(runner, "runtime_requirements", return_value="numpy\n"),
                patch.object(
                    runner.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)
                ) as execute,
                patch.object(runner, "validate_serving") as serve,
            ):
                self.assertEqual(runner.main(), 1)
            commands = [call.args[0] for call in execute.call_args_list]
            containers = [cmd for cmd in commands if cmd[:2] == ["docker", "run"]]
            self.assertEqual(len(containers), 1)
            command = containers[0]
            self.assertEqual(command[command.index("--case") + 1 :], [case])
            self.assertIn(f"{output / 'work/sklearn'}:/tmp:rw", command)
            self.assertIn("TMPDIR=/tmp", command)
            self.assertTrue((output / "work/sklearn").is_dir())
            rows = json.loads((output / "summary.json").read_text())
            self.assertEqual([row["case"] for row in rows], [case])
            self.assertEqual(rows[0]["triton_validation"], "blocked_by_local_failure")
            serve.assert_not_called()

    def test_gpu_requirements_and_build_do_not_reuse_cpu_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            requirements = root / runner.GROUPS["pytorch"][1]
            requirements.parent.mkdir(parents=True)
            requirements.write_text("numpy==2.5.4\ntorch==2.14.1+cpu\n")
            cpu = runner.runtime_requirements(root, "pytorch", "cpu")
            gpu = runner.runtime_requirements(root, "pytorch", "gpu")
            self.assertIn("torch==2.14.1+cpu", cpu)
            self.assertIn("torch==2.14.1\n", gpu)
            self.assertNotIn("+cpu", gpu)
            self.assertNotIn("whl/cpu", gpu)
        source = runner.dockerfile("nvcr.io/nvidia/tritonserver:26.06-py3", "3.11", "gpu")
        self.assertIn("TRITON_ENABLE_GPU=ON", source)
        self.assertNotIn("CUDA_VISIBLE_DEVICES=-1", source)

    def test_explicit_sklearn_gpu_selection_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "runner",
                "--data-root",
                tmp,
                "--output",
                str(Path(tmp) / "out"),
                "--device",
                "gpu",
                "--case",
                "sklearn_mixed_columns",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(runner.shutil, "which", return_value="/usr/bin/docker"),
                patch.object(runner.subprocess, "run") as execute,
            ):
                with self.assertRaises(SystemExit) as error:
                    runner.main()
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(any(call.args[0][:2] == ["docker", "run"] for call in execute.call_args_list))
