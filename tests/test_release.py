"""Exercise release preparation in an isolated build checkout."""

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import tomllib


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        repository = Path(__file__).resolve().parents[1]
        for name in (
            "VERSION",
            "scripts/check_release.py",
            "packages/mlflow-triton/pyproject.toml",
            "packages/mlflow-triton-control/pyproject.toml",
        ):
            destination = self.root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repository / name, destination)

    def run_release(self, *args):
        return subprocess.run(
            [sys.executable, str(self.root / "scripts/check_release.py"), *args],
            capture_output=True,
            text=True,
        )

    def test_tag_sets_both_versions_and_control_dependency(self):
        for _ in range(2):
            result = self.run_release("--tag", "v9.8.7", "--write-version")
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "VERSION").read_text(), "9.8.7\n")
        for name in ("mlflow-triton", "mlflow-triton-control"):
            project = tomllib.loads(
                (self.root / "packages" / name / "pyproject.toml").read_text()
            )["project"]
            self.assertEqual(project["version"], "9.8.7")
            if name == "mlflow-triton-control":
                self.assertIn("mlflow-triton==9.8.7", project["dependencies"])
        result = self.run_release("--tag", "v9.8.7")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_tag_does_not_modify_checkout(self):
        original = {
            path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()
        }
        for tag in ("9.8.7", "v9.8.7rc1", "v09.8.7", "v9.8.7/invalid"):
            with self.subTest(tag=tag):
                result = self.run_release("--tag", tag, "--write-version")
                self.assertNotEqual(result.returncode, 0)
                for path, content in original.items():
                    self.assertEqual(path.read_bytes(), content)

    def test_validation_without_write_rejects_mismatched_tag(self):
        result = self.run_release("--tag", "v9.8.7")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Expected tag", result.stderr)

    def test_write_requires_tag(self):
        result = self.run_release("--write-version")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires --tag", result.stderr)
