"""Core installed-package logging smoke test."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import mlflow
from mlflow_triton import sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


class SmokeTests(unittest.TestCase):
    def test_real_trained_models_compatible_and_incompatible_versions(self):
        # Arrange
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke"
            example = (
                Path(__file__).resolve().parents[1] / "examples/smoke_versioning.py"
            )
            # Act
            result = subprocess.run(
                [sys.executable, str(example), "--output", str(output)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            # Assert
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["status"], "passed")
            self.assertEqual(
                [row["triton_versions"] for row in report["versions"]],
                [["1"], ["2"], ["1"]],
            )

    def test_optimized_smoke_rejects_missing_backend_and_records_failure(self):
        # Arrange: damage a registered artifact after logging, before inspection.
        examples = Path(__file__).resolve().parents[1] / "examples"
        script = """
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
import smoke_versioning
original = smoke_versioning.mlflow.artifacts.download_artifacts
def download(*args, **kwargs):
    result = original(*args, **kwargs)
    if kwargs.get("artifact_uri") == "models:/iris-versioning-smoke/2":
        (Path(result) / "model_repository/iris/2/model.py").unlink()
    return result
with patch.object(smoke_versioning.mlflow.artifacts, "download_artifacts", download):
    smoke_versioning.run_smoke(sys.argv[2])
"""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "smoke"

            # Act
            result = subprocess.run(
                [sys.executable, "-O", "-c", script, str(examples), str(output)],
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Assert
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertIn("missing backend", report["error"])

    def test_log_register_and_save_real_mlflow_model(self):
        # Arrange
        tracking_uri, registry_uri = (
            mlflow.get_tracking_uri(),
            mlflow.get_registry_uri(),
        )
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                uri = f"sqlite:///{root / 'mlflow.db'}"
                mlflow.set_tracking_uri(uri)
                mlflow.set_registry_uri(uri)
                experiment = mlflow.create_experiment(
                    "triton-smoke", artifact_location=(root / "artifacts").as_uri()
                )
                inputs, targets = load_iris(return_X_y=True, as_frame=True)
                model = LogisticRegression(max_iter=200).fit(inputs, targets)
                # Act
                with mlflow.start_run(experiment_id=experiment):
                    info = sklearn.log_model(
                        model,
                        name="iris",
                        input_example=inputs.iloc[:2],
                        registered_model_name="iris-serving",
                    )
                repositories = []
                for model_uri in (info.model_uri, "models:/iris-serving/1"):
                    metadata = mlflow.models.get_model_info(model_uri)
                    model_root = Path(
                        mlflow.artifacts.download_artifacts(artifact_uri=model_uri)
                    )
                    repositories.append(model_root / metadata.flavors["triton"]["data"])

                # Assert
                self.assertEqual(str(info.registered_model_version), "1")
                for repository in repositories:
                    self.assertTrue((repository / "1/model.py").is_file())
                    self.assertTrue((repository / "1/artifact/MLmodel").is_file())
                    self.assertTrue((repository / "config.pbtxt").is_file())
        finally:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_registry_uri(registry_uri)
