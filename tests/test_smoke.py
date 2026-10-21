"""Smoke tests using installed MLflow and a local tracking/registry database."""

import tempfile
import unittest
from pathlib import Path

import mlflow

from mlflow_triton_control import triton


class SmokeTests(unittest.TestCase):
    def test_log_register_and_load_real_mlflow_model(self):
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
                source = root / "iris"
                (source / "1").mkdir(parents=True)
                (source / "1/model.onnx").write_bytes(b"smoke-model")
                # Act
                with mlflow.start_run(experiment_id=experiment):
                    info = triton.log_model(
                        source, registered_model_name="iris-serving"
                    )
                repositories = [
                    triton.load_model(model_uri)
                    for model_uri in (info.model_uri, "models:/iris-serving/1")
                ]

                # Assert
                self.assertEqual(str(info.registered_model_version), "1")
                for repository in repositories:
                    self.assertEqual(
                        (repository / "1/model.onnx").read_bytes(), b"smoke-model"
                    )
        finally:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_registry_uri(registry_uri)
