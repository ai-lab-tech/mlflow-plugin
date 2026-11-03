"""Core installed-package logging smoke test."""

import tempfile
import unittest
from pathlib import Path

import mlflow
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from mlflow_triton import sklearn


class SmokeTests(unittest.TestCase):
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
