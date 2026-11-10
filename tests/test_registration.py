"""Separate registration shares the existing log_model Triton version rules."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlflow
import numpy as np
from mlflow.models import Model
from mlflow_triton import register_model, sklearn
from sklearn.dummy import DummyRegressor


def snapshot(path):
    return {p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()}


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.addCleanup(mlflow.set_tracking_uri, mlflow.get_tracking_uri())
        self.addCleanup(mlflow.set_registry_uri, mlflow.get_registry_uri())
        uri = "sqlite:///" + str(self.root / "tracking.db")
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
        self.experiment = mlflow.create_experiment(
            "registration", artifact_location=(self.root / "artifacts").as_uri()
        )

    def log(self, value, columns=1, **kwargs):
        example = np.ones((2, columns))
        model = DummyRegressor(strategy="constant", constant=value).fit(example, [0, 0])
        with mlflow.start_run(experiment_id=self.experiment):
            return sklearn.log_model(
                model,
                input_example=example,
                pip_requirements=["scikit-learn"],
                **kwargs,
            )

    def path(self, uri):
        return Path(mlflow.artifacts.download_artifacts(artifact_uri=uri))

    def test_matching_configs_append_versions_and_preserve_source(self):
        first = self.log(1)
        version = register_model(first.model_uri, "classifier", tags={"owner": "first"})
        self.assertEqual(str(version.version), "1")
        self.assertEqual(version.tags["owner"], "first")
        first_repository = self.path(first.model_uri) / "model_repository/model"
        old_serving = snapshot(first_repository / "1")
        second = self.log(2)
        source_path = self.path(second.model_uri)
        source_snapshot = snapshot(source_path)
        version = register_model(
            second.model_uri, "classifier", tags={"owner": "second"}
        )
        self.assertEqual(str(version.version), "1")
        self.assertEqual(version.tags["owner"], "second")
        self.assertEqual(snapshot(source_path), source_snapshot)
        metadata = mlflow.models.get_model_info("models:/classifier/1")
        self.assertEqual(metadata.model_id, first.model_id)
        self.assertEqual(metadata.flavors["triton"]["model_version"], 2)
        self.assertEqual(
            {p.name for p in first_repository.iterdir() if p.is_dir()}, {"1", "2"}
        )
        self.assertEqual(snapshot(first_repository / "1"), old_serving)
        for uri, value in (
            ("models:/classifier/1", 2),
            (second.model_uri, 2),
            (str(first_repository / "1/artifact"), 1),
        ):
            np.testing.assert_array_equal(
                mlflow.pyfunc.load_model(uri).predict(np.ones((2, 1))), [value] * 2
            )
        register_model(self.log(3).model_uri, "classifier")
        self.assertTrue((first_repository / "3/model.py").is_file())
        self.assertEqual(len(mlflow.MlflowClient().search_model_versions()), 1)

    def test_changed_config_registers_new_model_at_one(self):
        first = self.log(1)
        register_model(first.model_uri, "classifier")
        original = snapshot(self.path(first.model_uri))
        second = self.log(2, columns=2)
        version = register_model(second.model_uri, "classifier")
        self.assertEqual(str(version.version), "2")
        self.assertEqual(snapshot(self.path(first.model_uri)), original)
        info = mlflow.models.get_model_info("models:/classifier/2")
        self.assertEqual(info.model_id, second.model_id)
        self.assertEqual(info.flavors["triton"]["model_version"], 1)
        self.assertEqual(
            {
                p.name
                for p in (
                    self.path(second.model_uri) / "model_repository/model"
                ).iterdir()
                if p.is_dir()
            },
            {"1"},
        )

    def test_works_after_logging_with_registration_and_native_registration(self):
        first = self.log(1, registered_model_name="classifier")
        second = self.log(2)
        register_model(second.model_uri, "classifier")
        self.assertTrue(
            (self.path(first.model_uri) / "model_repository/model/2/model.py").is_file()
        )
        plain = self.log(3)
        mlflow.register_model(plain.model_uri, "other")
        register_model(self.log(4).model_uri, "other")
        self.assertTrue(
            (self.path(plain.model_uri) / "model_repository/model/2/model.py").is_file()
        )

    def test_versioned_source_registered_under_new_name_starts_at_one(self):
        first = self.log(1, registered_model_name="classifier")
        self.log(2, registered_model_name="classifier")
        path = self.path(first.model_uri)
        original = snapshot(path)
        version = register_model("models:/classifier/1", "other")
        self.assertEqual(str(version.version), "1")
        self.assertEqual(snapshot(path), original)
        info = mlflow.models.get_model_info("models:/other/1")
        self.assertNotEqual(info.model_id, first.model_id)
        repo = self.path("models:/other/1") / info.flavors["triton"]["data"]
        self.assertEqual({p.name for p in repo.iterdir() if p.is_dir()}, {"1"})
        np.testing.assert_array_equal(
            mlflow.pyfunc.load_model("models:/other/1").predict(np.ones((2, 1))), [2, 2]
        )

    def test_source_without_triton_uses_standard_mlflow_registration(self):
        with (
            patch(
                "mlflow_triton._registration._previous_artifact_uri",
                return_value="source",
            ),
            patch("mlflow.models.get_model_info", return_value=Model()),
            patch("mlflow.register_model") as native,
        ):
            self.assertIs(
                register_model("models:/m-native", "classifier", 0), native.return_value
            )
        native.assert_called_once_with(
            "models:/m-native", "classifier", await_registration_for=0, tags=None
        )

    def test_registry_failure_does_not_upload_or_register(self):
        source = self.log(1)
        original = snapshot(self.path(source.model_uri))
        with (
            patch(
                "mlflow_triton._versions._latest_version",
                side_effect=PermissionError("registry"),
            ),
            patch("mlflow.register_model") as native,
            self.assertRaisesRegex(PermissionError, "registry"),
        ):
            register_model(source.model_uri, "classifier")
        native.assert_not_called()
        self.assertEqual(snapshot(self.path(source.model_uri)), original)


if __name__ == "__main__":
    unittest.main()
