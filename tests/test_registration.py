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
    # MLflow itself writes registered_model_meta when downloading via Registry URIs.
    return {
        p.relative_to(path): p.read_bytes()
        for p in path.rglob("*")
        if p.is_file() and p.name != "registered_model_meta"
    }


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

    def test_matching_configs_create_registry_versions_and_preserve_old_models(self):
        first = self.log(1)
        version = register_model(first.model_uri, "classifier", tags={"owner": "first"})
        self.assertEqual(str(version.version), "1")
        first_path = self.path(first.model_uri)
        original = snapshot(first_path)
        second = self.log(2)
        version = register_model(
            second.model_uri, "classifier", tags={"owner": "second"}
        )
        self.assertEqual(str(version.version), "2")
        self.assertEqual(version.tags["owner"], "second")
        self.assertEqual(snapshot(first_path), original)
        self.assertEqual(
            mlflow.MlflowClient().get_model_version("classifier", "1").tags["owner"],
            "first",
        )
        for number, info in ((1, first), (2, second)):
            uri = f"models:/classifier/{number}"
            metadata = mlflow.models.get_model_info(uri)
            self.assertEqual(metadata.model_id, info.model_id)
            self.assertEqual(metadata.flavors["triton"]["model_version"], number)
            repository = self.path(uri) / "model_repository/model"
            self.assertEqual(
                {p.name for p in repository.iterdir() if p.is_dir()}, {str(number)}
            )
            for native in (uri, str(repository / str(number) / "artifact")):
                np.testing.assert_array_equal(
                    mlflow.pyfunc.load_model(native).predict(np.ones((2, 1))),
                    [number] * 2,
                )
        third = self.log(3)
        version = register_model(third.model_uri, "classifier")
        self.assertEqual(str(version.version), "3")
        self.assertTrue(
            (self.path(third.model_uri) / "model_repository/model/3/model.py").is_file()
        )
        self.assertEqual(len(mlflow.MlflowClient().search_model_versions()), 3)
        # Registration uses the two/three existing training outputs, not extra logged models.
        self.assertEqual(
            len(
                mlflow.MlflowClient().search_logged_models(
                    experiment_ids=[self.experiment]
                )
            ),
            3,
        )
        self.assertEqual(snapshot(first_path), original)

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
        original = snapshot(self.path(first.model_uri))
        second = self.log(2)
        version = register_model(second.model_uri, "classifier")
        self.assertEqual(str(version.version), "2")
        self.assertEqual(snapshot(self.path(first.model_uri)), original)
        self.assertTrue(
            (
                self.path(second.model_uri) / "model_repository/model/2/model.py"
            ).is_file()
        )
        plain = self.log(3)
        mlflow.register_model(plain.model_uri, "other")
        original = snapshot(self.path(plain.model_uri))
        fourth = self.log(4)
        version = register_model(fourth.model_uri, "other")
        self.assertEqual(str(version.version), "2")
        self.assertEqual(snapshot(self.path(plain.model_uri)), original)

    def test_registered_source_is_copied_before_renumbering(self):
        first = self.log(1)
        register_model(first.model_uri, "classifier")
        original = snapshot(self.path(first.model_uri))
        version = register_model(first.model_uri, "classifier")
        self.assertEqual(str(version.version), "2")
        self.assertEqual(snapshot(self.path(first.model_uri)), original)
        info = mlflow.models.get_model_info("models:/classifier/2")
        self.assertNotEqual(info.model_id, first.model_id)
        self.assertEqual(info.flavors["triton"]["model_version"], 2)
        version = register_model("models:/classifier/2", "other")
        self.assertEqual(str(version.version), "1")
        info = mlflow.models.get_model_info("models:/other/1")
        repo = self.path("models:/other/1") / info.flavors["triton"]["data"]
        self.assertEqual({p.name for p in repo.iterdir() if p.is_dir()}, {"1"})
        np.testing.assert_array_equal(
            mlflow.pyfunc.load_model("models:/other/1").predict(np.ones((2, 1))), [1, 1]
        )
        self.assertEqual(snapshot(self.path(first.model_uri)), original)

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
