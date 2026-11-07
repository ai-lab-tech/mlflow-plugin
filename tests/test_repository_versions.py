"""Registered history preserves Triton versions using only config compatibility."""

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlflow
import numpy as np
from mlflow.models import Model
from mlflow_triton import _versions, sklearn
from sklearn.dummy import DummyRegressor


class RepositoryVersionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def package(self, name, config='name: "model"\nbackend: "python"\n'):
        package = self.root / name
        repository = package / "model_repository/model"
        (repository / "1/artifact").mkdir(parents=True)
        (repository / "1/artifact/weights").write_text(name)
        (repository / "1/model.py").write_text(name)
        (repository / "config.pbtxt").write_text(config)
        (repository / "requirements.txt").write_text(name)
        metadata = Model()
        metadata.add_flavor("triton", data="model_repository/model", model_name="model")
        metadata.save(package / "MLmodel")
        return package, repository

    def extend(self, package, previous, version="1"):
        metadata = Model.load(previous / "MLmodel")

        def download(artifact_uri, dst_path):
            relative = artifact_uri.split(f"/{version}/", 1)[1]
            return str(previous / relative)

        with (
            patch.object(_versions, "MlflowClient"),
            patch.object(
                _versions,
                "_latest_version",
                return_value=SimpleNamespace(version=version),
            ),
            patch("mlflow.models.get_model_info", return_value=metadata),
            patch(
                "mlflow.artifacts.download_artifacts", side_effect=download
            ) as downloads,
        ):
            _versions.extend_registered_repository(package, "registered")
        return downloads

    def test_equal_configs_accumulate_versions_with_new_weights_and_code(self):
        first, original = self.package("first")
        second, repository = self.package("second")
        self.extend(second, first)
        third, latest = self.package("third")
        self.extend(third, second, version="2")
        for number, expected in ((1, "first"), (2, "second"), (3, "third")):
            self.assertEqual(
                (latest / str(number) / "artifact/weights").read_text(), expected
            )
            self.assertEqual((latest / str(number) / "model.py").read_text(), expected)
        self.assertEqual((latest / "requirements.txt").read_text(), "third")
        self.assertFalse((original / "2").exists())
        self.assertFalse((repository / "3").exists())
        second_flavor = Model.load(second / "MLmodel").flavors["triton"]
        third_flavor = Model.load(third / "MLmodel").flavors["triton"]
        self.assertEqual(
            second_flavor["repository_group"], third_flavor["repository_group"]
        )
        self.assertEqual(third_flavor["model_version"], 3)
        self.assertEqual(third_flavor["previous_model_uri"], "models:/registered/2")

    def test_changed_config_starts_at_one_without_downloading_weights(self):
        first, _ = self.package("first")
        second, repository = self.package("second", 'name: "model"\nbackend: "other"\n')
        downloads = self.extend(second, first)
        self.assertEqual(downloads.call_count, 1)
        self.assertFalse((repository / "2").exists())
        self.assertEqual((repository / "1/artifact/weights").read_text(), "second")
        flavor = Model.load(second / "MLmodel").flavors["triton"]
        self.assertEqual(flavor["model_version"], 1)
        self.assertNotIn("previous_model_uri", flavor)

    def test_config_comparison_ignores_formatting_but_keeps_contract_version(self):
        self.assertEqual(
            _versions._config_tokens('name: "model" # comment\nbackend: "python"'),
            _versions._config_tokens('name : "model"\n\nbackend : "python"'),
        )
        self.assertNotEqual(
            _versions._config_tokens('string_value: "{\\"version\\":1}"'),
            _versions._config_tokens('string_value: "{\\"version\\":2}"'),
        )

    def test_history_access_failure_prevents_silent_reset(self):
        package, _ = self.package("new")
        with (
            patch.object(_versions, "MlflowClient"),
            patch.object(
                _versions, "_latest_version", side_effect=PermissionError("registry")
            ),
            self.assertRaisesRegex(PermissionError, "registry"),
        ):
            _versions.extend_registered_repository(package, "registered")

    def test_previous_native_model_starts_fresh(self):
        package, repository = self.package("new")
        with (
            patch.object(_versions, "MlflowClient"),
            patch.object(
                _versions, "_latest_version", return_value=SimpleNamespace(version="4")
            ),
            patch(
                "mlflow.models.get_model_info", return_value=SimpleNamespace(flavors={})
            ),
            patch("mlflow.artifacts.download_artifacts") as download,
        ):
            _versions.extend_registered_repository(package, "registered")
        download.assert_not_called()
        self.assertTrue((repository / "1/artifact/weights").is_file())
        self.assertEqual(
            Model.load(package / "MLmodel").flavors["triton"]["model_version"], 1
        )

    def test_artifact_access_failure_prevents_logging(self):
        package, _ = self.package("new")
        metadata = Model.load(package / "MLmodel")
        with (
            patch.object(_versions, "MlflowClient"),
            patch.object(
                _versions, "_latest_version", return_value=SimpleNamespace(version="1")
            ),
            patch("mlflow.models.get_model_info", return_value=metadata),
            patch(
                "mlflow.artifacts.download_artifacts",
                side_effect=PermissionError("artifacts"),
            ),
            patch(
                "mlflow_triton._logging.prepare_model",
                side_effect=lambda *args, **kwargs: shutil.copytree(package, args[2]),
            ),
            patch.object(Model, "log") as log,
            self.assertRaisesRegex(PermissionError, "artifacts"),
        ):
            sklearn.log_model(
                object(),
                input_example=np.ones((1, 1)),
                registered_model_name="registered",
            )
        log.assert_not_called()

    def test_missing_history_versions_are_rejected(self):
        previous, repository = self.package("previous")
        shutil.rmtree(repository / "1")
        package, _ = self.package("new")
        with self.assertRaisesRegex(ValueError, "invalid versions"):
            self.extend(package, previous)

    def test_real_registry_logs_updates_and_config_changes(self):
        tracking, registry = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
        self.addCleanup(mlflow.set_tracking_uri, tracking)
        self.addCleanup(mlflow.set_registry_uri, registry)
        uri = "sqlite:///" + str(self.root / "tracking.db")
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
        experiment = mlflow.create_experiment(
            "versioning", artifact_location=(self.root / "artifacts").as_uri()
        )
        values = np.array([[1.0], [2.0]])
        paths, groups = [], []
        with mlflow.start_run(experiment_id=experiment):
            for index, columns in enumerate((1, 1, 2, 2), 1):
                example = np.repeat(values, columns, axis=1)
                model = DummyRegressor(strategy="constant", constant=float(index))
                model.fit(example, np.array([0.0, 0.0]))
                info = sklearn.log_model(
                    model,
                    input_example=example,
                    name="model",
                    registered_model_name="registered",
                    pip_requirements=["scikit-learn"],
                )
                self.assertEqual(str(info.registered_model_version), str(index))
                path = Path(
                    mlflow.artifacts.download_artifacts(artifact_uri=info.model_uri)
                )
                paths.append(path)
                metadata = Model.load(path / "MLmodel")
                flavor = metadata.flavors["triton"]
                groups.append(flavor["repository_group"])
                expected_number = 1 if index in (1, 3) else 2
                self.assertEqual(flavor["model_version"], expected_number)
                repository = path / flavor["data"]
                self.assertEqual(
                    sorted(p.name for p in repository.iterdir() if p.is_dir()),
                    [str(n) for n in range(1, expected_number + 1)],
                )
                for number in range(1, expected_number + 1):
                    saved = mlflow.pyfunc.load_model(
                        str(repository / str(number) / "artifact")
                    )
                    expected_prediction = index - expected_number + number
                    np.testing.assert_array_equal(
                        saved.predict(example), [expected_prediction] * 2
                    )
                np.testing.assert_array_equal(
                    mlflow.pyfunc.load_model(info.model_uri).predict(example),
                    [index] * 2,
                )
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[2], groups[3])
        self.assertNotEqual(groups[1], groups[2])
        self.assertFalse((paths[0] / "model_repository/model/2").exists())


if __name__ == "__main__":
    unittest.main()
