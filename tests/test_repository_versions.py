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
        Model().save(repository / "1/artifact/MLmodel")
        (repository / "config.pbtxt").write_text(config)
        (repository / "requirements.txt").write_text(name)
        metadata = Model()
        metadata.add_flavor("triton", data="model_repository/model", model_name="model")
        metadata.save(package / "MLmodel")
        return package, repository

    def extend(self, package, previous, version="1", status="READY"):
        metadata = Model.load(previous / "MLmodel")

        def download(artifact_uri, dst_path):
            relative = artifact_uri.split(f"/{version}/", 1)[1]
            return str(previous / relative)

        with (
            patch.object(_versions, "MlflowClient"),
            patch.object(
                _versions,
                "_latest_version",
                return_value=SimpleNamespace(version=version, status=status),
            ),
            patch("mlflow.models.get_model_info", return_value=metadata),
            patch(
                "mlflow.artifacts.download_artifacts", side_effect=download
            ) as downloads,
        ):
            _versions.extend_registered_repository(package, "registered")
        return downloads

    def test_equal_configs_accumulate_versions_with_new_weights_and_code(self):
        # Arrange
        first, original = self.package("first")
        second, repository = self.package("second")
        third, latest = self.package("third")

        # Act
        self.extend(second, first)
        self.extend(third, second, version="2")

        # Assert
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
        # Arrange
        first, _ = self.package("first")
        second, repository = self.package("second", 'name: "model"\nbackend: "other"\n')
        # Act
        downloads = self.extend(second, first)

        # Assert
        self.assertEqual(downloads.call_count, 1)
        self.assertFalse((repository / "2").exists())
        self.assertEqual((repository / "1/artifact/weights").read_text(), "second")
        flavor = Model.load(second / "MLmodel").flavors["triton"]
        self.assertEqual(flavor["model_version"], 1)
        self.assertNotIn("previous_model_uri", flavor)

    def test_config_comparison_ignores_formatting_but_keeps_contract_version(self):
        # Arrange
        configs = (
            'name: "model" # comment\nbackend: "python"',
            'name : "model"\n\nbackend : "python"',
            'string_value: "{\\"version\\":1}"',
            'string_value: "{\\"version\\":2}"',
        )

        # Act
        tokens = [_versions._config_tokens(config) for config in configs]

        # Assert
        self.assertEqual(tokens[0], tokens[1])
        self.assertNotEqual(tokens[2], tokens[3])

    def test_formatting_only_change_preserves_history(self):
        # Arrange
        previous, _ = self.package("previous")
        package, repository = self.package(
            "new", 'name : "model" # comment\n\nbackend : "python"\n'
        )

        # Act
        self.extend(package, previous)

        # Assert
        self.assertEqual((repository / "1/artifact/weights").read_text(), "previous")
        self.assertEqual((repository / "2/artifact/weights").read_text(), "new")

    def test_next_version_uses_numeric_maximum_even_with_gaps(self):
        # Arrange
        previous, history = self.package("previous")
        shutil.copytree(history / "1", history / "10")
        shutil.copytree(history / "1", history / "2")
        package, repository = self.package("new")

        # Act
        self.extend(package, previous)

        # Assert
        self.assertEqual(
            {p.name for p in repository.iterdir() if p.is_dir()},
            {"1", "2", "10", "11"},
        )
        self.assertEqual((repository / "11/artifact/weights").read_text(), "new")

    def test_invalid_version_names_are_rejected(self):
        for number in ("0", "01", "١"):
            with self.subTest(number=number):
                # Arrange
                previous, history = self.package("previous-" + number)
                (history / "1").rename(history / number)
                package, _ = self.package("new-" + number)

                # Act / Assert
                with self.assertRaisesRegex(ValueError, "invalid versions"):
                    self.extend(package, previous)

    def test_linked_history_assets_are_rejected(self):
        # Arrange
        previous, history = self.package("previous")
        external = self.root / "external"
        external.write_text("external model content")
        (history / "1/artifact/link").symlink_to(external)
        package, repository = self.package("new")

        # Act / Assert
        with self.assertRaisesRegex(ValueError, "symbolic links"):
            self.extend(package, previous)
        self.assertFalse((repository / "2").exists())

    def test_invalid_repository_path_is_rejected_before_artifact_access(self):
        # Arrange
        package, _ = self.package("new")
        with (
            patch.object(_versions, "MlflowClient"),
            patch.object(
                _versions,
                "_latest_version",
                return_value=SimpleNamespace(version="1", status="READY"),
            ),
            patch(
                "mlflow.models.get_model_info",
                return_value=SimpleNamespace(
                    flavors={"triton": {"data": "model_repository/.."}}
                ),
            ),
            patch("mlflow.artifacts.download_artifacts") as download,
        ):
            # Act / Assert
            with self.assertRaisesRegex(ValueError, "Invalid Triton repository path"):
                _versions.extend_registered_repository(package, "registered")
            download.assert_not_called()

    def test_history_access_failure_prevents_silent_reset(self):
        # Arrange
        package, _ = self.package("new")
        with (
            patch.object(_versions, "MlflowClient"),
            patch.object(
                _versions, "_latest_version", side_effect=PermissionError("registry")
            ),
        ):
            # Act / Assert
            with self.assertRaisesRegex(PermissionError, "registry"):
                _versions.extend_registered_repository(package, "registered")

    def test_previous_native_model_starts_fresh(self):
        # Arrange
        package, repository = self.package("new")
        with (
            patch.object(_versions, "MlflowClient"),
            patch.object(
                _versions,
                "_latest_version",
                return_value=SimpleNamespace(version="4", status="READY"),
            ),
            patch(
                "mlflow.models.get_model_info", return_value=SimpleNamespace(flavors={})
            ),
            patch("mlflow.artifacts.download_artifacts") as download,
        ):
            # Act
            _versions.extend_registered_repository(package, "registered")
        # Assert
        download.assert_not_called()
        self.assertTrue((repository / "1/artifact/weights").is_file())
        self.assertEqual(
            Model.load(package / "MLmodel").flavors["triton"]["model_version"], 1
        )

    def test_artifact_access_failure_prevents_logging(self):
        # Arrange
        package, _ = self.package("new")
        metadata = Model.load(package / "MLmodel")
        with (
            patch.object(_versions, "MlflowClient"),
            patch.object(
                _versions,
                "_latest_version",
                return_value=SimpleNamespace(version="1", status="READY"),
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
        ):
            # Act / Assert
            with self.assertRaisesRegex(PermissionError, "artifacts"):
                sklearn.log_model(
                    object(),
                    input_example=np.ones((1, 1)),
                    registered_model_name="registered",
                )
        log.assert_not_called()

    def test_missing_history_versions_are_rejected(self):
        # Arrange
        previous, repository = self.package("previous")
        shutil.rmtree(repository / "1")
        package, _ = self.package("new")
        # Act / Assert
        with self.assertRaisesRegex(ValueError, "invalid versions"):
            self.extend(package, previous)

    def test_incomplete_history_is_rejected_before_changing_new_repository(self):
        for missing in ("model.py", "artifact/MLmodel"):
            with self.subTest(missing=missing):
                # Arrange
                previous, history = self.package(
                    "previous-" + missing.replace("/", "-")
                )
                (history / "1" / missing).unlink()
                package, repository = self.package("new-" + missing.replace("/", "-"))

                # Act / Assert
                with self.assertRaisesRegex(ValueError, "Incomplete Triton version"):
                    self.extend(package, previous)
                self.assertTrue((repository / "1/artifact/weights").is_file())
                self.assertFalse((repository / "2").exists())

    def test_unready_registration_is_not_used_as_history(self):
        for status in ("PENDING_REGISTRATION", "FAILED_REGISTRATION"):
            with self.subTest(status=status):
                # Arrange
                previous, _ = self.package("previous-" + status)
                package, repository = self.package("new-" + status)

                # Act / Assert
                with self.assertRaisesRegex(ValueError, status):
                    self.extend(package, previous, status=status)
                self.assertFalse((repository / "2").exists())

    def test_real_registry_logs_updates_and_config_changes(self):
        # Arrange: returning to an older config must not resurrect its history.
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
        cases = [(1, [1]), (1, [1, 2]), (2, [3]), (2, [3, 4]), (1, [5])]
        examples = [np.repeat(values, columns, axis=1) for columns, _ in cases]
        models = [
            DummyRegressor(strategy="constant", constant=float(index)).fit(
                example, [0.0, 0.0]
            )
            for index, example in enumerate(examples, 1)
        ]

        # Act
        infos = []
        with mlflow.start_run(experiment_id=experiment):
            for model, example in zip(models, examples):
                infos.append(
                    sklearn.log_model(
                        model,
                        input_example=example,
                        name="model",
                        registered_model_name="registered",
                        pip_requirements=["scikit-learn"],
                    )
                )

        # Assert
        paths, groups = [], []
        for index, (info, example, (_, expected_predictions)) in enumerate(
            zip(infos, examples, cases), 1
        ):
            self.assertEqual(str(info.registered_model_version), str(index))
            path = Path(
                mlflow.artifacts.download_artifacts(
                    artifact_uri=f"models:/registered/{index}"
                )
            )
            paths.append(path)
            flavor = Model.load(path / "MLmodel").flavors["triton"]
            groups.append(flavor["repository_group"])
            self.assertEqual(flavor["model_version"], len(expected_predictions))
            repository = path / flavor["data"]
            self.assertEqual(
                sorted(p.name for p in repository.iterdir() if p.is_dir()),
                [str(n) for n in range(1, len(expected_predictions) + 1)],
            )
            for number, expected_prediction in enumerate(expected_predictions, 1):
                saved = mlflow.pyfunc.load_model(
                    str(repository / str(number) / "artifact")
                )
                np.testing.assert_array_equal(
                    saved.predict(example), [expected_prediction] * 2
                )
            np.testing.assert_array_equal(
                mlflow.pyfunc.load_model(info.model_uri).predict(example), [index] * 2
            )
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[2], groups[3])
        self.assertNotEqual(groups[1], groups[2])
        self.assertNotIn(groups[4], groups[:4])
        self.assertFalse((paths[0] / "model_repository/model/2").exists())


if __name__ == "__main__":
    unittest.main()
