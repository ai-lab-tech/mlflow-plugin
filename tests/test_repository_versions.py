"""Registered updates download only configs and retain only new Triton model files."""

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, patch

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
        metadata.add_flavor(
            "triton", data="model_repository/model", model_name="model", model_version=1
        )
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

    def test_equal_configs_advance_versions_without_downloading_old_weights(self):
        # Arrange
        first, original = self.package("first")
        second, repository = self.package("second")
        third, latest = self.package("third")

        # Act
        downloads = self.extend(second, first)
        downloads.assert_called_once_with(
            artifact_uri="models:/registered/1/model_repository/model/config.pbtxt",
            dst_path=ANY,
        )
        downloads = self.extend(third, second, version="2")
        downloads.assert_called_once_with(
            artifact_uri="models:/registered/2/model_repository/model/config.pbtxt",
            dst_path=ANY,
        )

        # Assert
        for repo, number, expected in (
            (original, 1, "first"),
            (repository, 2, "second"),
            (latest, 3, "third"),
        ):
            self.assertEqual(
                {p.name for p in repo.iterdir() if p.is_dir()}, {str(number)}
            )
            self.assertEqual(
                (repo / str(number) / "artifact/weights").read_text(), expected
            )
            self.assertEqual((repo / str(number) / "model.py").read_text(), expected)
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

    def test_formatting_only_change_advances_version(self):
        # Arrange
        previous, _ = self.package("previous")
        package, repository = self.package(
            "new", 'name : "model" # comment\n\nbackend : "python"\n'
        )

        # Act
        self.extend(package, previous)

        # Assert
        self.assertFalse((repository / "1").exists())
        self.assertEqual((repository / "2/artifact/weights").read_text(), "new")

    def test_legacy_snapshot_uses_metadata_without_copying_old_versions(self):
        previous, history = self.package("previous")
        shutil.copytree(history / "1", history / "10")
        metadata = Model.load(previous / "MLmodel")
        metadata.flavors["triton"]["model_version"] = 10
        metadata.save(previous / "MLmodel")
        package, repository = self.package("new")

        downloads = self.extend(package, previous)

        self.assertEqual(downloads.call_count, 1)
        self.assertEqual({p.name for p in repository.iterdir() if p.is_dir()}, {"11"})
        self.assertEqual((repository / "11/artifact/weights").read_text(), "new")

    def test_missing_version_metadata_uses_legacy_version_one(self):
        previous, original = self.package("legacy")
        metadata = Model.load(previous / "MLmodel")
        del metadata.flavors["triton"]["model_version"]
        metadata.save(previous / "MLmodel")
        package, repository = self.package("new")

        downloads = self.extend(package, previous)

        downloads.assert_called_once_with(
            artifact_uri="models:/registered/1/model_repository/model/config.pbtxt",
            dst_path=ANY,
        )
        self.assertEqual({p.name for p in repository.iterdir() if p.is_dir()}, {"2"})
        self.assertEqual((repository / "2/artifact/weights").read_text(), "new")
        self.assertEqual((original / "1/artifact/weights").read_text(), "legacy")
        flavor = Model.load(package / "MLmodel").flavors["triton"]
        self.assertEqual(flavor["model_version"], 2)
        self.assertEqual(flavor["previous_model_uri"], "models:/registered/1")

    def test_invalid_previous_version_metadata_is_rejected(self):
        for number in (None, 0, -1, True, "2", 1.5):
            with self.subTest(number=number):
                previous, _ = self.package("previous-" + str(number))
                metadata = Model.load(previous / "MLmodel")
                metadata.flavors["triton"]["model_version"] = number
                metadata.save(previous / "MLmodel")
                package, repository = self.package("new-" + str(number))

                with self.assertRaisesRegex(ValueError, "Invalid Triton model_version"):
                    self.extend(package, previous)
                self.assertTrue((repository / "1/artifact/weights").is_file())

    def test_linked_previous_config_is_rejected(self):
        previous, history = self.package("previous")
        config = history / "config.pbtxt"
        external = self.root / "external-config"
        config.rename(external)
        config.symlink_to(external)
        package, repository = self.package("new")

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

    def test_previous_weights_are_not_required_for_update(self):
        previous, history = self.package("previous")
        shutil.rmtree(history / "1")
        package, repository = self.package("new")

        downloads = self.extend(package, previous)

        self.assertEqual(downloads.call_count, 1)
        self.assertEqual((repository / "2/artifact/weights").read_text(), "new")
        self.assertFalse((repository / "1").exists())

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
        cases = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 1)]
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
        for index, (info, example, (_, expected_version)) in enumerate(
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
            self.assertEqual(flavor["model_version"], expected_version)
            repository = path / flavor["data"]
            self.assertEqual(
                sorted(p.name for p in repository.iterdir() if p.is_dir()),
                [str(expected_version)],
            )
            saved = mlflow.pyfunc.load_model(
                str(repository / str(expected_version) / "artifact")
            )
            np.testing.assert_array_equal(saved.predict(example), [index] * 2)
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
