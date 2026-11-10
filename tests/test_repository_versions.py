"""Every logging call creates a snapshot; compatible configs advance Triton numbers."""

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
        (repository / "1/requirements.txt").write_text(name)
        metadata = Model()
        metadata.add_flavor(
            "triton", data="model_repository/model", model_name="model", model_version=1
        )
        metadata.save(package / "MLmodel")
        return package, repository

    def extend(self, package, previous, version="1", status="READY"):
        metadata = Model.load(previous / "MLmodel")
        metadata.model_id = "m-existing"

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
            update = _versions.extend_registered_repository(package, "registered")
            if update is not None:
                self.assertEqual(update.model_id, "m-existing")
                self.assertEqual(update.registered_model_version, int(version))
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
        self.assertEqual((latest / "3/requirements.txt").read_text(), "third")
        self.assertFalse((original / "2").exists())
        self.assertFalse((repository / "3").exists())
        second_flavor = Model.load(second / "MLmodel").flavors["triton"]
        third_flavor = Model.load(third / "MLmodel").flavors["triton"]
        self.assertEqual(
            second_flavor["repository_group"], third_flavor["repository_group"]
        )
        self.assertEqual(third_flavor["model_version"], 3)
        self.assertEqual(third_flavor["model_uri"], "models:/registered/2")

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
        self.assertNotIn("model_uri", flavor)

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
        self.assertEqual(flavor["model_uri"], "models:/registered/1")

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

    def test_explicit_alias_is_pinned_and_skips_registry_history(self):
        previous, _ = self.package("previous")
        package, repository = self.package("new")
        metadata = Model.load(previous / "MLmodel")
        metadata.model_id = "m-existing"
        with (
            patch("mlflow.models.get_model_info", return_value=metadata),
            patch.object(
                _versions.ModelsArtifactRepository,
                "get_underlying_uri",
                return_value=previous.as_uri(),
            ) as resolve,
            patch.object(_versions, "MlflowClient") as client,
        ):
            _versions.extend_registered_repository(
                package, model_uri="models:/source@champion"
            )
        resolve.assert_called_once_with("models:/source@champion")
        client.assert_not_called()
        self.assertEqual((repository / "2/requirements.txt").read_text(), "new")
        self.assertFalse((repository / "1").exists())

    def test_explicit_run_reference_uses_same_update_path(self):
        previous, _ = self.package("previous-run")
        package, repository = self.package("new-run")
        metadata = Model.load(previous / "MLmodel")
        metadata.model_id = "m-existing"
        with (
            patch("mlflow.models.get_model_info", return_value=metadata),
            patch.object(
                _versions.RunsArtifactRepository,
                "get_underlying_uri",
                return_value=previous.as_uri(),
            ) as resolve,
        ):
            _versions.extend_registered_repository(
                package, model_uri="runs:/run-id/model"
            )
        resolve.assert_called_once_with("runs:/run-id/model")
        self.assertEqual((repository / "2/requirements.txt").read_text(), "new-run")

    def test_invalid_explicit_reference_is_rejected(self):
        for index, uri in enumerate(("", 42, "file:///tmp/model", "s3://bucket/model")):
            package, _ = self.package(f"invalid-{index}")
            with (
                self.subTest(uri=uri),
                self.assertRaisesRegex(ValueError, "model_uri"),
            ):
                _versions.extend_registered_repository(package, model_uri=uri)

    def test_explicit_access_failure_does_not_fall_back_to_registry(self):
        package, _ = self.package("new")
        with (
            patch.object(
                _versions,
                "_previous_artifact_uri",
                side_effect=PermissionError("denied"),
            ),
            patch.object(_versions, "MlflowClient") as client,
            self.assertRaisesRegex(PermissionError, "denied"),
        ):
            _versions.extend_registered_repository(
                package, model_uri="models:/m-missing"
            )
        client.assert_not_called()

    def test_matching_legacy_model_without_logged_id_fails_before_upload(self):
        previous, _ = self.package("legacy-no-id")
        package, _ = self.package("new")
        with (
            patch.object(
                _versions, "_previous_artifact_uri", return_value=str(previous)
            ),
            self.assertRaisesRegex(ValueError, "no MLflow logged-model ID"),
        ):
            _versions.extend_registered_repository(
                package, model_uri="runs:/legacy/model"
            )

    def check_real_updates(self, registered):
        tracking, registry = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
        self.addCleanup(mlflow.set_tracking_uri, tracking)
        self.addCleanup(mlflow.set_registry_uri, registry)
        uri = "sqlite:///" + str(self.root / "tracking.db")
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
        experiment = mlflow.create_experiment(
            "versioning", artifact_location=(self.root / "artifacts").as_uri()
        )
        client = mlflow.MlflowClient()
        values = np.array([[1.0], [2.0]])
        # Same config, changed config, same again, then return to the first config.
        cases = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 1)]
        infos, snapshots, runs = [], [], []
        for index, (columns, number) in enumerate(cases, 1):
            example = np.repeat(values, columns, axis=1)
            model = DummyRegressor(strategy="constant", constant=float(index)).fit(
                example, [0.0, 0.0]
            )
            history = {"registered_model_name": "registered"} if registered else {}
            if not registered and infos:
                history["model_uri"] = infos[-1].model_uri
            requirements = ["scikit-learn"] + (["numpy"] if index % 2 == 0 else [])
            with mlflow.start_run(experiment_id=experiment) as run:
                info = sklearn.log_model(
                    model,
                    input_example=example,
                    pip_requirements=requirements,
                    **history,
                )
                runs.append(run.info.run_id)
            infos.append(info)
            self.assertEqual(
                info.registered_model_version,
                index if registered else None,
            )
            path = Path(
                mlflow.artifacts.download_artifacts(artifact_uri=info.model_uri)
            )
            flavor = Model.load(path / "MLmodel").flavors["triton"]
            self.assertEqual(flavor["model_version"], number)
            repository = path / flavor["data"]
            self.assertEqual(
                {p.name for p in repository.iterdir() if p.is_dir()},
                {str(number)},
            )
            self.assertFalse((repository / "requirements.txt").exists())
            self.assertEqual(
                "numpy" in (repository / str(number) / "requirements.txt").read_text(),
                index % 2 == 0,
            )
            np.testing.assert_array_equal(
                mlflow.pyfunc.load_model(info.model_uri).predict(example), [index] * 2
            )
            version_path = repository / str(number)
            snapshots.append(
                (
                    version_path,
                    {
                        f.relative_to(version_path): f.read_bytes()
                        for f in version_path.rglob("*")
                        if f.is_file()
                    },
                )
            )
            # Each run is associated with its own model snapshot.
            self.assertIn(
                info.model_id,
                [
                    m.model_id
                    for m in client.get_run(run.info.run_id).outputs.model_outputs
                ],
            )

        self.assertNotEqual(infos[0].model_id, infos[1].model_id)
        self.assertNotEqual(infos[2].model_id, infos[3].model_id)
        self.assertEqual(len({info.model_id for info in infos}), 5)
        self.assertEqual(
            len(client.search_logged_models(experiment_ids=[experiment])), 5
        )
        for index, ((version_path, snapshot), (columns, _)) in enumerate(
            zip(snapshots, cases), 1
        ):
            self.assertEqual(
                {
                    f.relative_to(version_path): f.read_bytes()
                    for f in version_path.rglob("*")
                    if f.is_file()
                },
                snapshot,
            )
            np.testing.assert_array_equal(
                mlflow.pyfunc.load_model(str(version_path / "artifact")).predict(
                    np.repeat(values, columns, axis=1)
                ),
                [index] * 2,
            )
        versions = client.search_model_versions()
        self.assertEqual(len(versions), 5 if registered else 0)
        if registered:
            for version, info in enumerate(infos, 1):
                entry = client.get_model_version("registered", str(version))
                self.assertEqual(
                    mlflow.models.get_model_info(
                        f"models:/registered/{version}"
                    ).model_id,
                    info.model_id,
                )
                self.assertEqual(entry.status, "READY")
            # Existing Registry versions retain their original weights.
            np.testing.assert_array_equal(
                mlflow.pyfunc.load_model("models:/registered/1").predict(values),
                [1.0] * 2,
            )
        else:
            # Branching from the original URI creates new, independent models.
            with mlflow.start_run(experiment_id=experiment):
                fourth = sklearn.log_model(
                    DummyRegressor(strategy="constant", constant=6.0).fit(
                        values, [0, 0]
                    ),
                    input_example=values,
                    model_uri=infos[0].model_uri,
                    pip_requirements=["scikit-learn"],
                )
                mlflow.log_metric("update_score", 0.75)
                repeated = sklearn.log_model(
                    DummyRegressor(strategy="constant", constant=7.0).fit(
                        values, [0, 0]
                    ),
                    input_example=values,
                    model_uri=infos[0].model_uri,
                    pip_requirements=["scikit-learn"],
                )
                update_run = mlflow.active_run().info.run_id
            self.assertNotEqual(repeated.model_id, fourth.model_id)
            self.assertEqual(repeated.flavors["triton"]["model_version"], 2)
            self.assertEqual(len(client.get_run(update_run).outputs.model_outputs), 2)
            self.assertTrue(
                any(
                    metric.key == "update_score" and metric.value == 0.75
                    for metric in client.get_logged_model(repeated.model_id).metrics
                )
            )
            self.assertNotEqual(fourth.model_id, infos[0].model_id)
            self.assertEqual(fourth.flavors["triton"]["model_version"], 2)
            self.assertFalse((snapshots[0][0].parent / "2").exists())

    def test_real_logged_model_history_without_registry(self):
        self.check_real_updates(registered=False)

    def test_real_registry_logs_updates_and_config_changes(self):
        self.check_real_updates(registered=True)


if __name__ == "__main__":
    unittest.main()
