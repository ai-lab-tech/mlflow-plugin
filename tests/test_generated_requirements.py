"""Generated serving dependencies are read without downloading model weights."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlflow
import numpy as np
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.utils.file_utils import local_file_uri_to_path
from mlflow_triton import sklearn
from mlflow_triton._requirements import serving_requirements
from mlflow_triton_control import _model_uri as resolver
from mlflow_triton_control.deployment_client import TritonControlDeploymentClient
from packaging.requirements import Requirement
from sklearn.dummy import DummyRegressor


class CapturedDependencyRegressor(DummyRegressor):
    def predict(self, values):
        if self.constant == 2:
            import yaml

            yaml.safe_load("serving: true")
        return super().predict(values)


class GeneratedRequirementsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repository = Path(self.temporary.name) / "classifier"
        self.repository.mkdir()
        self.uri = self.repository.as_uri()

    def test_captured_requirements_follow_registered_versions_into_deployment(self):
        tracking, registry = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
        self.addCleanup(mlflow.set_tracking_uri, tracking)
        self.addCleanup(mlflow.set_registry_uri, registry)
        root = Path(self.temporary.name)
        uri = "sqlite:///" + str(root / "tracking.db")
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)
        experiment = mlflow.create_experiment(
            "capture-versions", artifact_location=(root / "artifacts").as_uri()
        )
        values = np.array([[1.0], [2.0]])
        infos = []
        requirements = []
        for number in (1, 2):
            model = CapturedDependencyRegressor(
                strategy="constant", constant=number
            ).fit(values, [0, 0])
            with mlflow.start_run(experiment_id=experiment):
                info = sklearn.log_model(
                    model,
                    input_example=values,
                    registered_model_name="captured",
                    serialization_format="cloudpickle",
                )
            infos.append(info)
            self.assertEqual(int(info.registered_model_version), number)
            self.assertEqual(info.flavors["triton"]["model_version"], number)
            path = Path(mlflow.artifacts.download_artifacts(artifact_uri=info.model_uri))
            repository = path / "model_repository/model"
            self.assertFalse((repository / "requirements.txt").exists())
            content = (repository / str(number) / "requirements.txt").read_text()
            self.assertEqual("pyyaml==" in content.lower(), number == 2)
            requirements.append(content)

        self.assertNotEqual(infos[0].model_id, infos[1].model_id)
        owner = Path(mlflow.artifacts.download_artifacts(artifact_uri=infos[0].model_uri))
        resolve_uri = resolver._artifact_uri

        def local_uri(uri):
            # Exercise real Registry resolution while using local test storage.
            return local_file_uri_to_path(resolve_uri(uri))

        for number in (1, 2):
            self.assertEqual(
                (owner / f"model_repository/model/{number}/requirements.txt").read_text(),
                requirements[number - 1],
            )
            with (
                patch.object(resolver, "_artifact_uri", side_effect=local_uri),
                patch.object(resolver, "_s3_location", return_value="s3://bucket/owner"),
            ):
                artifacts = resolver.resolve_deployment_artifacts(
                    f"models:/captured/{number}"
                )
            self.assertEqual(artifacts.requirements_txt, requirements[number - 1])
            self.assertEqual(
                artifacts.repository_uri, "s3://bucket/owner/model_repository/model"
            )

    def test_reads_generated_file_with_real_mlflow_artifact_api(self):
        expected = "mlflow==3.14.0\nscikit-learn==1.7.2\ncloudpickle==3.1.2\n"
        (self.repository / "requirements.txt").write_text(expected)
        (self.repository / "1").mkdir()
        (self.repository / "1/model.onnx").write_bytes(
            b"weights must not be downloaded"
        )
        with patch.object(
            mlflow.artifacts,
            "download_artifacts",
            wraps=mlflow.artifacts.download_artifacts,
        ) as download:
            # Act
            actual = resolver._read_requirements(self.uri)

        # Assert
        self.assertEqual(actual, expected)
        self.assertEqual(download.call_count, 1)
        self.assertEqual(
            download.call_args.kwargs["artifact_uri"], self.uri + "/requirements.txt"
        )
        self.assertFalse(Path(download.call_args.kwargs["dst_path"]).exists())

    def test_flattened_core_requirements_reach_deployment_without_sidecar_access(self):
        # Arrange
        native = self.repository / "native"
        native.mkdir()
        (native / "requirements.txt").write_text("numpy>=1.20\n-c constraints.txt\n")
        (native / "constraints.txt").write_text("numpy<2\n")
        repository = self.repository / "model_repository/classifier"
        (repository / "2").mkdir(parents=True)
        generated = serving_requirements(native, "3.14.0")
        (repository / "2/requirements.txt").write_text(generated)
        metadata = Model()
        metadata.add_flavor(
            "triton",
            data="model_repository/classifier",
            model_name="classifier",
            model_version=2,
        )
        metadata.save(self.repository / "MLmodel")
        # Only the standalone serving file remains available to artifact access.
        (native / "constraints.txt").unlink()
        (native / "requirements.txt").unlink()
        with (
            patch.dict("os.environ", {"TRITON_CONTROL_URL": "http://control.test"}),
            patch.object(resolver, "_artifact_uri", return_value=str(self.repository)),
            patch.object(resolver, "_s3_location", return_value="s3://bucket/logged"),
            patch.object(
                TritonControlDeploymentClient,
                "_request",
                side_effect=[
                    {"id": 7, "name": "training"},
                    {"instance_id": 42},
                    {"name": "demo", "status": "ready"},
                ],
            ) as request,
        ):
            client = TritonControlDeploymentClient("triton-control")

            # Act
            deployment = client.create_deployment(
                "demo",
                "models:/m-123",
                config={"s3_profile_name": "training", "image": "triton:test"},
            )

        # Assert
        payload = request.call_args_list[1].kwargs["json"]
        self.assertEqual(payload["requirements_txt"], generated)
        # Triton Control validates each line with packaging.Requirement as well.
        requirements = [
            Requirement(line) for line in payload["requirements_txt"].splitlines()
        ]
        numpy = next(item for item in requirements if item.name == "numpy")
        self.assertIn("1.26.4", numpy.specifier)
        self.assertNotIn("2.0", numpy.specifier)
        self.assertNotIn(".txt", payload["requirements_txt"])
        self.assertEqual(deployment["status"], "ready")

    def test_version_local_requirements_take_precedence_and_legacy_is_supported(self):
        (self.repository / "requirements.txt").write_text("legacy")
        (self.repository / "2").mkdir()
        local = self.repository / "2/requirements.txt"
        local.write_text("version-two")
        self.assertEqual(
            resolver._read_version_requirements(self.uri, {"model_version": 2}),
            "version-two",
        )
        local.write_text("")
        self.assertEqual(
            resolver._read_version_requirements(self.uri, {"model_version": 2}), ""
        )
        local.unlink()
        self.assertEqual(
            resolver._read_version_requirements(self.uri, {"model_version": 2}),
            "legacy",
        )
        self.assertEqual(resolver._read_version_requirements(self.uri, {}), "legacy")

    def test_invalid_version_cannot_select_requirements_path(self):
        for number in (0, -1, True, "../outside", None):
            with (
                self.subTest(number=number),
                self.assertRaisesRegex(MlflowException, "model_version"),
            ):
                resolver._read_version_requirements(self.uri, {"model_version": number})

    def test_missing_file_is_optional(self):
        # Arrange: setUp creates a repository without requirements.txt.

        # Act
        requirements = resolver._read_requirements(self.uri)

        # Assert
        self.assertIsNone(requirements)

    def test_empty_file_is_preserved(self):
        # Arrange
        (self.repository / "requirements.txt").write_text("")

        # Act
        requirements = resolver._read_requirements(self.uri)

        # Assert
        self.assertEqual(requirements, "")

    def test_invalid_utf8_or_oversized_file_fails(self):
        for content in [b"\xff", b"x" * (1024 * 1024 + 1)]:
            with self.subTest(size=len(content)):
                # Arrange
                (self.repository / "requirements.txt").write_bytes(content)

                # Act / Assert
                with self.assertRaisesRegex(MlflowException, "Could not read"):
                    resolver._read_requirements(self.uri)

    def test_requirements_directory_is_rejected(self):
        # Arrange
        (self.repository / "requirements.txt").mkdir()

        # Act / Assert
        with self.assertRaises(MlflowException):
            resolver._read_requirements(self.uri)

    def test_artifact_permission_errors_are_not_ignored(self):
        with (
            patch.object(
                mlflow.artifacts,
                "list_artifacts",
                side_effect=PermissionError("denied"),
            ),
            self.assertRaisesRegex(MlflowException, "Could not read"),
        ):
            resolver._read_requirements(self.uri)

    def test_registry_alias_is_resolved_once_for_metadata_and_requirements(self):
        metadata = SimpleNamespace(
            flavors={"triton": {"data": "model_repository/classifier"}}
        )
        with (
            patch.object(
                resolver.ModelsArtifactRepository,
                "get_underlying_uri",
                return_value="s3://bucket/models/pinned-version",
            ) as alias,
            patch.object(resolver, "get_model_info", return_value=metadata) as info,
            patch.object(
                resolver,
                "_read_version_requirements",
                return_value="scikit-learn==1.7.2\n",
            ) as read,
        ):
            artifacts = resolver.resolve_deployment_artifacts(
                "models:/classifier@champion"
            )
        alias.assert_called_once_with("models:/classifier@champion")
        info.assert_called_once_with("s3://bucket/models/pinned-version")
        read.assert_called_once_with(
            "s3://bucket/models/pinned-version/model_repository/classifier",
            metadata.flavors["triton"],
        )
        self.assertEqual(
            artifacts.repository_uri,
            "s3://bucket/models/pinned-version/model_repository/classifier",
        )
        self.assertEqual(artifacts.requirements_txt, "scikit-learn==1.7.2\n")

    def test_managed_proxy_reads_requirements_through_authenticated_mlflow_uri(self):
        metadata = SimpleNamespace(
            flavors={"triton": {"data": "model_repository/classifier"}}
        )
        with (
            patch.object(
                resolver,
                "_artifact_uri",
                return_value="mlflow-artifacts:/3/models/m-123/artifacts",
            ),
            patch.object(
                resolver,
                "_s3_location",
                return_value="s3://bucket/mlflow/3/models/m-123/artifacts",
            ),
            patch.object(resolver, "get_model_info", return_value=metadata),
            patch.object(
                resolver, "_read_version_requirements", return_value="mlflow==3.14.0\n"
            ) as read,
        ):
            result = resolver.resolve_deployment_artifacts("models:/m-123")
        read.assert_called_once_with(
            "mlflow-artifacts:/3/models/m-123/artifacts/model_repository/classifier",
            metadata.flavors["triton"],
        )
        self.assertEqual(result.requirements_txt, "mlflow==3.14.0\n")

    def test_direct_s3_input_is_rejected_before_reading_requirements(self):
        with (
            patch.object(resolver, "_read_requirements") as read,
            self.assertRaises(MlflowException),
        ):
            resolver.resolve_deployment_artifacts("s3://bucket/repository/classifier")
        read.assert_not_called()
