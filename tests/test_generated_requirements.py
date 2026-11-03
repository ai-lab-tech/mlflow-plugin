"""Generated serving dependencies are read without downloading model weights."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlflow
from mlflow.exceptions import MlflowException
from mlflow_triton_control import _model_uri as resolver


class GeneratedRequirementsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repository = Path(self.temporary.name) / "classifier"
        self.repository.mkdir()
        self.uri = self.repository.as_uri()

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
            self.assertEqual(resolver._read_requirements(self.uri), expected)
        self.assertEqual(download.call_count, 1)
        self.assertEqual(
            download.call_args.kwargs["artifact_uri"], self.uri + "/requirements.txt"
        )
        self.assertFalse(Path(download.call_args.kwargs["dst_path"]).exists())

    def test_missing_file_is_optional_and_empty_file_is_preserved(self):
        self.assertIsNone(resolver._read_requirements(self.uri))
        (self.repository / "requirements.txt").write_text("")
        self.assertEqual(resolver._read_requirements(self.uri), "")

    def test_invalid_utf8_directory_or_oversized_file_fails(self):
        requirements = self.repository / "requirements.txt"
        for content in [b"\xff", b"x" * (1024 * 1024 + 1)]:
            requirements.write_bytes(content)
            with self.assertRaisesRegex(MlflowException, "Could not read"):
                resolver._read_requirements(self.uri)
        requirements.unlink()
        requirements.mkdir()
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
                resolver, "_read_requirements", return_value="scikit-learn==1.7.2\n"
            ) as read,
        ):
            artifacts = resolver.resolve_deployment_artifacts(
                "models:/classifier@champion"
            )
        alias.assert_called_once_with("models:/classifier@champion")
        info.assert_called_once_with("s3://bucket/models/pinned-version")
        read.assert_called_once_with(
            "s3://bucket/models/pinned-version/model_repository/classifier"
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
                resolver, "_read_requirements", return_value="mlflow==3.14.0\n"
            ) as read,
        ):
            result = resolver.resolve_deployment_artifacts("models:/m-123")
        read.assert_called_once_with(
            "mlflow-artifacts:/3/models/m-123/artifacts/model_repository/classifier"
        )
        self.assertEqual(result.requirements_txt, "mlflow==3.14.0\n")

    def test_direct_s3_input_is_rejected_before_reading_requirements(self):
        with (
            patch.object(resolver, "_read_requirements") as read,
            self.assertRaises(MlflowException),
        ):
            resolver.resolve_deployment_artifacts("s3://bucket/repository/classifier")
        read.assert_not_called()
