"""MLflow reference resolution without downloading model weights."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow_triton_control import _model_uri as resolver


class ModelUriTests(unittest.TestCase):
    def setUp(self):
        self.metadata = SimpleNamespace(flavors={
            "triton": {"data": "model_repository/classifier", "model_name": "classifier"}
        })

    def test_models_and_runs_resolve_flavor_without_weight_download(self):
        for model_uri, repository in [
            ("models:/m-123", resolver.ModelsArtifactRepository),
            ("models:/iris/3", resolver.ModelsArtifactRepository),
            ("models:/iris@champion", resolver.ModelsArtifactRepository),
            ("runs:/run-123/model", resolver.RunsArtifactRepository),
        ]:
            with (
                self.subTest(model_uri=model_uri),
                patch.object(repository, "get_underlying_uri", return_value="s3://bucket/models/m-123") as resolve,
                patch.object(resolver, "get_model_info", return_value=self.metadata) as metadata,
                patch("mlflow.artifacts.download_artifacts", side_effect=AssertionError("weights download")),
            ):
                self.assertEqual(
                    resolver.resolve_model_uri(model_uri),
                    "s3://bucket/models/m-123/model_repository/classifier",
                )
                resolve.assert_called_once_with(model_uri)
                metadata.assert_called_once_with("s3://bucket/models/m-123")

    def test_nested_registry_run_reference(self):
        with (
            patch.object(resolver.ModelsArtifactRepository, "get_underlying_uri", return_value="runs:/run-123/model"),
            patch.object(resolver.RunsArtifactRepository, "get_underlying_uri", return_value="s3://bucket/run/model"),
            patch.object(resolver, "get_model_info", return_value=self.metadata),
        ):
            self.assertEqual(resolver.resolve_model_uri("models:/iris@champion"),
                             "s3://bucket/run/model/model_repository/classifier")

    def test_proxy_uses_mlflow_credentials_and_preserves_artifact_path(self):
        response = Mock()
        response.json.return_value = {"artifact_root": "s3://bucket/team/mlflow"}
        with (
            patch.object(resolver.ModelsArtifactRepository, "get_underlying_uri", return_value="mlflow-artifacts:/3/models/m-123/artifacts"),
            patch.object(resolver, "get_model_info", return_value=self.metadata),
            patch.object(mlflow, "get_tracking_uri", return_value="https://control/api/mlflow/tracking"),
            patch.object(resolver, "get_default_host_creds", return_value="tracking-creds") as creds,
            patch.object(resolver, "http_request", return_value=response) as request,
        ):
            self.assertEqual(resolver.resolve_model_uri("models:/m-123"),
                             "s3://bucket/team/mlflow/3/models/m-123/artifacts/model_repository/classifier")
            creds.assert_called_once_with("https://control/api/mlflow/tracking")
            self.assertEqual(request.call_args.args[0], "tracking-creds")
            self.assertEqual(request.call_args.args[1], "/api/2.0/mlflow/triton-control/artifact-store")

    def test_direct_storage_paths_are_rejected_without_artifact_access(self):
        for model_uri in ["s3://bucket/repository/classifier", "file:///tmp/classifier", "https://storage/classifier"]:
            with self.subTest(model_uri=model_uri), patch.object(resolver, "get_model_info") as metadata, patch.object(resolver, "_read_requirements") as read:
                with self.assertRaisesRegex(MlflowException, "models:/ or runs:/"):
                    resolver.resolve_model_uri(model_uri)
                with self.assertRaisesRegex(MlflowException, "models:/ or runs:/"):
                    resolver.resolve_deployment_artifacts(model_uri)
                metadata.assert_not_called()
                read.assert_not_called()

    def test_missing_flavor_or_unsafe_data_is_rejected(self):
        flavors = [
            {}, {"triton": {}}, {"triton": {"data": ""}},
            *({"triton": {"data": path}} for path in ["../outside", "/absolute", "model/../outside", "model/%2e%2e/outside", "model//name", "model/name?query", "model/name#fragment", "model\\name"]),
            {"triton": {"data": "model/other", "model_name": "classifier"}},
        ]
        for flavor in flavors:
            with (
                self.subTest(flavor=flavor),
                patch.object(resolver, "_artifact_uri", return_value="s3://bucket/model"),
                patch.object(resolver, "get_model_info", return_value=SimpleNamespace(flavors=flavor)),
                self.assertRaises(MlflowException),
            ):
                resolver.resolve_model_uri("models:/m-123")

    def test_only_generated_repository_layout_is_supported(self):
        for path in ["model/classifier", "model", "classifier", "model_repository", "model_repository/classifier/1", "prefix/model_repository/classifier"]:
            with (
                self.subTest(path=path),
                patch.object(resolver, "_artifact_uri", return_value="s3://bucket/model"),
                patch.object(resolver, "get_model_info", return_value=SimpleNamespace(flavors={"triton": {"data": path}})),
                patch.object(resolver, "_read_requirements") as read,
                self.assertRaisesRegex(MlflowException, "model_repository/<model-name>"),
            ):
                resolver.resolve_deployment_artifacts("models:/m-123")
            read.assert_not_called()

    def test_non_s3_artifact_storage_is_rejected(self):
        for location in ["file:///tmp/model", "gs://bucket/model", "https://other/model", "s3://user:password@bucket/model", "s3://bucket/../outside", "s3://bucket/model?query"]:
            with self.subTest(location=location), patch.object(resolver, "_artifact_uri", return_value=location), self.assertRaises(MlflowException):
                resolver.resolve_model_uri("models:/m-123")

    def test_unexpected_proxy_authority_and_non_s3_destination_are_rejected(self):
        with self.assertRaises(MlflowException):
            resolver._s3_location("mlflow-artifacts://other-host/model")
        response = Mock()
        response.json.return_value = {"artifact_root": "file:///tmp/artifacts"}
        with patch.object(mlflow, "get_tracking_uri", return_value="https://tracking"), patch.object(resolver, "http_request", return_value=response), self.assertRaises(MlflowException):
            resolver._s3_location("mlflow-artifacts:/3/model")

    def test_cyclic_model_reference_is_rejected(self):
        with patch.object(resolver.ModelsArtifactRepository, "get_underlying_uri", return_value="models:/iris@champion"), self.assertRaisesRegex(MlflowException, "Cyclic"):
            resolver.resolve_model_uri("models:/iris@champion")

    def test_real_mlflow_logged_id_registry_alias_and_run_resolution(self):
        tracking, registry = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
        try:
            with tempfile.TemporaryDirectory() as directory:
                uri = f"sqlite:///{Path(directory) / 'mlflow.db'}"
                mlflow.set_tracking_uri(uri)
                mlflow.set_registry_uri(uri)
                client = MlflowClient()
                experiment = client.create_experiment("uri-resolution", artifact_location="s3://bucket/experiment")
                run = client.create_run(experiment)
                client.create_registered_model("iris")
                version = client.create_model_version("iris", "s3://bucket/model-version", run_id=run.info.run_id)
                client.set_registered_model_alias("iris", "champion", version.version)
                self.assertEqual(resolver._artifact_uri("models:/iris@champion"), "s3://bucket/model-version")
                self.assertEqual(resolver._artifact_uri(f"runs:/{run.info.run_id}/model"), run.info.artifact_uri + "/model")
                logged = client.create_logged_model(experiment_id=experiment, name="classifier")
                self.assertEqual(resolver._artifact_uri(f"models:/{logged.model_id}"), logged.artifact_location)
        finally:
            mlflow.set_tracking_uri(tracking)
            mlflow.set_registry_uri(registry)
