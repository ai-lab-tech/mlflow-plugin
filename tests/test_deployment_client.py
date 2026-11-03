"""Deployment client checks using real MLflow without a remote server."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mlflow.exceptions import MlflowException
from mlflow_triton_control._model_uri import DeploymentArtifacts
from mlflow_triton_control.deployment_client import TritonControlDeploymentClient


class DeploymentClientTests(unittest.TestCase):
    def setUp(self) -> None:
        requirements = patch(
            "mlflow_triton_control._model_uri._read_requirements", return_value=None
        )
        requirements.start()
        self.addCleanup(requirements.stop)
        backend = patch.dict(
            "os.environ", {"TRITON_CONTROL_URL": "http://control.test"}
        )
        backend.start()
        self.addCleanup(backend.stop)
        self.client = TritonControlDeploymentClient("triton-control")
        for dependency, value in [
            ("_artifact_uri", "s3://models/dev"),
            (
                "get_model_info",
                SimpleNamespace(
                    flavors={"triton": {"data": "model_repository/iris_classifier"}}
                ),
            ),
        ]:
            mocked = patch(
                "mlflow_triton_control._model_uri." + dependency, return_value=value
            )
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_create_derives_repository_from_model_uri_and_waits_for_model(self) -> None:
        # Arrange
        with (
            patch.object(self.client, "_request") as request,
            patch.dict(os.environ, {"TRITON_CONTROL_TOKEN": "token"}),
        ):
            request.side_effect = [
                {"id": 7, "name": "training"},
                {"instance_id": 42},
                {"name": "iris", "status": "starting"},
                {"name": "iris", "status": "ready"},
            ]
            # Act
            with patch("time.sleep"):
                result = self.client.create_deployment(
                    "iris",
                    "models:/m-123",
                    config={"s3_profile_name": "training", "image": "triton:latest"},
                )
        # Assert
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["flavor"], "triton")
        payload = request.call_args_list[1].kwargs["json"]
        self.assertEqual(payload["repository_prefix"], "dev/model_repository")
        self.assertEqual(payload["model_name"], "iris_classifier")

    def test_mlflow_model_uri_resolves_before_create(self):
        config = {"s3_profile_name": "training", "image": "triton:latest"}
        with (
            patch(
                "mlflow_triton_control.deployment_client.resolve_deployment_artifacts",
                return_value=DeploymentArtifacts(
                    "s3://bucket/logged/model_repository/classifier",
                    "scikit-learn==1.7.2\n",
                ),
            ) as resolve,
            patch.object(
                self.client,
                "_request",
                side_effect=[{"id": 7, "name": "training"}, {"instance_id": 42}],
            ) as request,
            patch.object(
                self.client, "get_deployment", return_value={"status": "ready"}
            ),
        ):
            self.client.create_deployment("iris", "models:/m-123", config=config)
        resolve.assert_called_once_with("models:/m-123")
        payload = request.call_args.kwargs["json"]
        self.assertEqual(
            payload["model_uri"], "s3://bucket/logged/model_repository/classifier"
        )
        self.assertEqual(payload["repository_prefix"], "logged/model_repository")
        self.assertEqual(payload["model_name"], "classifier")
        self.assertEqual(payload["requirements_txt"], "scikit-learn==1.7.2\n")

    def test_resolution_failure_does_not_create_deployment(self):
        with (
            patch(
                "mlflow_triton_control.deployment_client.resolve_deployment_artifacts",
                side_effect=MlflowException("no triton flavor"),
            ),
            patch.object(
                self.client, "_request", return_value={"id": 7, "name": "training"}
            ) as request,
            self.assertRaisesRegex(MlflowException, "no triton flavor"),
        ):
            self.client.create_deployment(
                "iris",
                "models:/m-123",
                config={"s3_profile_name": "training", "image": "triton:latest"},
            )
        request.assert_called_once_with(
            "GET", "/api/s3-profiles/resolve", params={"name": "training"}
        )

    def test_invalid_timeout_is_rejected_before_create(self) -> None:
        for timeout in ("later", "nan", "inf", "-inf", 0, -1):
            with self.subTest(timeout=timeout):
                # Arrange
                self.client._request = Mock()
                config = {
                    "s3_profile_name": "training",
                    "image": "triton:latest",
                    "wait_timeout_seconds": timeout,
                }

                # Act / Assert
                with self.assertRaisesRegex(
                    MlflowException, "wait_timeout_seconds must be a positive number"
                ):
                    self.client.create_deployment(
                        "iris", "models:/m-123", config=config
                    )
                self.client._request.assert_not_called()

    def test_target_rejects_ignored_credentials_query_and_fragment(self):
        # Arrange
        targets = (
            "triton-control://user:password@control.test",
            "triton-control://control.test?option=value",
            "triton-control://control.test#fragment",
            "triton-control:///missing-host",
        )

        # Act / Assert
        for target in targets:
            with self.subTest(target=target), self.assertRaises(MlflowException):
                TritonControlDeploymentClient(target)


class DeploymentTargetEnvironmentTests(unittest.TestCase):
    def test_short_target_uses_injected_backend_url(self):
        for target in ["triton-control"]:
            with (
                self.subTest(target=target),
                patch.dict(
                    os.environ, {"TRITON_CONTROL_URL": "https://custom-control:9443/"}
                ),
            ):
                client = TritonControlDeploymentClient(target)
                self.assertEqual(client._base_url, "https://custom-control:9443")

    def test_only_plugin_name_is_accepted_even_with_valid_environment(self):
        for target in [
            "triton-control://",
            "triton-control://explicit-service:8000",
            "http://control.test:8000",
            "triton-control/",
            "other-plugin",
        ]:
            with (
                self.subTest(target=target),
                patch.dict(
                    os.environ, {"TRITON_CONTROL_URL": "https://control.test:9443"}
                ),
                self.assertRaisesRegex(MlflowException, "Only target"),
            ):
                TritonControlDeploymentClient(target)

    def test_short_target_requires_backend_url(self):
        with (
            patch.dict(os.environ, {"TRITON_CONTROL_URL": ""}),
            self.assertRaisesRegex(MlflowException, "TRITON_CONTROL_URL is required"),
        ):
            TritonControlDeploymentClient("triton-control")

    def test_invalid_backend_environment_is_rejected(self):
        for url in [
            "ftp://service",
            "http://user:password@service",
            "http://service/api",
            "http://service?x=y",
            "http://service#fragment",
            "http://service:bad",
            "http://service:99999",
            "http://bad host",
            "http:///missing-host",
        ]:
            with (
                self.subTest(url=url),
                patch.dict(os.environ, {"TRITON_CONTROL_URL": url}),
                self.assertRaisesRegex(MlflowException, "TRITON_CONTROL_URL must be"),
            ):
                TritonControlDeploymentClient("triton-control")
