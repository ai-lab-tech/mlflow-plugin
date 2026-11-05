"""External deployment options and administrator policy."""

import unittest
from unittest.mock import Mock

from mlflow.exceptions import MlflowException
from mlflow_triton_control._deployment_access import access_payload


class AccessTests(unittest.TestCase):
    def test_domains_rejected_regardless_of_base_domain(self):
        for domain in ("", "models.example.com"):
            for host in ("model.example.com", "model.example.com.", "sub.model"):
                with self.subTest(base_domain=domain, host=host):
                    request = Mock(
                        return_value={"ingress_enabled": True, "base_domain": domain}
                    )
                    with self.assertRaisesRegex(MlflowException, "short hostname only"):
                        access_payload({"ingress_host": host}, request)
                    request.assert_not_called()

    def test_short_hostname_passed_without_appending_domain(self):
        for domain in ("", "models.example.com"):
            with self.subTest(base_domain=domain):
                request = Mock(
                    return_value={"ingress_enabled": True, "base_domain": domain}
                )
                result = access_payload({"ingress_host": " Smollm "}, request)
                self.assertEqual(result["ingress_host"], "smollm")
                request.assert_called_once_with("GET", "/api/deployments/policy")

    def test_internal_default_does_not_query_policy(self):
        # Arrange
        request = Mock()
        # Act
        result = access_payload({}, request)
        # Assert
        self.assertEqual(result, {"access_mode": "internal"})
        request.assert_not_called()

    def test_https_and_cli_auth_fields(self):
        # Arrange
        request = Mock(return_value={"ingress_enabled": True, "allow_http": False})
        config = {
            "ingress_host": "model",
            "basic_auth_enabled": "true",
            "basic_auth_username": "user",
            "basic_auth_password": "secret",
        }
        # Act
        result = access_payload(config, request)
        # Assert
        self.assertEqual(result["ingress_scheme"], "https")
        self.assertTrue(result["basic_auth_enabled"])
        self.assertEqual(result["basic_auth_password"], "secret")
        request.assert_called_once_with("GET", "/api/deployments/policy")

    def test_disabled_ingress_rejects_https(self):
        # Arrange
        request = Mock(return_value={"ingress_enabled": False, "allow_http": True})
        # Act / Assert
        with self.assertRaisesRegex(MlflowException, "Ingress is disabled"):
            access_payload({"ingress_host": "model"}, request)

    def test_http_url_is_rejected(self):
        # Arrange
        request = Mock(return_value={"ingress_enabled": True, "allow_http": False})
        # Act / Assert
        with self.assertRaisesRegex(MlflowException, "hostname only"):
            access_payload({"ingress_host": "http://model"}, request)

    def test_permitted_http(self):
        # Arrange
        request = Mock(return_value={"ingress_enabled": True, "allow_http": True})
        # Act
        result = access_payload(
            {
                "ingress_host": "model",
                "ingress_scheme": "http",
                "basic_auth_enabled": "false",
            },
            request,
        )
        # Assert
        self.assertEqual(result["ingress_scheme"], "http")
        self.assertFalse(result["basic_auth_enabled"])

    def test_http_basic_auth_rejected_before_policy_call(self):
        # Arrange
        request = Mock()
        # Act / Assert
        with self.assertRaisesRegex(MlflowException, "requires HTTPS"):
            access_payload(
                {
                    "ingress_host": "model",
                    "ingress_scheme": "http",
                    "basic_auth_enabled": True,
                    "basic_auth_username": "user",
                    "basic_auth_password": "secret",
                },
                request,
            )
        request.assert_not_called()

    def test_class_cannot_be_overridden(self):
        # Arrange
        request = Mock()
        # Act / Assert
        with self.assertRaisesRegex(MlflowException, "administrator"):
            access_payload({"ingress_class_name": "nginx"}, request)

    def test_partial_credentials_rejected(self):
        # Arrange
        request = Mock()
        # Act / Assert
        with self.assertRaisesRegex(MlflowException, "both username and password"):
            access_payload(
                {
                    "ingress_host": "model",
                    "basic_auth_enabled": True,
                    "basic_auth_username": "user",
                },
                request,
            )

    def test_create_forwards_exposure_after_policy_check(self):
        # Arrange
        from types import SimpleNamespace
        from unittest.mock import patch

        from mlflow_triton_control.deployment_client import (
            TritonControlDeploymentClient,
        )

        with (
            patch.dict("os.environ", {"TRITON_CONTROL_URL": "https://control.test"}),
            patch(
                "mlflow_triton_control.deployment_client.resolve_deployment_artifacts",
                return_value=SimpleNamespace(
                    repository_uri="s3://models/repository/classifier",
                    requirements_txt=None,
                ),
            ),
        ):
            client = TritonControlDeploymentClient("triton-control")
            with (
                patch.object(client, "_resolve_s3_profile", return_value=1),
                patch.object(
                    client,
                    "_request",
                    side_effect=[
                        {"ingress_enabled": True, "allow_http": False},
                        {"instance_id": 1},
                        {"status": "ready", "external_url": "https://model.test"},
                    ],
                ) as request,
            ):
                # Act
                result = client.create_deployment(
                    "model",
                    "models:/model-id",
                    config={
                        "image": "triton:latest",
                        "s3_profile_name": "training",
                        "ingress_host": "model",
                        "basic_auth_enabled": True,
                        "basic_auth_username": "user",
                        "basic_auth_password": "secret",
                    },
                )
        # Assert
        self.assertEqual(
            request.call_args_list[0].args, ("GET", "/api/deployments/policy")
        )
        payload = request.call_args_list[1].kwargs["json"]
        self.assertEqual(payload["ingress_host"], "model")
        self.assertEqual(payload["basic_auth_password"], "secret")
        self.assertEqual(result["external_url"], "https://model.test")
        self.assertNotIn("basic_auth_password", result)
