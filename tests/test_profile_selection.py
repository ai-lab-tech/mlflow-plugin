"""Name-only profile selection without exposing storage credentials."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mlflow.exceptions import MlflowException
from mlflow_triton_control.deployment_client import TritonControlDeploymentClient


class ProfileSelectionTests(unittest.TestCase):
    def setUp(self):
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
        self.config = {"image": "triton:test"}

    def test_name_is_resolved_and_only_internal_id_is_sent_to_deployment_api(self):
        with (
            patch.object(
                self.client,
                "_request",
                side_effect=[
                    {"id": 7, "name": "training"},
                    {"instance_id": 42},
                ],
            ) as request,
            patch.object(
                self.client, "get_deployment", return_value={"status": "ready"}
            ),
        ):
            self.client.create_deployment(
                "iris",
                "models:/m-123",
                config={**self.config, "s3_profile_name": " training "},
            )
        self.assertEqual(
            request.call_args_list[0].args, ("GET", "/api/s3-profiles/resolve")
        )
        self.assertEqual(
            request.call_args_list[0].kwargs["params"], {"name": "training"}
        )
        self.assertEqual(request.call_args_list[1].kwargs["json"]["s3_profile_id"], 7)

    def test_id_or_missing_name_is_rejected_without_api_requests(self):
        configurations = [
            {},
            {"s3_profile_id": 7},
            {"s3_profile_id": None, "s3_profile_name": "training"},
            {"s3_profile_id": 7, "s3_profile_name": "training"},
            {"s3_profile_name": ""},
            {"s3_profile_name": "   "},
            {"s3_profile_name": 7},
        ]
        for config in configurations:
            with (
                self.subTest(config=config),
                patch.object(self.client, "_request") as request,
                self.assertRaises(MlflowException),
            ):
                self.client.create_deployment(
                    "iris", "models:/m-123", config={**self.config, **config}
                )
            request.assert_not_called()

    def test_resolution_error_never_creates_deployment(self):
        for status in (403, 404):
            with (
                self.subTest(status=status),
                patch.object(
                    self.client,
                    "_request",
                    side_effect=MlflowException(f"HTTP {status}"),
                ) as request,
                self.assertRaisesRegex(MlflowException, str(status)),
            ):
                self.client.create_deployment(
                    "iris",
                    "models:/m-123",
                    config={**self.config, "s3_profile_name": "training"},
                )
            request.assert_called_once_with(
                "GET", "/api/s3-profiles/resolve", params={"name": "training"}
            )

    def test_invalid_reference_response_is_rejected(self):
        for profile in [
            {"id": 0, "name": "training"},
            {"id": True, "name": "training"},
            {"id": "7", "name": "training"},
            {"id": 7, "name": "other"},
            {},
        ]:
            with (
                self.subTest(profile=profile),
                patch.object(self.client, "_request", return_value=profile),
                self.assertRaisesRegex(MlflowException, "Invalid S3 profile"),
            ):
                self.client._resolve_s3_profile({"s3_profile_name": "training"})
