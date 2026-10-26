"""Resolve MLflow models to S3 repositories without copying model artifacts."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mlflow.exceptions import MlflowException

from mlflow_triton_control import triton


class ModelUriTests(unittest.TestCase):
    def setUp(self):
        self.info = SimpleNamespace(
            model_id="m-example",
            flavors={"triton": {"data": "model/iris_classifier"}},
        )

    def test_s3_repository_uses_logged_model_location_and_flavor(self):
        with patch.object(
            triton.mlflow,
            "get_logged_model",
            return_value=SimpleNamespace(
                artifact_location="s3://bucket/experiment/models/m-example/artifacts"
            ),
        ):
            self.assertEqual(
                triton._repository_s3_uri(self.info),
                "s3://bucket/experiment/models/m-example/artifacts/model/iris_classifier",
            )

    def test_proxy_repository_uses_server_root_and_tracking_auth(self):
        response = Mock()
        response.json.return_value = {"artifact_root": "s3://bucket/mlflow"}
        with (
            patch.object(
                triton.mlflow,
                "get_logged_model",
                return_value=SimpleNamespace(
                    artifact_location="mlflow-artifacts:/7/models/m-example/artifacts"
                ),
            ),
            patch.object(
                triton.mlflow, "get_tracking_uri", return_value="http://tracking.test"
            ),
            patch.object(triton.httpx, "get", return_value=response) as get,
            patch.dict(os.environ, {"MLFLOW_TRACKING_TOKEN": "test-token"}),
        ):
            self.assertEqual(
                triton._repository_s3_uri(self.info),
                "s3://bucket/mlflow/7/models/m-example/artifacts/model/iris_classifier",
            )
        self.assertEqual(
            get.call_args.args[0],
            "http://tracking.test/api/2.0/mlflow/triton-control/artifact-store",
        )
        self.assertEqual(
            get.call_args.kwargs["headers"], {"Authorization": "Bearer test-token"}
        )

    def test_non_s3_store_remains_loggable_but_not_deployable(self):
        with (
            patch.object(
                triton.mlflow,
                "get_logged_model",
                return_value=SimpleNamespace(artifact_location="file:///tmp/artifacts"),
            ),
            patch.object(triton, "get_model_info", return_value=self.info),
        ):
            self.assertIsNone(triton._repository_s3_uri(self.info))
            with self.assertRaisesRegex(MlflowException, "S3-backed"):
                triton._resolve_repository_uri("models:/m-example")

    def test_non_triton_flavor_and_path_traversal_are_rejected(self):
        for flavors in ({"sklearn": {}}, {"triton": {"data": "../model"}}):
            with self.subTest(flavors=flavors), self.assertRaises(MlflowException):
                triton._repository_s3_uri(SimpleNamespace(flavors=flavors))

    def test_supported_mlflow_references_delegate_resolution_to_mlflow(self):
        for uri in (
            "models:/m-example",
            "models:/iris/2",
            "models:/iris@candidate",
            "runs:/run/model",
        ):
            with (
                self.subTest(uri=uri),
                patch.object(
                    triton, "get_model_info", return_value=self.info
                ) as get_info,
                patch.object(
                    triton, "_repository_s3_uri", return_value="s3://bucket/repo/iris"
                ),
            ):
                self.assertEqual(
                    triton._resolve_repository_uri(uri), "s3://bucket/repo/iris"
                )
                get_info.assert_called_once_with(uri)

    def test_direct_s3_repository_does_not_query_mlflow(self):
        with patch.object(triton, "get_model_info") as get_info:
            self.assertEqual(
                triton._resolve_repository_uri("s3://bucket/repo/iris"),
                "s3://bucket/repo/iris",
            )
            get_info.assert_not_called()
