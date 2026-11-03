"""Optional deployment entry-point smoke test."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from mlflow.deployments import get_deploy_client
from mlflow_triton_control import TritonControlDeploymentClient


class DeploymentSmokeTests(unittest.TestCase):
    def test_installed_entry_point_and_deployment_lifecycle(self):
        # Arrange
        requests = []

        def handle(request):
            requests.append(request)
            if request.url.path == "/api/s3-profiles/resolve":
                return httpx.Response(200, json={"id": 7, "name": "training"})
            if request.method == "POST":
                return httpx.Response(200, json={"instance_id": 42})
            if request.method == "DELETE":
                return httpx.Response(200, json={"message": "deleted"})
            deployment = {"name": "iris", "status": "ready", "instance_id": 42}
            body = [deployment] if request.url.path.endswith("/mlflow") else deployment
            return httpx.Response(200, json=body)

        real_client = httpx.Client
        with (
            patch(
                "mlflow_triton_control._model_uri._artifact_uri",
                return_value="s3://models/repository",
            ),
            patch(
                "mlflow_triton_control._model_uri.get_model_info",
                return_value=SimpleNamespace(
                    flavors={"triton": {"data": "model_repository/iris"}}
                ),
            ),
            patch(
                "mlflow_triton_control._model_uri._read_requirements",
                return_value="scikit-learn==1.7.2\n",
            ),
            patch.dict(
                "os.environ",
                {
                    "TRITON_CONTROL_TOKEN": "smoke-token",
                    "TRITON_CONTROL_URL": "http://control.test:8000",
                },
            ),
            patch(
                "mlflow_triton_control.deployment_client.httpx.Client",
                side_effect=lambda **kwargs: real_client(
                    **kwargs, transport=httpx.MockTransport(handle)
                ),
            ),
        ):
            # Act
            client = get_deploy_client("triton-control")
            created = client.create_deployment(
                "iris",
                "models:/m-123",
                config={"s3_profile_name": "training", "image": "triton:latest"},
            )
            deployments = client.list_deployments()
            deleted = client.delete_deployment("iris")

        # Assert
        self.assertEqual(requests[0].url.params["name"], "training")
        self.assertEqual(
            json.loads(requests[1].content)["requirements_txt"],
            "scikit-learn==1.7.2\n",
        )
        self.assertIsInstance(client, TritonControlDeploymentClient)
        self.assertEqual(created["flavor"], "triton")
        self.assertEqual(deployments, [created])
        self.assertEqual(deleted, {"message": "deleted"})
        for request in requests:
            self.assertEqual(request.headers["Authorization"], "Bearer smoke-token")
        self.assertEqual(
            [(request.method, request.url.path) for request in requests],
            [
                ("GET", "/api/s3-profiles/resolve"),
                ("POST", "/api/deployments"),
                ("GET", "/api/deployments/mlflow/iris"),
                ("GET", "/api/deployments/mlflow"),
                ("DELETE", "/api/deployments/mlflow/iris"),
            ],
        )
