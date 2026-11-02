"""Optional deployment entry-point smoke test."""

import unittest
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
            if request.method == "POST":
                return httpx.Response(200, json={"instance_id": 42})
            if request.method == "DELETE":
                return httpx.Response(200, json={"message": "deleted"})
            deployment = {"name": "iris", "status": "ready", "instance_id": 42}
            body = [deployment] if request.url.path.endswith("/mlflow") else deployment
            return httpx.Response(200, json=body)

        real_client = httpx.Client
        with (
            patch.dict("os.environ", {"TRITON_CONTROL_TOKEN": "smoke-token"}),
            patch(
                "mlflow_triton_control.deployment_client.httpx.Client",
                side_effect=lambda **kwargs: real_client(
                    **kwargs, transport=httpx.MockTransport(handle)
                ),
            ),
        ):
            # Act
            client = get_deploy_client("triton-control://control.test")
            created = client.create_deployment(
                "iris",
                "s3://models/repository/iris",
                config={"s3_profile_id": 7, "image": "triton:latest"},
            )
            deployments = client.list_deployments()
            deleted = client.delete_deployment("iris")

        # Assert
        self.assertIsInstance(client, TritonControlDeploymentClient)
        self.assertEqual(created["flavor"], "triton")
        self.assertEqual(deployments, [created])
        self.assertEqual(deleted, {"message": "deleted"})
        for request in requests:
            self.assertEqual(request.headers["Authorization"], "Bearer smoke-token")
        self.assertEqual(
            [(request.method, request.url.path) for request in requests],
            [
                ("POST", "/api/deployments"),
                ("GET", "/api/deployments/mlflow/iris"),
                ("GET", "/api/deployments/mlflow"),
                ("DELETE", "/api/deployments/mlflow/iris"),
            ],
        )
