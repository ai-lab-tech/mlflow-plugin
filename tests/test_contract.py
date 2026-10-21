"""Exercise the installed plugin against the current upstream contract."""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from hypothesis import find, settings
from hypothesis_jsonschema import from_schema
from jsonschema import Draft202012Validator
from mlflow.deployments import get_deploy_client


class ContractTests(unittest.TestCase):
    def test_deployment_lifecycle_from_upstream_contract(self):
        contract_path = os.environ.get("TRITON_CONTROL_CONTRACT")
        if not contract_path:
            self.skipTest("Set TRITON_CONTROL_CONTRACT; see README contract tests")
        contract = json.loads(Path(contract_path).read_text())
        api = contract["openapi"]
        requests = []
        responses = []
        states = iter(["starting", "ready"])

        def operation(request):
            path = request.url.path
            parameters = {}
            if path.startswith("/api/deployments/mlflow/"):
                parameters["name"] = path.rsplit("/", 1)[1]
                path = "/api/deployments/mlflow/{name}"
            # Missing/changed endpoints or methods must fail, never fall back.
            op = api["paths"][path][request.method.lower()]
            for parameter in op.get("parameters", []):
                if parameter["in"] == "path":
                    validate(parameters[parameter["name"]], parameter["schema"])
            if "requestBody" in op:
                schema = op["requestBody"]["content"]["application/json"]["schema"]
                validate(json.loads(request.content), schema)
            return op

        def schema_with_refs(schema):
            return {**schema, "components": api.get("components", {})}

        def validate(value, schema):
            Draft202012Validator(schema_with_refs(schema)).validate(value)

        def handle(request):
            requests.append(request)
            op = operation(request)
            successes = [code for code in op["responses"] if code.startswith("2")]
            self.assertTrue(successes, "No successful response in upstream contract")
            code = successes[0]
            schema = op["responses"][code]["content"]["application/json"]["schema"]
            if request.method == "GET":
                state = next(states) if request.url.path.endswith("/iris") else "ready"
                body = contract["descriptions"][state]
                if request.url.path.endswith("/mlflow"):
                    body = [body]
            else:
                # Include only fields generated from the live upstream schema.
                body = find(
                    from_schema(schema_with_refs(schema)),
                    lambda value: True,
                    settings=settings(database=None, derandomize=True),
                )
            validate(body, schema)
            responses.append(body)
            return httpx.Response(int(code), json=body)

        real_client = httpx.Client
        with (
            patch.dict(os.environ, {"TRITON_CONTROL_TOKEN": "contract-token"}),
            patch(
                "mlflow_triton_control.deployment_client.httpx.Client",
                side_effect=lambda **kw: real_client(
                    **kw, transport=httpx.MockTransport(handle)
                ),
            ),
            patch("time.sleep"),
        ):
            client = get_deploy_client("triton-control://control.test")
            created = client.create_deployment(
                "iris",
                "s3://models/repository/iris",
                config={"s3_profile_id": 7, "image": "triton:latest"},
            )
            listed = client.list_deployments()
            deleted = client.delete_deployment("iris")
        self.assertEqual(
            created, {**contract["descriptions"]["ready"], "flavor": "triton"}
        )
        self.assertEqual(listed, [created])
        self.assertEqual(deleted, responses[-1])
        self.assertEqual(len(requests), 5)
        for request in requests:
            self.assertEqual(request.headers["Authorization"], "Bearer contract-token")
