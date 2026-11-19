"""Control adapter identities, explicit submission and storage-independent export."""

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from mlflow.exceptions import MlflowException
from mlflow_triton_control import _lineage, _model_uri
from mlflow_triton_control.deployment_client import TritonControlDeploymentClient

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/control-lineage.json").read_text()
)


class ControlLineageTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(
            patch.dict(
                os.environ,
                {
                    "TRITON_CONTROL_URL": "http://control.test",
                    "TRITON_CONTROL_TOKEN": "fixture-token",
                },
            )
        )
        self.client = TritonControlDeploymentClient("triton-control")
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.manifest = self.root / "lineage.json"
        self.manifest.write_text(json.dumps(FIXTURE["manifest"]))
        self.metadata = SimpleNamespace(
            model_id="m-derived",
            metadata={"triton_lineage": FIXTURE["declaration"]},
            flavors={
                "triton": {"data": "model_repository/iris", "lineage": "lineage.json"}
            },
        )
        self.enterContext(
            patch.object(_model_uri, "get_model_info", return_value=self.metadata)
        )
        self.enterContext(
            patch.object(
                _lineage.mlflow.artifacts,
                "list_artifacts",
                return_value=[
                    SimpleNamespace(
                        path="lineage.json",
                        is_dir=False,
                        file_size=self.manifest.stat().st_size,
                    )
                ],
            )
        )
        self.download = self.enterContext(
            patch.object(
                _lineage.mlflow.artifacts,
                "download_artifacts",
                return_value=str(self.manifest),
            )
        )
        self.enterContext(
            patch.object(_model_uri, "_read_version_requirements", return_value=None)
        )

    def test_export_preserves_all_origins_across_storage_locations(self):
        for location in (
            "s3://first/model",
            "s3://second/moved",
            "file:///tmp/model",
            "mlflow-artifacts:/models/model",
        ):
            with (
                self.subTest(location=location),
                patch.object(_model_uri, "_artifact_uri", return_value=location),
            ):
                result = self.client.export_lineage("models:/serving@ready")
                self.assertEqual(result, FIXTURE["lineage"])
                self.download.assert_called_with(
                    artifact_uri=location + "/lineage.json", dst_path=unittest.mock.ANY
                )

    def test_tampered_manifest_and_conflicting_installation_fail_closed(self):
        with patch.object(_model_uri, "_artifact_uri", return_value="s3://bucket/root"):
            with self.assertRaises(MlflowException):
                self.client.export_lineage(
                    "models:/m-derived", tracking_installation_id="other"
                )
            self.manifest.write_text("{}")
            with self.assertRaises(MlflowException):
                self.client.export_lineage("models:/m-derived")

    def test_legacy_export_requires_explicit_installation_and_invents_no_lineage(self):
        self.metadata.metadata = {}
        self.metadata.flavors["triton"].pop("lineage")
        with patch.object(_model_uri, "_artifact_uri", return_value="s3://bucket/root"):
            with self.assertRaises(MlflowException):
                self.client.export_lineage("models:/m-derived")
            result = self.client.export_lineage(
                "models:/m-derived", tracking_installation_id="legacy-lab"
            )
            self.assertEqual(
                result["origin"],
                {"installation_id": "legacy-lab", "record_id": "models:/m-derived"},
            )
            self.assertNotIn("candidate", result)
            self.download.assert_not_called()

    def test_prepare_is_explicit_and_retries_do_not_resolve_alias_again(self):
        with patch.object(
            _model_uri, "_artifact_uri", return_value="s3://models/allowed/run/model"
        ) as resolve:
            with patch.object(
                self.client,
                "_request",
                side_effect=[
                    {"id": 1, "name": "Lab"},
                    [{"id": 1, "name": "Lab", "bucket": "models", "prefix": "allowed"}],
                ],
            ) as request:
                prepared = self.client.prepare_handoff(
                    "models:/serving@ready",
                    s3_profile_name="Lab",
                    revision="1",
                    title="Iris",
                )
                self.assertEqual(
                    [call.args[0] for call in request.call_args_list], ["GET", "GET"]
                )
            self.assertEqual(prepared["source"]["path"], "run/model/model_repository")
            self.assertEqual(prepared["origin"], FIXTURE["lineage"]["origin"])
            frozen = copy.deepcopy(prepared)
            with patch.object(
                self.client,
                "_request",
                side_effect=[MlflowException("Lost response"), {"id": "same-job"}],
            ) as request:
                with self.assertRaises(MlflowException):
                    self.client.submit_ready(prepared, idempotency_key="attempt-1")
                result = self.client.submit_ready(prepared, idempotency_key="attempt-1")
                self.assertEqual(result["id"], "same-job")
                self.assertEqual(request.call_args_list[0], request.call_args_list[1])
                self.assertEqual(prepared, frozen)
            resolve.assert_called_once()

    def test_ready_authentication_and_authorization_errors_are_not_bypassed(self):
        with (
            patch.dict(os.environ, {"TRITON_CONTROL_TOKEN": ""}),
            self.assertRaises(MlflowException),
        ):
            self.client.submit_ready({}, idempotency_key="attempt")
        response = httpx.Response(
            403,
            json={"detail": "Interactive credentials required"},
            request=httpx.Request("POST", "http://control.test"),
        )
        with patch("httpx.Client") as client:
            client.return_value.__enter__.return_value.request.return_value = response
            with self.assertRaisesRegex(MlflowException, "403"):
                self.client.submit_ready({}, idempotency_key="attempt")
            headers = (
                client.return_value.__enter__.return_value.request.call_args.kwargs[
                    "headers"
                ]
            )
            self.assertEqual(headers["Idempotency-Key"], "attempt")
            self.assertEqual(headers["Authorization"], "Bearer fixture-token")

    def test_production_approval_is_not_part_of_the_lineage_contract(self):
        with self.assertRaises(MlflowException):
            _lineage.validate({**FIXTURE["lineage"], "production_approval": True})

    def test_deployment_carries_and_exports_lineage(self):
        with (
            patch.object(
                _model_uri, "_artifact_uri", return_value="s3://models/allowed/model"
            ),
            patch.object(self.client, "_request") as request,
        ):
            request.side_effect = [
                {"id": 1, "name": "Lab"},
                {"instance_id": 1},
                {"status": "ready", "lineage": FIXTURE["lineage"]},
            ]
            result = self.client.create_deployment(
                "iris",
                "models:/m-derived",
                config={"image": "triton", "s3_profile_name": "Lab"},
            )
            self.assertEqual(
                request.call_args_list[1].kwargs["json"]["lineage"], FIXTURE["lineage"]
            )
            self.assertEqual(result["lineage"], FIXTURE["lineage"])
            request.side_effect = [{"lineage": result["lineage"]}]
            self.assertEqual(
                self.client.export_deployment_lineage("iris"), FIXTURE["lineage"]
            )


class PublishedModelLineageTests(unittest.TestCase):
    def test_real_publication_and_relocated_export_keep_the_same_origins(self):
        import shutil

        import mlflow
        from test_derived_serving import ServingTests

        publication = ServingTests(
            "test_distinct_model_measured_lineage_and_read_only_source"
        )
        self.addCleanup(publication.doCleanups)
        publication.setUp()
        with patch.object(
            TritonControlDeploymentClient,
            "_request",
            side_effect=AssertionError("Logging must not call Control"),
        ):
            info = publication.log()
            original = _model_uri.export_lineage(info.model_uri)
        self.assertEqual(
            original["origin"],
            {"installation_id": "target", "record_id": info.model_uri},
        )
        self.assertEqual(original["source"]["model_uri"], publication.source)
        self.assertEqual(original["conversion_run_id"], publication.conversion)
        source = mlflow.artifacts.download_artifacts(artifact_uri=info.model_uri)
        moved = publication.root / "relocated-model"
        shutil.copytree(source, moved)
        with patch.object(_model_uri, "_artifact_uri", return_value=moved.as_uri()):
            self.assertEqual(_model_uri.export_lineage(info.model_uri), original)
