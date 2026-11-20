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
            metadata={"triton_lineage": copy.deepcopy(FIXTURE["declaration"])},
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

    def test_conflicting_installation_fails_closed(self):
        # Arrange
        with patch.object(_model_uri, "_artifact_uri", return_value="s3://bucket/root"):
            # Act / Assert
            with self.assertRaises(MlflowException):
                self.client.export_lineage(
                    "models:/m-derived", tracking_installation_id="other"
                )

    def test_tampered_manifest_fails_closed(self):
        # Arrange
        self.manifest.write_text("{}")
        with patch.object(_model_uri, "_artifact_uri", return_value="s3://bucket/root"):
            # Act / Assert
            with self.assertRaises(MlflowException):
                self.client.export_lineage("models:/m-derived")

    def test_legacy_export_requires_explicit_installation(self):
        # Arrange
        self.metadata.metadata = {}
        self.metadata.flavors["triton"].pop("lineage")
        with patch.object(_model_uri, "_artifact_uri", return_value="s3://bucket/root"):
            # Act / Assert
            with self.assertRaises(MlflowException):
                self.client.export_lineage("models:/m-derived")

    def test_legacy_export_invents_no_derived_lineage(self):
        # Arrange
        self.metadata.metadata = {}
        self.metadata.flavors["triton"].pop("lineage")
        with patch.object(_model_uri, "_artifact_uri", return_value="s3://bucket/root"):
            # Act
            result = self.client.export_lineage(
                "models:/m-derived", tracking_installation_id="legacy-lab"
            )

        # Assert
        self.assertEqual(
            result,
            {
                "schema_version": 1,
                "provider": "mlflow",
                "origin": {
                    "installation_id": "legacy-lab",
                    "record_id": "models:/m-derived",
                },
            },
        )
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

    def test_ready_submission_requires_credentials(self):
        # Arrange / Act / Assert
        with (
            patch.dict(os.environ, {"TRITON_CONTROL_TOKEN": ""}),
            self.assertRaises(MlflowException),
        ):
            self.client.submit_ready({}, idempotency_key="attempt")

    def test_ready_submission_forwards_credentials_and_surfaces_denial(self):
        # Arrange
        response = httpx.Response(
            403,
            json={"detail": "Interactive credentials required"},
            request=httpx.Request("POST", "http://control.test"),
        )
        with patch("httpx.Client") as client:
            client.return_value.__enter__.return_value.request.return_value = response
            # Act / Assert
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

    def test_deployment_carries_lineage(self):
        # Arrange
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
            # Act
            result = self.client.create_deployment(
                "iris",
                "models:/m-derived",
                config={"image": "triton", "s3_profile_name": "Lab"},
            )
            # Assert
            self.assertEqual(
                request.call_args_list[1].kwargs["json"]["lineage"], FIXTURE["lineage"]
            )
            self.assertEqual(result["lineage"], FIXTURE["lineage"])

    def test_export_deployment_uses_recorded_lineage(self):
        # Arrange
        with patch.object(
            self.client, "_request", return_value={"lineage": FIXTURE["lineage"]}
        ) as request:
            # Act
            result = self.client.export_deployment_lineage("iris")

        # Assert
        self.assertEqual(result, FIXTURE["lineage"])
        request.assert_called_once_with("GET", "/api/deployments/mlflow/iris")
        self.download.assert_not_called()

    def test_export_legacy_deployment_does_not_invent_origin(self):
        # Arrange
        with patch.object(self.client, "_request", return_value={"status": "ready"}):
            # Act / Assert
            with self.assertRaisesRegex(MlflowException, "no recorded lineage"):
                self.client.export_deployment_lineage("iris")
        self.download.assert_not_called()

    def test_legacy_deployment_explicit_installation_preserves_origin(self):
        # Arrange
        self.metadata.metadata = {}
        self.metadata.flavors["triton"].pop("lineage")
        with (
            patch.object(
                _model_uri, "_artifact_uri", return_value="s3://models/root/model"
            ),
            patch.object(self.client, "_request") as request,
        ):
            request.side_effect = [
                {"id": 1, "name": "Lab"},
                {"instance_id": 1},
                {"status": "ready"},
            ]

            # Act
            self.client.create_deployment(
                "iris",
                "models:/m-derived",
                config={
                    "image": "triton",
                    "s3_profile_name": "Lab",
                    "tracking_installation_id": "legacy-lab",
                },
            )

        # Assert
        self.assertEqual(
            request.call_args_list[1].kwargs["json"].get("lineage"),
            {
                "schema_version": 1,
                "provider": "mlflow",
                "origin": {
                    "installation_id": "legacy-lab",
                    "record_id": "models:/m-derived",
                },
            },
        )
        self.download.assert_not_called()

    def test_conversion_requires_matching_installation_and_operation(self):
        for field, value in (
            ("installation_id", "another-lab"),
            ("operation", "validation"),
        ):
            with self.subTest(field=field):
                # Arrange
                lineage = copy.deepcopy(FIXTURE["lineage"])
                attempt = lineage["candidate"]["attempts"][0]
                (attempt["origin"] if field == "installation_id" else attempt)[
                    field
                ] = value

                # Act / Assert
                with self.assertRaises(MlflowException):
                    _lineage.validate(lineage)

    def test_candidate_cannot_be_its_own_predecessor(self):
        # Arrange
        lineage = copy.deepcopy(FIXTURE["lineage"])
        lineage["candidate"]["previous_serving_candidate"] = lineage["candidate"][
            "origin"
        ]
        lineage["previous_serving_model"]["candidate_origin"] = lineage["candidate"][
            "origin"
        ]

        # Act / Assert
        with self.assertRaises(MlflowException):
            _lineage.validate(lineage)

    def test_manifest_publication_identity_must_match_declaration(self):
        # Arrange
        self.metadata.metadata["triton_lineage"]["publication_operation_id"] = (
            "other-publication"
        )
        with patch.object(
            _model_uri, "_artifact_uri", return_value="s3://bucket/model"
        ):
            # Act / Assert
            with self.assertRaises(MlflowException):
                self.client.export_lineage("models:/m-derived")

    def test_manifest_inventory_must_match_flavor(self):
        # Arrange
        self.metadata.flavors["triton"]["inventory_digest"] = "sha256:" + "0" * 64
        with patch.object(
            _model_uri, "_artifact_uri", return_value="s3://bucket/model"
        ):
            # Act / Assert
            with self.assertRaises(MlflowException):
                self.client.export_lineage("models:/m-derived")

    def test_rehashed_invalid_inventory_is_not_accepted_as_valid_lineage(self):
        for mutation in (
            "unsafe-path",
            "duplicate-path",
            "negative-size",
            "boolean-size",
            "invalid-digest",
        ):
            with self.subTest(mutation=mutation):
                # Arrange
                document = copy.deepcopy(FIXTURE["manifest"])
                inventory = document["inventories"][-1]
                entry = inventory["files"][0]
                if mutation == "unsafe-path":
                    entry["path"] = "../outside"
                elif mutation == "duplicate-path":
                    inventory["files"].append(copy.deepcopy(entry))
                elif mutation == "negative-size":
                    entry["size"] = -1
                elif mutation == "boolean-size":
                    entry["size"] = True
                else:
                    entry["digest"] = "not-a-digest"
                checksum = _lineage.digest(inventory)
                document["candidate"]["output_inventory"] = checksum
                document["candidate"]["attempts"][-1]["output_inventory"] = checksum
                self.manifest.write_text(json.dumps(document))
                self.metadata.metadata["triton_lineage"].update(
                    digest=_lineage.digest(document),
                    candidate_digest=_lineage.digest(document["candidate"]),
                )
                with patch.object(
                    _model_uri, "_artifact_uri", return_value="s3://bucket/model"
                ):
                    # Act / Assert
                    with self.assertRaises(MlflowException):
                        self.client.export_lineage("models:/m-derived")

    def test_non_object_manifest_raises_adapter_error(self):
        # Arrange
        self.manifest.write_text("[]")
        with patch.object(
            _model_uri, "_artifact_uri", return_value="s3://bucket/model"
        ):
            # Act / Assert
            with self.assertRaises(MlflowException):
                self.client.export_lineage("models:/m-derived")


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
