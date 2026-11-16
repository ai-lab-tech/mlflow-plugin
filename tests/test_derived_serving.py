"""Prepared model provenance and publication recovery against real MLflow stores."""

import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import mlflow
from mlflow.models import Model
from mlflow_triton import serving
from mlflow_triton._publication import AmbiguousPublication
from mlflow_triton._serving_artifacts import capture, digest


class ServingTests(unittest.TestCase):
    def setUp(self):
        try:
            from tritonclient.grpc import model_config_pb2  # noqa: F401
        except ImportError:
            self.skipTest(
                "Install mlflow-triton[serving] for prepared-repository tests"
            )
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.addCleanup(mlflow.set_tracking_uri, mlflow.get_tracking_uri())
        self.addCleanup(mlflow.set_registry_uri, mlflow.get_registry_uri())
        self.uri = "sqlite:///" + str(self.root / "mlflow.db")
        mlflow.set_tracking_uri(self.uri)
        mlflow.set_registry_uri(self.uri)
        self.client = mlflow.MlflowClient()
        self.experiment = self.client.create_experiment(
            "serving", artifact_location=(self.root / "artifacts").as_uri()
        )
        self.training = self.client.create_run(self.experiment).info.run_id
        self.conversion = self.client.create_run(self.experiment).info.run_id
        self.source = self.make_source(
            self.client, self.experiment, self.training, "original"
        )
        self.repo = self.root / "repository"
        (self.repo / "iris/1").mkdir(parents=True)
        (self.repo / "iris/config.pbtxt").write_text(
            'name: "iris"\nbackend: "onnxruntime"\n'
        )
        (self.repo / "iris/1/model.onnx").write_bytes(b"converted bytes")
        self.database = os.getenv("SERVING_TEST_DATABASE_URL") or "sqlite:///" + str(
            self.root / "receipts.db"
        )

    def make_source(self, client, experiment, run, weights):
        logged = client.create_logged_model(
            experiment, name="scientist", source_run_id=run
        )
        with tempfile.TemporaryDirectory() as path:
            Model(model_id=logged.model_id, run_id=run, flavors={"test": {}}).save(
                Path(path) / "MLmodel"
            )
            (Path(path) / "weights.bin").write_text(weights)
            client.log_model_artifacts(logged.model_id, path)
        client.finalize_logged_model(logged.model_id, "READY")
        return "models:/" + logged.model_id

    def log(self, **kwargs):
        return serving.log_model(
            **{
                "source_model_uri": self.source,
                "triton_repository_path": self.repo,
                "conversion_run_id": self.conversion,
                "publication_database_url": self.database,
                "tracking_installation_id": "target",
                "source_tracking_installation_id": "science",
                **kwargs,
            }
        )

    def lineage(self, info):
        path = mlflow.artifacts.download_artifacts(artifact_uri=info.model_uri)
        return json.loads((Path(path) / "lineage.json").read_text())

    def test_distinct_model_measured_lineage_and_read_only_source(self):
        source = Path(
            self.client.get_logged_model(
                self.source.split("/")[-1]
            ).artifact_location.removeprefix("file://")
        )
        before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
        result = self.log(registered_model_name="serving")
        lineage = self.lineage(result)
        self.assertNotEqual(result.model_uri, self.source)
        self.assertEqual(lineage["source"]["training_run_id"], self.training)
        self.assertEqual(lineage["conversion_run_id"], self.conversion)
        self.assertEqual(
            lineage["candidate"]["output_inventory"], digest(lineage["inventories"][1])
        )
        self.assertEqual(
            {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}, before
        )
        self.assertEqual(result.registered_model_version, "1")
        self.assertEqual(
            self.client.get_logged_model(result.model_id).tags[
                "triton.source.model_uri"
            ],
            self.source,
        )
        self.assertEqual(
            self.log(registered_model_name="serving").model_id, result.model_id
        )

    def test_alias_is_pinned_across_retry_and_source_change(self):
        self.client.create_registered_model("scientist")
        first = self.client.create_model_version(
            "scientist", self.source, model_id=self.source.split("/")[-1]
        )
        self.client.set_registered_model_alias("scientist", "ready", first.version)
        result = self.log(
            source_model_uri="models:/scientist@ready",
            publication_operation_id="pinned",
        )
        second_source = self.make_source(
            self.client, self.experiment, self.training, "new weights"
        )
        second = self.client.create_model_version(
            "scientist", second_source, model_id=second_source.split("/")[-1]
        )
        self.client.set_registered_model_alias("scientist", "ready", second.version)
        with patch.object(
            mlflow.MlflowClient,
            "get_model_version_by_alias",
            side_effect=AssertionError("alias re-resolved"),
        ):
            self.assertEqual(
                self.log(
                    source_model_uri="models:/scientist@ready",
                    publication_operation_id="pinned",
                ).model_id,
                result.model_id,
            )
        self.assertEqual(self.lineage(result)["source"]["model_uri"], self.source)

    def test_fork_keeps_source_and_explicit_predecessor_with_changed_weights(self):
        first = self.log()
        (self.repo / "iris/1/model.onnx").write_bytes(b"changed weights")
        second = self.log(previous_serving_model_uri=first.model_uri, precision="fp16")
        self.assertNotEqual(first.model_id, second.model_id)
        self.assertNotEqual(
            self.lineage(first)["candidate"]["output_inventory"],
            self.lineage(second)["candidate"]["output_inventory"],
        )
        self.assertEqual(
            self.lineage(second)["candidate"]["previous_serving_candidate"],
            self.lineage(first)["candidate"]["origin"],
        )
        self.assertEqual(self.lineage(second)["source"]["model_uri"], self.source)
        (self.repo / "iris/config.pbtxt").write_text(
            'name: "iris"\nbackend: "python"\n'
        )
        (self.repo / "iris/1/model.onnx").rename(self.repo / "iris/1/model.py")
        fork = self.log(previous_serving_model_uri=first.model_uri)
        self.assertEqual(self.lineage(fork)["backend"], "python")
        self.assertNotIn("previous_model_uri", fork.flavors["triton"])

    def test_changed_content_cannot_reuse_operation(self):
        self.log(publication_operation_id="same")
        (self.repo / "iris/1/model.onnx").write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "different content"):
            self.log(publication_operation_id="same")

    def test_registration_timeout_reconciles_without_duplicate(self):
        original = mlflow.MlflowClient.create_model_version

        def timeout(client, *args, **kwargs):
            original(client, *args, **kwargs)
            raise TimeoutError("response lost")

        with (
            patch.object(mlflow.MlflowClient, "create_model_version", timeout),
            self.assertRaises(TimeoutError),
        ):
            self.log(registered_model_name="serving")
        with patch.object(
            mlflow.MlflowClient,
            "create_model_version",
            side_effect=AssertionError("duplicate write"),
        ):
            result = self.log(registered_model_name="serving")
        self.assertEqual(result.registered_model_version, "1")
        self.assertEqual(len(self.client.search_model_versions()), 1)

    def test_logged_model_timeout_reconciles_without_duplicate(self):
        original = mlflow.MlflowClient.create_logged_model

        def timeout(client, *args, **kwargs):
            original(client, *args, **kwargs)
            raise TimeoutError("response lost")

        with (
            patch.object(mlflow.MlflowClient, "create_logged_model", timeout),
            self.assertRaises(TimeoutError),
        ):
            self.log()
        with patch.object(
            mlflow.MlflowClient,
            "create_logged_model",
            side_effect=AssertionError("duplicate write"),
        ):
            self.log()
        self.assertEqual(len(self.client.search_logged_models([self.experiment])), 2)

    def test_unresolved_write_fails_closed(self):
        with (
            patch.object(
                mlflow.MlflowClient, "create_logged_model", side_effect=TimeoutError
            ),
            self.assertRaises(TimeoutError),
        ):
            self.log()
        with (
            patch.object(
                mlflow.MlflowClient,
                "create_logged_model",
                side_effect=AssertionError("unsafe retry"),
            ),
            self.assertRaises(AmbiguousPublication),
        ):
            self.log()

    def test_upload_timeout_reuses_owned_model(self):
        original = mlflow.MlflowClient.log_model_artifacts

        def timeout(client, *args, **kwargs):
            original(client, *args, **kwargs)
            raise TimeoutError

        with (
            patch.object(mlflow.MlflowClient, "log_model_artifacts", timeout),
            self.assertRaises(TimeoutError),
        ):
            self.log()
        self.log()
        self.assertEqual(len(self.client.search_logged_models([self.experiment])), 2)

    def test_simultaneous_retries_create_one_model_and_registration(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(lambda _: self.log(registered_model_name="serving"), range(2))
            )
        self.assertEqual(results[0].model_id, results[1].model_id)
        self.assertEqual(len(self.client.search_model_versions()), 1)

    def test_cross_server_origin_keeps_training_and_conversion_runs_separate(self):
        other_uri = "sqlite:///" + str(self.root / "other.db")
        other = mlflow.MlflowClient(tracking_uri=other_uri, registry_uri=other_uri)
        experiment = other.create_experiment(
            "other", artifact_location=(self.root / "other-artifacts").as_uri()
        )
        run = other.create_run(experiment).info.run_id
        source = self.make_source(other, experiment, run, "remote")
        result = self.log(
            source_model_uri=source,
            source_tracking_uri=other_uri,
            source_tracking_installation_id="other-server",
        )
        self.assertEqual(
            self.lineage(result)["source"]["installation_id"], "other-server"
        )
        self.assertEqual(self.lineage(result)["source"]["training_run_id"], run)
        self.assertEqual(self.lineage(result)["conversion_run_id"], self.conversion)

    def test_unsafe_and_invalid_repositories_never_publish(self):
        (self.repo / "iris/1/link").symlink_to(self.repo / "iris/1/model.onnx")
        with self.assertRaises(ValueError):
            self.log()
        (self.repo / "iris/1/link").unlink()
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            self.log(backend="python")
        self.assertEqual(len(self.client.search_logged_models([self.experiment])), 1)

    def test_emitted_candidate_validates_against_shared_s05_a_contract(self):
        import jsonschema

        result = self.log()
        lineage = self.lineage(result)
        schemas = Path(__file__).parent / "fixtures/lineage-v1"
        jsonschema.validate(
            lineage["candidate"],
            json.loads((schemas / "ServingCandidateV1.schema.json").read_text()),
        )
        for inventory in lineage["inventories"]:
            jsonschema.validate(
                inventory, json.loads((schemas / "InventoryV1.schema.json").read_text())
            )
        saved = mlflow.models.get_model_info(result.model_uri)
        self.assertEqual(saved.metadata, result.metadata)
        attempt = lineage["candidate"]["attempts"][0]
        self.assertEqual(
            attempt["input_inventory"],
            lineage["candidate"]["derived_from_snapshot"]["inventory"],
        )
        self.assertEqual(
            attempt["output_inventory"], lineage["candidate"]["output_inventory"]
        )

    def test_separate_operations_serialize_registry_writes(self):
        from threading import Lock
        from time import sleep

        active = maximum = 0
        guard = Lock()
        original = mlflow.MlflowClient.create_model_version

        def slow(client, *args, **kwargs):
            nonlocal active, maximum
            with guard:
                active += 1
                maximum = max(maximum, active)
            try:
                sleep(0.05)
                return original(client, *args, **kwargs)
            finally:
                with guard:
                    active -= 1

        with (
            patch.object(mlflow.MlflowClient, "create_model_version", slow),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            results = list(
                pool.map(
                    lambda number: self.log(
                        registered_model_name="serving",
                        publication_operation_id=f"operation-{number}",
                    ),
                    range(2),
                )
            )
        self.assertEqual(maximum, 1)
        self.assertEqual(
            {item.registered_model_version for item in results}, {"1", "2"}
        )
        self.assertNotEqual(results[0].model_id, results[1].model_id)

    def test_conversion_run_access_failure_never_publishes(self):
        with (
            patch.object(
                mlflow.MlflowClient, "get_run", side_effect=PermissionError("denied")
            ),
            self.assertRaises(PermissionError),
        ):
            self.log()
        self.assertEqual(len(self.client.search_logged_models([self.experiment])), 1)

    def test_predecessor_alias_pins_original_candidate(self):
        first = self.log(registered_model_name="serving")
        self.client.set_registered_model_alias("serving", "previous", "1")
        second = self.log(previous_serving_model_uri="models:/serving@previous")
        self.log(
            registered_model_name="serving", publication_operation_id="new-predecessor"
        )
        self.client.set_registered_model_alias("serving", "previous", "2")
        with patch.object(
            mlflow.MlflowClient,
            "get_model_version_by_alias",
            side_effect=AssertionError("alias re-resolved"),
        ):
            self.assertEqual(
                self.log(
                    previous_serving_model_uri="models:/serving@previous"
                ).model_id,
                second.model_id,
            )
        self.assertEqual(
            self.lineage(second)["candidate"]["previous_serving_candidate"],
            self.lineage(first)["candidate"]["origin"],
        )

    def test_source_registry_name_cannot_be_reused(self):
        self.client.create_registered_model("scientist")
        self.client.create_model_version(
            "scientist", self.source, model_id=self.source.split("/")[-1]
        )
        with self.assertRaisesRegex(ValueError, "differ from source"):
            self.log(
                source_model_uri="models:/scientist/1",
                registered_model_name="scientist",
            )

    def test_runtime_only_retry_reuses_candidate(self):
        first = self.log()
        second = self.log()
        self.assertEqual(first.model_id, second.model_id)
        self.assertNotIn("replaces_release", self.lineage(second)["candidate"])

    def test_inventory_matches_s05_a_canonical_encoding(self):
        path = self.root / "inventory"
        path.mkdir()
        (path / "é").write_bytes(b"abc")
        inventory = capture(path, self.root / "captured")
        self.assertEqual(
            digest(inventory),
            "sha256:cf79dcba335458ea1eed9163d5f7de27b46fe658327a1df641c91598534e2f13",
        )


if __name__ == "__main__":
    unittest.main()
