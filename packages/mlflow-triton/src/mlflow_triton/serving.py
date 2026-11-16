"""Publish an already-prepared repository as a distinct, traceable serving model."""

import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.entities import LoggedModelOutput
from mlflow.exceptions import MlflowException
from mlflow.models import Model

from ._publication import AmbiguousPublication, PublicationStore
from ._serving_artifacts import canonical, capture, digest, validate_repository
from ._serving_sources import resolve, validate_uri

OP_TAG = "triton.publication.operation"
CONTENT_TAG = "triton.publication.content"


def _identity(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,256}", value):
        raise ValueError(name + " must be a non-secret stable identifier")
    return value


def _pages(method, **kwargs):
    token = None
    while True:
        page = method(page_token=token, **kwargs)
        yield from page
        token = page.token
        if not token:
            return


def _match(items, operation, content):
    matches = [item for item in items if item.tags.get(OP_TAG) == operation]
    if len(matches) != 1 or matches[0].tags.get(CONTENT_TAG) != content:
        raise AmbiguousPublication(
            "Publication write is unresolved; retry reconciliation after checking server state"
        )
    return matches[0]


def _publish(client, store, operation, receipt, package, registered_name):
    tags = receipt["tags"]
    if not receipt.get("model_id"):
        if receipt.get("create_pending"):
            logged = _match(
                _pages(
                    client.search_logged_models,
                    experiment_ids=[receipt["experiment_id"]],
                    max_results=100,
                ),
                operation,
                tags[CONTENT_TAG],
            )
        else:
            receipt["create_pending"] = True
            store.save(operation, receipt)
            # Intent is durable BEFORE calling a provider without idempotency keys.
            logged = client.create_logged_model(
                experiment_id=receipt["experiment_id"],
                name=receipt["name"],
                source_run_id=receipt["run_id"],
                tags=tags,
                model_type="triton",
            )
        if logged.source_run_id != receipt["run_id"]:
            raise AmbiguousPublication("Reconciled logged model has a different run")
        receipt["model_id"] = logged.model_id
        store.save(operation, receipt)
    logged = client.get_logged_model(receipt["model_id"])
    if (
        logged.tags.get(OP_TAG) != operation
        or logged.tags.get(CONTENT_TAG) != tags[CONTENT_TAG]
    ):
        raise AmbiguousPublication(
            "Published model identity no longer matches its receipt"
        )
    receipt["metadata"]["triton_lineage"]["model_uri"] = "models:/" + logged.model_id
    metadata = Model(
        run_id=receipt["run_id"],
        model_id=logged.model_id,
        model_uuid=logged.model_id,
        artifact_path=logged.artifact_location,
        utc_time_created=receipt["created_at"],
        metadata=receipt["metadata"],
        flavors={"triton": receipt["flavor"]},
    )
    if not receipt.get("uploaded"):
        metadata.save(package / "MLmodel")
        # This retry writes the same captured bytes to this operation's own model.
        client.log_model_artifacts(logged.model_id, str(package))
        client.finalize_logged_model(logged.model_id, "READY")
        receipt["uploaded"] = True
        store.save(operation, receipt)
    run = client.get_run(receipt["run_id"])
    if not any(
        item.model_id == logged.model_id
        for item in (run.outputs.model_outputs if run.outputs else [])
    ):
        client.log_outputs(
            receipt["run_id"], [LoggedModelOutput(logged.model_id, step=0)]
        )
    for key, value in tags.items():
        client.set_tag(receipt["run_id"], key, value)
    if registered_name and not receipt.get("registry_version"):
        if receipt.get("register_pending"):
            version = _match(
                (
                    item
                    for item in _pages(client.search_model_versions, max_results=100)
                    if item.name == registered_name
                ),
                operation,
                tags[CONTENT_TAG],
            )
        else:
            try:
                client.get_registered_model(registered_name)
            except MlflowException as error:
                if error.error_code != "RESOURCE_DOES_NOT_EXIST":
                    raise
                client.create_registered_model(registered_name)
            receipt["register_pending"] = True
            store.save(operation, receipt)
            version = client.create_model_version(
                registered_name,
                "models:/" + logged.model_id,
                run_id=receipt["run_id"],
                model_id=logged.model_id,
                tags=tags,
                await_creation_for=0,
            )
        if (
            version.model_id not in {None, logged.model_id}
            or version.source != "models:/" + logged.model_id
            or version.run_id != receipt["run_id"]
        ):
            raise AmbiguousPublication(
                "Registration does not match publication receipt"
            )
        if version.status != "READY":
            raise AmbiguousPublication(
                "Registration is not READY; retry reconciliation"
            )
        receipt["registry_version"] = str(version.version)
        store.save(operation, receipt)
    receipt["complete"] = True
    store.save(operation, receipt)
    info = metadata.get_model_info(client.get_logged_model(logged.model_id))
    info.registered_model_version = receipt.get("registry_version")
    return info


def log_model(
    *,
    source_model_uri,
    triton_repository_path,
    conversion_run_id,
    previous_serving_model_uri=None,
    registered_model_name=None,
    name="serving-model",
    backend=None,
    precision=None,
    source_tracking_installation_id=None,
    tracking_installation_id=None,
    previous_tracking_installation_id=None,
    source_tracking_uri=None,
    source_registry_uri=None,
    previous_tracking_uri=None,
    previous_registry_uri=None,
    publication_operation_id=None,
    publication_database_url=None,
):
    """Log frozen prepared bytes; use a shared receipt database across publishers.

    Source and predecessor may use independent MLflow servers. IDs default to
    hashes of their tracking URIs; explicit installation IDs are recommended.
    Retries must use the same inputs and operation ID. No conversion is performed.
    """
    validate_uri(source_model_uri)
    if previous_serving_model_uri:
        validate_uri(previous_serving_model_uri)
    _identity(conversion_run_id, "conversion_run_id")
    if precision is not None:
        _identity(precision, "precision")
    for label, value in (
        ("name", name),
        ("registered_model_name", registered_model_name),
    ):
        if value is not None and (
            not isinstance(value, str) or not value.strip() or len(value) > 256
        ):
            raise ValueError(label + " must be a nonempty bounded name")
    target_uri, registry_uri = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    source_uri = source_tracking_uri or target_uri
    prior_uri = previous_tracking_uri or target_uri
    installation = _identity(
        tracking_installation_id or digest(target_uri), "tracking_installation_id"
    )
    source_installation = _identity(
        source_tracking_installation_id or digest(source_uri),
        "source_tracking_installation_id",
    )
    prior_installation = _identity(
        previous_tracking_installation_id or digest(prior_uri),
        "previous_tracking_installation_id",
    )
    client = mlflow.MlflowClient(tracking_uri=target_uri, registry_uri=registry_uri)
    source_client = mlflow.MlflowClient(
        tracking_uri=source_uri,
        registry_uri=source_registry_uri
        or (registry_uri if source_uri == target_uri else source_uri),
    )
    previous_client = mlflow.MlflowClient(
        tracking_uri=prior_uri,
        registry_uri=previous_registry_uri
        or (registry_uri if prior_uri == target_uri else prior_uri),
    )
    run = client.get_run(conversion_run_id)
    store = PublicationStore(
        publication_database_url or os.getenv("MLFLOW_TRITON_PUBLICATION_DATABASE_URL")
    )
    with tempfile.TemporaryDirectory(prefix="mlflow-triton-serving-") as temporary:
        root = Path(temporary)
        package = root / "package"
        output = capture(triton_repository_path, package / "model_repository")
        flavor = validate_repository(package / "model_repository", output, backend)
        request = {
            "source_uri": source_model_uri,
            "previous_uri": previous_serving_model_uri,
            "run_id": conversion_run_id,
            "name": name,
            "registered_name": registered_model_name,
            "output_inventory": digest(output),
            "backend": backend,
            "precision": precision,
            "installation": installation,
            "source_installation": source_installation,
            "previous_installation": prior_installation,
            "source_server": digest([source_uri, source_client._registry_uri]),
            "previous_server": digest([prior_uri, previous_client._registry_uri]),
        }
        identity = _identity(
            publication_operation_id or digest(request), "publication_operation_id"
        )
        operation = digest([target_uri, registry_uri, identity]).split(":")[1]
        with (
            store.lock("operation:" + operation),
            store.lock(
                "registry:" + digest([registry_uri, registered_model_name or identity])
            ),
        ):
            receipt = store.get(operation)
            if receipt is not None and receipt["request"] != request:
                raise ValueError(
                    "Publication operation already binds different content; use a new operation ID"
                )
            if receipt is None:
                source, source_inventory = resolve(
                    source_client,
                    source_model_uri,
                    source_installation,
                    root / "source",
                )
                previous = None
                if previous_serving_model_uri:
                    previous, _ = resolve(
                        previous_client,
                        previous_serving_model_uri,
                        prior_installation,
                        root / "previous",
                        predecessor=True,
                    )
                if (
                    registered_model_name
                    and source_uri == target_uri
                    and source_client._registry_uri == registry_uri
                    and source["registry_name"] == registered_model_name
                ):
                    raise ValueError(
                        "Serving Registry name must differ from source Registry name"
                    )
                candidate = {
                    "schema_version": 1,
                    "origin": {"installation_id": installation, "record_id": identity},
                    "derived_from_snapshot": {
                        "origin": {
                            "installation_id": source_installation,
                            "record_id": source["model_uri"],
                        },
                        "kind": "mlflow",
                        "inventory": digest(source_inventory),
                    },
                    "attempts": [
                        {
                            "origin": {
                                "installation_id": installation,
                                "record_id": conversion_run_id,
                            },
                            "operation": "conversion",
                            "input_inventory": digest(source_inventory),
                            "output_inventory": digest(output),
                            "tool": None,
                        }
                    ],
                    "output_inventory": digest(output),
                    "previous_serving_candidate": (
                        previous["candidate_origin"]
                        or {
                            "installation_id": prior_installation,
                            "record_id": previous["model_uri"],
                        }
                    )
                    if previous
                    else None,
                    "plugin_references": [
                        {
                            "provider": "mlflow",
                            "installation_id": source_installation,
                            "record_id": source["model_uri"],
                        }
                    ],
                }
                if candidate["previous_serving_candidate"] == candidate["origin"]:
                    raise ValueError("Candidate cannot be its own predecessor")
                lineage = {
                    "schema_version": 1,
                    "candidate": candidate,
                    "source": source,
                    "previous_serving_model": previous,
                    "conversion_run_id": conversion_run_id,
                    "backend": flavor["backend"],
                    "precision": precision,
                    "inventories": [source_inventory, output],
                }
                tags = {
                    OP_TAG: operation,
                    CONTENT_TAG: digest(lineage),
                    "triton.source.installation": source_installation,
                    "triton.source.model_uri": source["model_uri"],
                    "triton.conversion.run_id": conversion_run_id,
                    "triton.output.inventory": digest(output),
                }
                if source["training_run_id"]:
                    tags["triton.source.run_id"] = source["training_run_id"]
                if previous:
                    tags["triton.previous.model_uri"] = previous["model_uri"]
                    tags["triton.previous.installation"] = prior_installation
                flavor.update(
                    lineage="lineage.json",
                    inventory_digest=digest(output),
                    repository_group=operation,
                )
                receipt = {
                    "request": request,
                    "lineage": lineage,
                    "flavor": flavor,
                    "tags": tags,
                    "metadata": {
                        "triton_lineage": {
                            "schema_version": 1,
                            "manifest": "lineage.json",
                            "digest": digest(lineage),
                            "candidate_digest": digest(candidate),
                            "publication_operation_id": identity,
                        }
                    },
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "run_id": conversion_run_id,
                    "name": name,
                    "experiment_id": run.info.experiment_id,
                }
                store.save(operation, receipt)
            (package / "lineage.json").write_bytes(canonical(receipt["lineage"]))
            info = _publish(
                client, store, operation, receipt, package, registered_model_name
            )
            return info
