"""Read exact MLflow source identities through explicitly selected clients."""

import re

import mlflow
from mlflow.models import Model

from ._serving_artifacts import capture, digest, safe_path


def validate_uri(uri):
    if (
        not isinstance(uri, str)
        or len(uri) > 256
        or not uri.startswith(("models:/", "runs:/"))
    ):
        raise ValueError("Source must be a models:/ or runs:/ reference")
    if any(c in uri for c in "?#\\\x00") or "://" in uri:
        raise ValueError(
            "Source must not contain credentials, query parameters or authority"
        )
    if any(ord(c) < 32 for c in uri):
        raise ValueError("Invalid source URI")
    safe_path(uri.split(":/", 1)[1])


def resolve(client, uri, installation, directory, *, predecessor=False):
    validate_uri(uri)
    (directory / "download").mkdir(parents=True)
    requested = uri
    registry_name = registry_version = model_id = run_id = None
    body = uri.split(":/", 1)[1]
    if uri.startswith("models:/"):
        if re.fullmatch(r"m-[A-Za-z0-9_-]+", body):
            model_id = body
        else:
            if "@" in body and "/" not in body:
                name, alias = body.rsplit("@", 1)
                version = client.get_model_version_by_alias(name, alias)
            else:
                name, separator, number = body.rpartition("/")
                if not separator or not number.isascii() or not number.isdecimal():
                    raise ValueError(
                        "Use a numeric Registry version or explicit @alias"
                    )
                version = client.get_model_version(name, number)
            if version.status != "READY":
                raise ValueError("Selected Registry version is not READY")
            registry_name, registry_version = version.name, str(version.version)
            model_id, run_id = version.model_id, version.run_id
            if not model_id and re.fullmatch(
                r"models:/m-[A-Za-z0-9_-]+", version.source
            ):
                model_id = version.source.split("/")[-1]
            if not model_id:
                raise ValueError("Registry source requires an MLflow 3 logged-model ID")
        logged = client.get_logged_model(model_id)
        if logged.status != "READY":
            raise ValueError("Selected logged model is not READY")
        artifact_uri = logged.artifact_location
        run_id = logged.source_run_id
        resolved_uri = "models:/" + model_id
        local = mlflow.artifacts.download_artifacts(
            artifact_uri=artifact_uri,
            dst_path=str(directory / "download"),
            tracking_uri=client.tracking_uri,
        )
    else:
        run_id, separator, artifact_path = body.partition("/")
        if not separator:
            raise ValueError("Run source requires an artifact path")
        safe_path(artifact_path)
        client.get_run(run_id)
        local = client.download_artifacts(
            run_id, artifact_path, str(directory / "download")
        )
        resolved_uri = uri
    inventory = capture(local, directory / "capture")
    metadata_path = directory / "capture" / "MLmodel"
    if not metadata_path.is_file() or metadata_path.stat().st_size > 1024 * 1024:
        raise ValueError("Source requires a bounded MLmodel document")
    metadata = Model.load(metadata_path)
    if model_id and metadata.model_id != model_id:
        raise ValueError(
            "Source metadata disagrees with resolved logged-model identity"
        )
    if metadata.run_id and metadata.run_id != run_id:
        raise ValueError("Source metadata disagrees with source run")
    if predecessor and "triton" not in metadata.flavors:
        raise ValueError("Serving predecessor requires a Triton flavor")
    candidate_origin = None
    if predecessor and (directory / "capture" / "lineage.json").is_file():
        import json

        lineage_path = directory / "capture" / "lineage.json"
        if lineage_path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("Predecessor lineage exceeds metadata bounds")
        lineage = json.loads(lineage_path.read_text())
        declared = (metadata.metadata or {}).get("triton_lineage", {}).get("digest")
        if declared and digest(lineage) != declared:
            raise ValueError("Predecessor lineage does not match its metadata digest")
        candidate_origin = lineage["candidate"]["origin"]
        if (
            not isinstance(candidate_origin, dict)
            or set(candidate_origin) != {"installation_id", "record_id"}
            or any(
                not isinstance(value, str) or not 1 <= len(value) <= 256
                for value in candidate_origin.values()
            )
        ):
            raise ValueError("Invalid predecessor candidate origin")
    record = {
        "installation_id": installation,
        "requested_uri": requested,
        "model_uri": resolved_uri,
        "model_id": model_id,
        "training_run_id": run_id,
        "registry_name": registry_name,
        "registry_version": registry_version,
        "candidate_origin": candidate_origin,
    }
    return record, inventory
