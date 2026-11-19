"""Read bounded, integrity-checked lineage independently of artifact storage."""

import json
import re
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory

import jsonschema
import mlflow
from mlflow.exceptions import MlflowException
from mlflow_triton._serving_artifacts import canonical, digest
from mlflow_triton._serving_sources import validate_uri

MAX_LINEAGE_BYTES = 16 * 1024 * 1024


def validate(value):
    schema = json.loads(
        files(__package__).joinpath("control-lineage.schema.json").read_text()
    )
    try:
        jsonschema.Draft202012Validator(schema).validate(value)
        if len(canonical(value)) > 48 * 1024:
            raise ValueError("Lineage references exceed 48 KiB")
        origin = value["origin"]
        record = origin["record_id"]
        if not (record.startswith("models:/m-") or record.startswith("runs:/")):
            raise ValueError("Use an exact logged model or run reference")
        if any(c in record for c in "?#\\") or "://" in record:
            raise ValueError("Invalid logged model identity")
        validate_uri(record)
        if not re.fullmatch(r"models:/m-[A-Za-z0-9_-]+|runs:/[^/?#\\]+/.+", record):
            raise ValueError("Origin is not an exact model reference")
        for reference in (value.get("source"), value.get("previous_serving_model")):
            if reference:
                validate_uri(reference["requested_uri"])
                validate_uri(reference["model_uri"])
                if not re.fullmatch(
                    r"models:/m-[A-Za-z0-9_-]+|runs:/[^/?#\\]+/.+",
                    reference["model_uri"],
                ):
                    raise ValueError("Resolved source must not be an alias")
                if (
                    reference.get("model_id")
                    and reference["model_uri"] != "models:/" + reference["model_id"]
                ):
                    raise ValueError("Model ID disagrees with its reference")
        candidate = value.get("candidate")
        if candidate:
            if candidate["derived_from_snapshot"]["kind"] != "mlflow":
                raise ValueError("Expected an MLflow source snapshot")
            source = value["source"]
            if (
                candidate["derived_from_snapshot"]["origin"]
                != {
                    "installation_id": source["installation_id"],
                    "record_id": source["model_uri"],
                }
                or candidate["origin"]["installation_id"] != origin["installation_id"]
            ):
                raise ValueError("Source identities disagree")
            previous = value.get("previous_serving_model")
            expected = (
                (
                    previous.get("candidate_origin")
                    or {
                        "installation_id": previous["installation_id"],
                        "record_id": previous["model_uri"],
                    }
                )
                if previous
                else None
            )
            if expected != candidate.get("previous_serving_candidate"):
                raise ValueError("Predecessor references disagree")
            current = candidate["derived_from_snapshot"]["inventory"]
            seen = set()
            for attempt in candidate["attempts"]:
                identity = tuple(sorted(attempt["origin"].items()))
                if identity in seen or attempt["input_inventory"] != current:
                    raise ValueError("Invalid candidate inventory chain")
                seen.add(identity)
                current = attempt["output_inventory"]
            if current != candidate["output_inventory"] or not any(
                a["origin"]["record_id"] == value.get("conversion_run_id")
                for a in candidate["attempts"]
            ):
                raise ValueError("Conversion run or output inventory disagrees")
            if not value.get("conversion_run_id") or not value.get("lineage_digest"):
                raise ValueError("Missing derived lineage references")
        elif any(
            value.get(k)
            for k in (
                "source",
                "previous_serving_model",
                "conversion_run_id",
                "lineage_digest",
            )
        ):
            raise ValueError("Legacy models must not invent derived lineage")
    except (jsonschema.ValidationError, ValueError, TypeError, KeyError) as error:
        raise MlflowException("Invalid Control lineage references") from error
    return value


def read(artifact_uri, metadata, model_uri, installation=None, *, required=False):
    declaration = (getattr(metadata, "metadata", None) or {}).get("triton_lineage")
    flavor = metadata.flavors.get("triton", {})
    if not declaration and not flavor.get("lineage") and not required:
        return None  # Existing deployment clients retain their wire contract.
    model_id = getattr(metadata, "model_id", None)
    exact = (
        "models:/" + model_id
        if model_id
        else model_uri
        if model_uri.startswith("runs:/")
        else None
    )
    if not exact:
        raise MlflowException(
            "Model metadata must identify an exact logged model or run artifact"
        )
    if not declaration and not flavor.get("lineage"):
        if not installation:
            raise MlflowException(
                "tracking_installation_id is required for legacy model lineage"
            )
        return validate(
            {
                "schema_version": 1,
                "provider": "mlflow",
                "origin": {
                    "installation_id": installation,
                    "record_id": exact,
                },
            }
        )
    if (
        not isinstance(declaration, dict)
        or declaration.get("manifest") != "lineage.json"
        or flavor.get("lineage") != "lineage.json"
    ):
        raise MlflowException("Derived model requires a declared lineage.json manifest")
    entries = mlflow.artifacts.list_artifacts(artifact_uri=artifact_uri)
    entry = next(
        (item for item in entries if Path(item.path).name == "lineage.json"), None
    )
    if (
        entry is None
        or entry.is_dir
        or not isinstance(entry.file_size, int)
        or not 0 < entry.file_size <= MAX_LINEAGE_BYTES
    ):
        raise MlflowException("Missing or oversized lineage.json")
    with TemporaryDirectory(prefix="control-lineage-") as directory:
        path = mlflow.artifacts.download_artifacts(
            artifact_uri=artifact_uri.rstrip("/") + "/lineage.json", dst_path=directory
        )
        with Path(path).open("rb") as stream:
            content = stream.read(MAX_LINEAGE_BYTES + 1)
        if len(content) > MAX_LINEAGE_BYTES:
            raise MlflowException("Lineage manifest exceeds size limit")
        try:
            document = json.loads(content)
            if (
                document.get("schema_version") != 1
                or declaration.get("schema_version") != 1
            ):
                raise ValueError("Unsupported lineage version")
            if digest(document) != declaration.get("digest") or digest(
                document["candidate"]
            ) != declaration.get("candidate_digest"):
                raise ValueError("Lineage digest mismatch")
            candidate = document["candidate"]
            inventories = {digest(item): item for item in document["inventories"]}
            if (
                candidate["output_inventory"] not in inventories
                or candidate["derived_from_snapshot"]["inventory"] not in inventories
            ):
                raise ValueError("Referenced inventories are missing")
            declared_installation = candidate["origin"]["installation_id"]
            if installation and installation != declared_installation:
                raise ValueError("Tracking installation differs from logged lineage")
            if declaration.get("model_uri") != exact:
                raise ValueError(
                    "Lineage declaration identifies a different logged model"
                )
            return validate(
                {
                    "schema_version": 1,
                    "provider": "mlflow",
                    "origin": {
                        "installation_id": declared_installation,
                        "record_id": exact,
                    },
                    "source": document["source"],
                    "candidate": candidate,
                    "previous_serving_model": document["previous_serving_model"],
                    "conversion_run_id": document["conversion_run_id"],
                    "lineage_digest": declaration["digest"],
                }
            )
        except (ValueError, TypeError, KeyError) as error:
            raise MlflowException(
                "Lineage manifest does not match the logged model"
            ) from error
