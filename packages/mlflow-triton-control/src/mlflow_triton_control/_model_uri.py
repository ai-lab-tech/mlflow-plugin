"""Resolve MLflow model references to an existing S3 Triton model directory."""

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import unquote, urlsplit

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.models import get_model_info
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking._tracking_service.utils import get_default_host_creds
from mlflow.utils.rest_utils import http_request


@dataclass(frozen=True)
class DeploymentArtifacts:
    repository_uri: str
    requirements_txt: str | None = None


def _read_requirements(repository_artifact_uri):
    """Read only the optional requirements file through MLflow artifact access."""
    try:
        entries = mlflow.artifacts.list_artifacts(artifact_uri=repository_artifact_uri)
        entry = next(
            (item for item in entries if Path(item.path).name == "requirements.txt"),
            None,
        )
        if entry is None:
            return None
        limit = 1024 * 1024
        if entry.is_dir or (entry.file_size is not None and entry.file_size > limit):
            raise MlflowException(
                "Invalid or oversized Triton requirements.txt (maximum 1 MiB)"
            )
        with TemporaryDirectory(prefix="triton-requirements-") as directory:
            local = mlflow.artifacts.download_artifacts(
                artifact_uri=repository_artifact_uri.rstrip("/") + "/requirements.txt",
                dst_path=directory,
            )
            with Path(local).open("rb") as source:
                content = source.read(limit + 1)
            if len(content) > limit:
                raise MlflowException("Triton requirements.txt exceeds 1 MiB")
            return content.decode("utf-8")
    except Exception as exc:
        raise MlflowException(
            "Could not read the Triton requirements.txt from MLflow artifacts; "
            "check artifact access and the UTF-8 file (maximum 1 MiB)"
        ) from exc


def _read_version_requirements(repository_uri, flavor):
    """Prefer version-local requirements, retaining support for older artifacts."""
    number = flavor.get("model_version", 1)
    if type(number) is not int or number < 1:
        raise MlflowException("Invalid Triton model_version in model metadata")
    entries = mlflow.artifacts.list_artifacts(artifact_uri=repository_uri)
    if any(item.is_dir and Path(item.path).name == str(number) for item in entries):
        requirements = _read_requirements(repository_uri + "/" + str(number))
        if requirements is not None:
            return requirements
    return _read_requirements(repository_uri)


def resolve_deployment_artifacts(model_uri):
    """Resolve one model version and retrieve its generated serving requirements."""
    return _resolve_model(model_uri, include_requirements=True)


def _relative_path(path):
    if not isinstance(path, str) or not path or path.startswith("/"):
        raise MlflowException("Invalid relative Triton artifact path")
    decoded = unquote(path)
    if any(part in {"", ".", ".."} for part in decoded.split("/")) or any(
        char in decoded for char in ("\\", "?", "#", "%", "\x00")
    ):
        raise MlflowException("Invalid relative Triton artifact path")
    return path


def _artifact_uri(model_uri):
    # Resolve Registry aliases once, before reading metadata, to avoid mixing
    # versions if an alias changes while deployment is being prepared.
    current = model_uri
    seen = set()
    for _ in range(10):
        if current in seen:
            raise MlflowException("Cyclic MLflow model URI")
        seen.add(current)
        scheme = urlsplit(current).scheme
        if scheme == "models":
            current = ModelsArtifactRepository.get_underlying_uri(current)
        elif scheme == "runs":
            current = RunsArtifactRepository.get_underlying_uri(current)
        else:
            return current
    raise MlflowException("Too many nested MLflow model references")


def _s3_location(artifact_uri):
    parsed = urlsplit(artifact_uri)
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise MlflowException("Invalid MLflow artifact location")
    if parsed.scheme == "s3" and parsed.netloc:
        if parsed.path.strip("/"):
            _relative_path(parsed.path.strip("/"))
        return artifact_uri.rstrip("/")
    if parsed.scheme == "mlflow-artifacts" and not parsed.netloc:
        path = _relative_path(parsed.path.lstrip("/"))
        tracking_uri = mlflow.get_tracking_uri()
        if urlsplit(tracking_uri).scheme not in {"http", "https"}:
            raise MlflowException("MLflow artifact proxy requires an HTTP tracking URI")
        # Use MLflow's configured credentials and TLS settings, not the separate
        # Triton Control deployment token. This endpoint exposes no credentials.
        response = http_request(
            get_default_host_creds(tracking_uri),
            "/api/2.0/mlflow/triton-control/artifact-store",
            "GET",
            timeout=30,
        )
        response.raise_for_status()
        destination = response.json().get("artifact_root")
        if not isinstance(destination, str) or urlsplit(destination).scheme != "s3":
            raise MlflowException("MLflow artifact proxy must use S3 storage")
        return _s3_location(destination) + "/" + path
    raise MlflowException(
        "MLflow model artifacts must use S3 storage or Triton Control's "
        "managed MLflow artifact proxy; local files are not uploaded"
    )


def resolve_model_uri(model_uri):
    """Read only MLmodel metadata; never download or reload model weights."""
    return _resolve_model(model_uri, include_requirements=False).repository_uri


def _resolve_model(model_uri, *, include_requirements):
    if not isinstance(model_uri, str) or not model_uri:
        raise MlflowException("model_uri is required")
    scheme = urlsplit(model_uri).scheme
    if scheme not in {"models", "runs"}:
        raise MlflowException(
            "model_uri must be a models:/ or runs:/ MLflow URI; direct storage paths are not supported"
        )
    try:
        artifact_uri = _artifact_uri(model_uri)
        root = _s3_location(artifact_uri)
        metadata = get_model_info(artifact_uri)
        flavor = metadata.flavors.get("triton")
        if not isinstance(flavor, dict):
            raise MlflowException("The MLflow model has no triton flavor")
        data = _relative_path(flavor.get("data"))
        parts = data.split("/")
        if len(parts) != 2 or parts[0] != "model_repository":
            raise MlflowException(
                "Triton flavor data must be model_repository/<model-name>; "
                "re-log the model with a framework wrapper"
            )
        model_name = flavor.get("model_name")
        if model_name is not None and data.split("/")[-1] != model_name:
            raise MlflowException(
                "Triton flavor model_name does not match its data path"
            )
        requirements = (
            _read_version_requirements(artifact_uri.rstrip("/") + "/" + data, flavor)
            if include_requirements
            else None
        )
        return DeploymentArtifacts(root + "/" + data, requirements)
    except MlflowException:
        raise
    except Exception as exc:
        raise MlflowException(
            "Could not resolve the MLflow model to S3. Check tracking/registry "
            "access, artifact storage and the triton flavor."
        ) from exc
