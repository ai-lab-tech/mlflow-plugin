"""MLflow flavor for a complete Triton model repository directory."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlsplit

import httpx
import mlflow
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.models import Model, get_model_info

FLAVOR_NAME = "triton"
_MODEL_DATA = "model"


def _validate_repository(model_path: Path) -> None:
    if not model_path.is_dir():
        raise MlflowException(f"Triton model directory does not exist: {model_path}")
    if model_path.is_symlink() or any(
        path.is_symlink() for path in model_path.rglob("*")
    ):
        raise MlflowException("Triton model directory must not contain symbolic links")
    versions = [
        path
        for path in model_path.iterdir()
        if path.is_dir()
        and path.name.isascii()
        and path.name.isdecimal()
        and int(path.name) > 0
    ]
    if not versions:
        raise MlflowException("Triton model directory must contain a numbered version")


def save_model(triton_model_path, path, mlflow_model=None):
    """Save one Triton model directory as an MLflow Model with the triton flavor.

    ``triton_model_path`` points at the model-name directory containing
    ``config.pbtxt`` (if needed) and numbered version directories.
    """
    source = Path(triton_model_path).expanduser()
    _validate_repository(source)
    source = source.resolve()
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise MlflowException(f"MLflow model path already exists: {destination}")
    if destination == source or destination.is_relative_to(source):
        raise MlflowException(
            "MLflow model path must be outside the Triton model directory"
        )

    destination.mkdir(parents=True)
    data_path = f"{_MODEL_DATA}/{source.name}"
    shutil.copytree(source, destination / data_path)
    model = mlflow_model or Model()
    model.add_flavor(FLAVOR_NAME, data=data_path, model_name=source.name)
    model.save(str(destination / "MLmodel"))


def log_training_artifacts(
    training_code_path,
    *,
    artifact_path="training-artifacts",
):
    """Log training source and installed packages to the active MLflow run.

    The source directory is uploaded recursively under ``<artifact_path>/code``.
    Keep datasets outside this directory. ``requirements.txt`` is captured with
    ``pip freeze`` using the current Python interpreter and logged directly under
    ``artifact_path``. It describes the training environment, not a minimized
    serving environment. The source directory is not modified.

    Returns the run-relative artifact paths for code and requirements.
    """
    source = Path(training_code_path).expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"Training code directory does not exist: {source}")
    if mlflow.active_run() is None:
        raise RuntimeError("log_training_artifacts() requires an active MLflow run")

    requirements = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    mlflow.log_artifacts(str(source), artifact_path=f"{artifact_path}/code")
    with tempfile.TemporaryDirectory() as temp_dir:
        requirements_file = Path(temp_dir) / "requirements.txt"
        requirements_file.write_text(requirements, encoding="utf-8")
        mlflow.log_artifact(str(requirements_file), artifact_path=artifact_path)

    return {
        "code": f"{artifact_path}/code/",
        "requirements": f"{artifact_path}/requirements.txt",
    }


def log_model(
    triton_model_path,
    name="triton-model",
    registered_model_name=None,
    await_registration_for=300,
    run_id=None,
    metadata=None,
):
    """Log a Triton repository and optionally register a model version.

    Returns MLflow's ``ModelInfo``. A registry version is created only when
    ``registered_model_name`` is given. S3-backed registered versions receive a
    ``triton_model_uri`` tag automatically. ``ModelInfo.model_uri`` can be passed
    directly to the deployment client.
    """
    info = Model.log(
        artifact_path=None,
        flavor=sys.modules[__name__],
        name=name,
        registered_model_name=registered_model_name,
        await_registration_for=await_registration_for,
        run_id=run_id,
        metadata=metadata,
        triton_model_path=triton_model_path,
    )

    if registered_model_name and info.registered_model_version:
        repository_uri = _repository_s3_uri(info)
        if repository_uri is not None:
            MlflowClient().set_model_version_tag(
                name=registered_model_name,
                version=info.registered_model_version,
                key="triton_model_uri",
                value=repository_uri,
            )
    return info


def _repository_s3_uri(model_info):
    """Resolve a Triton flavor's data directory in an S3-backed artifact store.

    Local/non-S3 stores remain usable for logging but cannot be deployed through
    an S3 profile. Proxy stores use Triton Control's artifact-store endpoint.
    """
    flavor = model_info.flavors.get(FLAVOR_NAME)
    if not flavor:
        raise MlflowException("MLflow model has no triton flavor")
    data = flavor.get("data", "")
    if (
        not isinstance(data, str)
        or any(part in {"", ".", ".."} for part in data.split("/"))
        or "\\" in data
    ):
        raise MlflowException("Triton flavor contains an invalid data path")
    if model_info.model_id:
        location = mlflow.get_logged_model(model_info.model_id).artifact_location
    else:
        run_root = MlflowClient().get_run(model_info.run_id).info.artifact_uri
        location = run_root.rstrip("/") + "/" + model_info.artifact_path
    parsed = urlsplit(location)
    if parsed.scheme == "mlflow-artifacts":
        parts = unquote(parsed.path).strip("/").split("/")
        if (
            parsed.netloc
            or parsed.query
            or parsed.fragment
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise MlflowException("Invalid MLflow proxy artifact location")
        tracking_uri = mlflow.get_tracking_uri().rstrip("/")
        if urlsplit(tracking_uri).scheme not in {"http", "https"}:
            raise MlflowException("Proxy artifact resolution requires HTTP tracking")
        token = os.getenv("MLFLOW_TRACKING_TOKEN") or os.getenv("TRITON_CONTROL_TOKEN")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        try:
            response = httpx.get(
                tracking_uri + "/api/2.0/mlflow/triton-control/artifact-store",
                headers=headers,
                timeout=30,
                follow_redirects=False,
                verify=os.getenv("MLFLOW_TRACKING_SERVER_CERT_PATH") or True,
            )
            response.raise_for_status()
            root = response.json()["artifact_root"]
            if not isinstance(root, str) or urlsplit(root).scheme != "s3":
                raise ValueError("Artifact store must be S3-backed")
            location = root.rstrip("/") + "/" + "/".join(parts)
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise MlflowException(
                "Cannot resolve Triton Control's S3 artifact store"
            ) from exc
        parsed = urlsplit(location)
    elif parsed.scheme != "s3":
        return None
    if (
        not parsed.netloc
        or parsed.query
        or parsed.fragment
        or any(
            part in {"", ".", ".."}
            for part in unquote(parsed.path).strip("/").split("/")
        )
    ):
        raise MlflowException("Invalid S3 model artifact location")
    return location.rstrip("/") + "/" + data


def _resolve_repository_uri(model_uri):
    """Accept direct S3 repositories or resolve MLflow model references."""
    if urlsplit(model_uri).scheme in {"models", "runs"}:
        repository_uri = _repository_s3_uri(get_model_info(model_uri))
        if repository_uri is None:
            raise MlflowException(
                "Triton deployment requires an S3-backed MLflow model"
            )
        return repository_uri
    return model_uri


def load_model(model_uri):
    """Download an MLflow Triton model and return its repository directory."""
    model_root = Path(download_artifacts(artifact_uri=model_uri)).resolve()
    model = Model.load(str(model_root / "MLmodel"))
    flavor = model.flavors.get(FLAVOR_NAME)
    if not flavor:
        raise MlflowException(f"Model has no {FLAVOR_NAME} flavor: {model_uri}")
    data_path = (model_root / flavor.get("data", "")).resolve()
    if not data_path.is_relative_to(model_root) or data_path == model_root:
        raise MlflowException("Triton flavor contains an invalid data path")
    _validate_repository(data_path)
    return data_path
