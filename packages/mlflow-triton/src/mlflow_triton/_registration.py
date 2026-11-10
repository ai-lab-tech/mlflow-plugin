"""Register already logged models using the logging wrappers' Triton history rules."""

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
from mlflow.models import Model
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

from . import _logging
from ._versions import (
    RepositoryUpdate,
    _previous_artifact_uri,
    _repository_data,
    append_shared_version,
    extend_registered_repository,
)


def register_model(model_uri, name, await_registration_for=300, *, tags=None):
    """Create a Registry version, using the next Triton number for matching configs.

    Existing Registry versions and their artifacts are never updated. Calls for
    one registered name must be sequential. Native MLflow/UI registration does
    not perform this plugin's Triton config comparison.
    """
    source = _previous_artifact_uri(model_uri)
    info = mlflow.models.get_model_info(source)
    flavor = info.flavors.get("triton")
    if flavor is None:
        return mlflow.register_model(
            model_uri, name, await_registration_for=await_registration_for, tags=tags
        )
    data = _repository_data(flavor)
    number = flavor.get("model_version", 1)
    if type(number) is not int or number < 1:
        raise ValueError("Invalid Triton model_version in source model metadata")

    with TemporaryDirectory(prefix="mlflow-triton-register-") as temporary:
        root = Path(temporary)
        prepared = root / "prepared"
        prepared.mkdir()
        # Copy only the current source model and serving version. Earlier source
        # versions and predecessor weights are never downloaded or copied.
        downloads = root / "downloads"
        downloads.mkdir()
        for entry in mlflow.artifacts.list_artifacts(artifact_uri=source):
            if Path(entry.path).name == "model_repository":
                continue
            local = Path(
                mlflow.artifacts.download_artifacts(
                    artifact_uri=source + "/" + Path(entry.path).name,
                    dst_path=str(downloads),
                )
            )
            destination = prepared / local.name
            if local.is_dir():
                shutil.copytree(local, destination)
            else:
                shutil.copy2(local, destination)
        repository = prepared / data
        repository.mkdir(parents=True)
        repository_source = source + "/" + data
        for entry in mlflow.artifacts.list_artifacts(artifact_uri=repository_source):
            filename = Path(entry.path).name
            if entry.is_dir and filename != str(number):
                continue
            local = Path(
                mlflow.artifacts.download_artifacts(
                    artifact_uri=repository_source + "/" + filename,
                    dst_path=str(downloads),
                )
            )
            if entry.is_dir:
                shutil.copytree(local, repository / "1")
            else:
                shutil.copy2(local, repository / filename)
        update = extend_registered_repository(prepared, name)
        prepared_info = Model.load(prepared / "MLmodel")
        target_number = prepared_info.flavors["triton"]["model_version"]
        client = mlflow.MlflowClient()
        stable_uri = f"models:/{info.model_id}" if info.model_id else model_uri
        append_shared_version(prepared, update)
        same_owner = prepared_info.flavors["triton"].get(
            "repository_model_id"
        ) == flavor.get("repository_model_id")
        if target_number == number and same_owner:
            # No renumbering is needed, so registration can reference the source.
            return mlflow.register_model(
                stable_uri,
                name,
                await_registration_for=await_registration_for,
                tags=tags,
            )
        if info.model_id and not _is_registered(client, info.model_id, source):
            # A fresh, unregistered training output keeps its own model ID.
            # Renumber only its serving directory; never change a registered source.
            _logging._log_update(prepared, RepositoryUpdate(info.model_id), info.run_id)
            if target_number != number:
                get_artifact_repository(source).delete_artifacts(
                    data + "/" + str(number)
                )
            return mlflow.register_model(
                stable_uri,
                name,
                await_registration_for=await_registration_for,
                tags=tags,
            )
        # A source already referenced by Registry versions stays unchanged.
        logged = Model.log(
            artifact_path=None,
            flavor=_logging,
            name=info.name,
            run_id=info.run_id,
            prepared_path=str(prepared),
            validate_serving_input=False,
        )
        return mlflow.register_model(
            logged.model_uri,
            name,
            await_registration_for=await_registration_for,
            tags=tags,
        )


def _is_registered(client, model_id, source):
    """Check all visible Registry versions before renumbering a training output."""
    page_token = None
    while True:
        versions = client.search_model_versions(max_results=100, page_token=page_token)
        for version in versions:
            if getattr(version, "model_id", None) == model_id or version.source in {
                source,
                f"models:/{model_id}",
            }:
                return True
            # Legacy registrations can use runs:/ references without model_id.
            if not getattr(version, "model_id", None):
                if mlflow.models.get_model_info(version.source).model_id == model_id:
                    return True
        page_token = versions.token
        if not page_token:
            return False
