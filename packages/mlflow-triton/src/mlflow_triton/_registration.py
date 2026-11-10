"""Register already logged models using the logging wrappers' Triton history rules."""

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
from mlflow.models import Model

from . import _logging
from ._versions import (
    _previous_artifact_uri,
    _repository_data,
    extend_registered_repository,
)


def register_model(model_uri, name, await_registration_for=300, *, tags=None):
    """Register a logged model, appending a Triton version for matching configs.

    Like log_model(registered_model_name=...), compatible models update the
    existing logged-model ID and Registry version. Changed configs create a
    Registry version starting at Triton 1/. Calls for one name must be sequential.
    Native mlflow.register_model and UI registration do not call this function.
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
        client = mlflow.MlflowClient()
        if update is not None:
            _logging._log_update(prepared, update, info.run_id)
            version = client.get_model_version(
                name, str(update.registered_model_version)
            )
            for key, value in (tags or {}).items():
                client.set_model_version_tag(name, version.version, key, value)
            return client.get_model_version(name, version.version)
        if number == 1:
            # Standard registration retains the original source model identity.
            return mlflow.register_model(
                model_uri,
                name,
                await_registration_for=await_registration_for,
                tags=tags,
            )
        # Registering an already versioned source into a new/incompatible history
        # needs a fresh package at 1/; never renumber its existing artifacts.
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
