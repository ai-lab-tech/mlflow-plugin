"""Number new Triton models using predecessor metadata and config compatibility."""

import uuid
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

import mlflow
from google.protobuf import text_format
from mlflow.models import Model
from mlflow.store.artifact.models_artifact_repo import ModelsArtifactRepository
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.tracking import MlflowClient
from mlflow.utils.file_utils import local_file_uri_to_path


@dataclass(frozen=True)
class RepositoryUpdate:
    """Existing logged model (and optional registration) to reuse."""

    model_id: str
    registered_model_version: int | None = None


def _config_tokens(text):
    """Ignore protobuf whitespace/comments, preserving every configuration value."""
    tokenizer = text_format.Tokenizer(text.splitlines())
    tokens = []
    while not tokenizer.AtEnd():
        tokens.append(tokenizer.token)
        tokenizer.NextToken()
    return tokens


def _latest_version(client, name):
    # MLflow's filter parser strips quotes without unescaping them. Choose a
    # delimiter absent from the name; for names containing both, filter locally.
    quote = next((q for q in ("'", '"') if q not in name), None)
    query = f"name = {quote}{name}{quote}" if quote else None
    page_token = None
    while True:
        versions = client.search_model_versions(
            filter_string=query,
            order_by=["version_number DESC"],
            max_results=1 if quote else 100,
            page_token=page_token,
        )
        for version in versions:
            if version.name == name:
                return version
        page_token = versions.token
        if not page_token:
            return None


def _repository_data(flavor):
    if not isinstance(flavor, dict):
        raise ValueError("Invalid Triton flavor in previous model")
    data = flavor.get("data")
    if (
        not isinstance(data, str)
        or len(data.split("/")) != 2
        or data.split("/")[0] != "model_repository"
        or data.split("/")[1] in {"", ".", ".."}
        or any(char in data for char in ("\\", "%", "?", "#", "\x00"))
    ):
        raise ValueError("Invalid Triton repository path in previous model")
    return data


def _advance_compatible_version(repository, source, previous_flavor):
    """Prepare only the new directory; older directories remain in remote storage."""
    with TemporaryDirectory(prefix="mlflow-triton-config-") as temporary:
        config = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=source + "/config.pbtxt", dst_path=temporary
            )
        )
        if config.is_symlink():
            raise ValueError("Previous Triton config contains symbolic links")
        if _config_tokens(config.read_text()) != _config_tokens(
            (repository / "config.pbtxt").read_text()
        ):
            return None

    # Older packages and models registered after logging contain only folder 1/
    # and omit this field. Explicitly invalid values must still fail validation.
    previous_number = previous_flavor.get("model_version", 1)
    if type(previous_number) is not int or previous_number < 1:
        raise ValueError("Invalid Triton model_version in previous model metadata")
    number = previous_number + 1
    (repository / "1").rename(repository / str(number))
    return number


def _previous_artifact_uri(uri):
    """Pin explicit references once so moving Registry aliases cannot mix artifacts."""
    if (
        not isinstance(uri, str)
        or not uri
        or urlsplit(uri).scheme not in {"models", "runs"}
    ):
        raise ValueError("model_uri must be a models:/ or runs:/ MLflow URI")
    seen = set()
    for _ in range(10):
        if uri in seen:
            raise ValueError("Cyclic model_uri")
        seen.add(uri)
        scheme = urlsplit(uri).scheme
        if scheme == "models":
            uri = ModelsArtifactRepository.get_underlying_uri(uri)
        elif scheme == "runs":
            uri = RunsArtifactRepository.get_underlying_uri(uri)
        else:
            # Model.load expects a filesystem path for local artifacts, not file:/ URIs.
            return local_file_uri_to_path(uri) if scheme == "file" else uri.rstrip("/")
    raise ValueError("Too many nested model_uri references")


def extend_registered_repository(
    prepared, registered_model_name=None, *, model_uri=None
):
    """Prepare an append and return its existing model identity, or None for new models.

    The two history parameters are mutually exclusive. Matching configs reuse
    the logged model and registration. Serialize calls per model/history.
    """
    if registered_model_name is not None and model_uri is not None:
        raise ValueError("registered_model_name and model_uri are mutually exclusive")
    metadata = Model.load(prepared / "MLmodel")
    flavor = metadata.flavors["triton"]
    repository = prepared / flavor["data"]
    flavor.update(repository_group=uuid.uuid4().hex, model_version=1)
    uri = model_uri
    source = None
    registry_version = None
    update = None
    if uri is not None:
        source = _previous_artifact_uri(uri)
    elif registered_model_name is not None:
        previous = _latest_version(MlflowClient(), registered_model_name)
        if previous is not None:
            uri = f"models:/{registered_model_name}/{previous.version}"
            if previous.status != "READY":
                raise ValueError(
                    f"Previous model {uri} is {previous.status}; expected READY"
                )
            source = uri
            registry_version = int(previous.version)
    if source is not None:
        previous_info = mlflow.models.get_model_info(source)
        previous_flavor = previous_info.flavors.get("triton")
        if previous_flavor is not None:
            data = _repository_data(previous_flavor)
            number = _advance_compatible_version(
                repository, source + "/" + data, previous_flavor
            )
            if number is not None:
                model_id = getattr(previous_info, "model_id", None)
                if not isinstance(model_id, str) or not model_id:
                    raise ValueError(
                        "Compatible predecessor has no MLflow logged-model ID; "
                        "log it with MLflow 3 before updating it in place"
                    )
                update = RepositoryUpdate(model_id, registry_version)
                flavor.update(
                    repository_group=previous_flavor.get("repository_group") or source,
                    model_version=number,
                    model_uri=uri,
                )
    metadata.save(prepared / "MLmodel")
    return update
