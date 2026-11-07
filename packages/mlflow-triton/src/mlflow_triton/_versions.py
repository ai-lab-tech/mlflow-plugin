"""Number new Triton models using registered metadata and config compatibility."""

import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
from google.protobuf import text_format
from mlflow.models import Model
from mlflow.tracking import MlflowClient


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
    """Compare only the previous config and keep only the new model's files."""
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


def extend_registered_repository(prepared, registered_model_name):
    """Compare config.pbtxt and number the new model without copying old assets.

    Never modify registered artifacts. Calls for a registered name must be
    serialized to produce a linear version sequence.
    """
    metadata = Model.load(prepared / "MLmodel")
    flavor = metadata.flavors["triton"]
    repository = prepared / flavor["data"]
    flavor.update(repository_group=uuid.uuid4().hex, model_version=1)
    previous = _latest_version(MlflowClient(), registered_model_name)
    if previous is not None:
        uri = f"models:/{registered_model_name}/{previous.version}"
        if previous.status != "READY":
            raise ValueError(
                f"Previous model {uri} is {previous.status}; expected READY"
            )
        previous_flavor = mlflow.models.get_model_info(uri).flavors.get("triton")
        if previous_flavor is not None:
            data = _repository_data(previous_flavor)
            number = _advance_compatible_version(
                repository, uri + "/" + data, previous_flavor
            )
            if number is not None:
                flavor.update(
                    repository_group=previous_flavor.get("repository_group") or uri,
                    model_version=number,
                    previous_model_uri=uri,
                )
    metadata.save(prepared / "MLmodel")
