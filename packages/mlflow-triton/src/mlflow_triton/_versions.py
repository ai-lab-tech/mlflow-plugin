"""Build immutable Triton repository snapshots from registered model history."""

import shutil
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


def extend_registered_repository(prepared, registered_model_name):
    """Compare only config.pbtxt; copy history only when it matches.

    Each log owns its complete snapshot. Never modify registered artifacts.
    Calls for a registered name must be serialized to produce a linear history.
    """
    metadata = Model.load(prepared / "MLmodel")
    flavor = metadata.flavors["triton"]
    repository = prepared / flavor["data"]
    flavor.update(repository_group=uuid.uuid4().hex, model_version=1)
    previous = _latest_version(MlflowClient(), registered_model_name)
    if previous is not None:
        uri = f"models:/{registered_model_name}/{previous.version}"
        previous_flavor = mlflow.models.get_model_info(uri).flavors.get("triton")
        if previous_flavor is not None:
            data = _repository_data(previous_flavor)
            source = uri + "/" + data
            with TemporaryDirectory(prefix="mlflow-triton-history-") as temporary:
                config = Path(
                    mlflow.artifacts.download_artifacts(
                        artifact_uri=source + "/config.pbtxt", dst_path=temporary
                    )
                )
                if _config_tokens(config.read_text()) == _config_tokens(
                    (repository / "config.pbtxt").read_text()
                ):
                    history = Path(
                        mlflow.artifacts.download_artifacts(
                            artifact_uri=source, dst_path=temporary
                        )
                    )
                    if any(p.is_symlink() for p in history.rglob("*")):
                        raise ValueError(
                            "Previous Triton assets contain symbolic links"
                        )
                    versions = [
                        p
                        for p in history.iterdir()
                        if p.is_dir() and p.name.isascii() and p.name.isdecimal()
                    ]
                    if not versions or any(
                        int(p.name) < 1 or str(int(p.name)) != p.name for p in versions
                    ):
                        raise ValueError(
                            "Previous Triton repository has invalid versions"
                        )
                    number = max(int(p.name) for p in versions) + 1
                    (repository / "1").rename(repository / str(number))
                    for version in versions:
                        shutil.copytree(version, repository / version.name)
                    flavor.update(
                        repository_group=previous_flavor.get("repository_group") or uri,
                        model_version=number,
                        previous_model_uri=uri,
                    )
    metadata.save(prepared / "MLmodel")
