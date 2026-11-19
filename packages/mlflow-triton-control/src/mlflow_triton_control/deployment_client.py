"""MLflow deployment client backed by Triton Control's deployment API."""

from __future__ import annotations

import math
import os
import re
import time
from urllib.parse import quote, urlsplit

import httpx
from mlflow.deployments import BaseDeploymentClient
from mlflow.exceptions import MlflowException

from ._deployment_access import access_payload
from ._model_uri import export_lineage, resolve_deployment_artifacts

_NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_MODEL = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*\Z")


def target_help() -> str:
    return (
        "Target: triton-control. Backend address comes from TRITON_CONTROL_URL. "
        "Install this package in the MLflow CLI environment. "
        "Use -m models:/<model-id>, models:/<name>@<alias> or runs:/<run-id>/<path> and "
        "-C s3_profile_name=<owned-profile-name> "
        "-C image=<triton-image>. "
        "The logged model must contain a triton flavor backed by S3 artifacts. "
        "Set TRITON_CONTROL_TOKEN to a user-bound API token. "
        "Configure MLflow tracking/registry access. Update, predict and local serving are unsupported."
    )


def run_local(target, name, model_uri, flavor=None, config=None):
    raise MlflowException("Local Triton serving is not supported by triton-control")


class TritonControlDeploymentClient(BaseDeploymentClient):
    """Create and manage user-owned Triton Control deployments."""

    def __init__(self, target_uri):
        super().__init__(target_uri)
        if target_uri != "triton-control":
            raise MlflowException(
                "Only target 'triton-control' is supported; "
                "the backend address must come from TRITON_CONTROL_URL"
            )
        base_url = os.getenv("TRITON_CONTROL_URL", "").strip()
        if not base_url:
            raise MlflowException(
                "TRITON_CONTROL_URL is required for target triton-control"
            )
        try:
            parsed = urlsplit(base_url)
            parsed.port  # Validate a malformed or out-of-range port.
            valid = (
                parsed.scheme in {"http", "https"}
                and parsed.hostname
                and parsed.path in {"", "/"}
                and not parsed.query
                and not parsed.fragment
                and parsed.username is None
                and parsed.password is None
                and not any(char.isspace() for char in base_url)
            )
        except ValueError:
            valid = False
        if not valid:
            raise MlflowException(
                "TRITON_CONTROL_URL must be an HTTP(S) backend URL without "
                "credentials, a path, query or fragment"
            )
        self._base_url = base_url.rstrip("/")

    def _request(self, method, path, **kwargs):
        token = os.getenv("TRITON_CONTROL_TOKEN", "").strip()
        if not token:
            raise MlflowException("TRITON_CONTROL_TOKEN is required")
        try:
            with httpx.Client(
                base_url=self._base_url, timeout=10.0, follow_redirects=False
            ) as client:
                response = client.request(
                    method,
                    path,
                    headers={
                        **kwargs.pop("headers", {}),
                        "Authorization": f"Bearer {token}",
                    },
                    **kwargs,
                )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.json().get("detail", "Request failed")
                if isinstance(detail, list):
                    detail = "; ".join(
                        str(item.get("msg", "Invalid request"))
                        for item in detail
                        if isinstance(item, dict)
                    )
            except (ValueError, AttributeError):
                detail = "Request failed"
            raise MlflowException(
                f"Triton Control returned HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.RequestError as exc:
            raise MlflowException(
                f"Triton Control is unavailable: {type(exc).__name__}"
            ) from exc

    def _resolve_s3_profile(self, config):
        if "s3_profile_id" in config:
            raise MlflowException("s3_profile_id is not supported; use s3_profile_name")
        profile_name = config.get("s3_profile_name")
        if not isinstance(profile_name, str) or not profile_name.strip():
            raise MlflowException(
                "s3_profile_name is required and must be a non-empty string"
            )
        profile_name = profile_name.strip()
        profile = self._request(
            "GET", "/api/s3-profiles/resolve", params={"name": profile_name}
        )
        profile_id = profile.get("id")
        if (
            profile.get("name") != profile_name
            or isinstance(profile_id, bool)
            or not isinstance(profile_id, int)
            or profile_id <= 0
        ):
            raise MlflowException("Invalid S3 profile resolution response")
        return profile_id

    def create_deployment(
        self, name, model_uri, flavor=None, config=None, endpoint=None
    ):
        if endpoint is not None:
            raise MlflowException("endpoint is not supported")
        if flavor not in {None, "triton"}:
            raise MlflowException("Only the triton flavor is supported")
        if not _NAME.fullmatch(name):
            raise MlflowException("name must be a lowercase Kubernetes deployment name")
        config = config or {}
        image = config.get("image")
        if not isinstance(image, str) or not image.strip():
            raise MlflowException("image is required and must not be empty")
        image = image.strip()
        try:
            timeout = float(config.get("wait_timeout_seconds", 300))
        except (TypeError, ValueError) as exc:
            raise MlflowException(
                "wait_timeout_seconds must be a positive number"
            ) from exc
        if not math.isfinite(timeout) or timeout <= 0:
            raise MlflowException("wait_timeout_seconds must be a positive number")
        exposure = access_payload(config, self._request)
        profile_id = self._resolve_s3_profile(config)
        artifacts = (
            resolve_deployment_artifacts(
                model_uri, tracking_installation_id=config["tracking_installation_id"]
            )
            if config.get("tracking_installation_id")
            else resolve_deployment_artifacts(model_uri)
        )
        repository_uri = artifacts.repository_uri
        uri = urlsplit(repository_uri)
        parts = uri.path.strip("/").split("/")
        if (
            uri.scheme != "s3"
            or not uri.netloc
            or uri.query
            or uri.fragment
            or len(parts) < 2
            or any(part in {"", ".", ".."} for part in parts)
            or not _MODEL.fullmatch(parts[-1])
        ):
            raise MlflowException(
                "Resolved model URI must be s3://<bucket>/<repository-prefix>/<model-name>"
            )
        prefix, model_name = "/".join(parts[:-1]), parts[-1]
        payload = {
            "deployment_name": name,
            "s3_profile_id": profile_id,
            "model_uri": repository_uri,
            "repository_prefix": prefix,
            "model_name": model_name,
            "image": image,
            "model_control_mode": "explicit",
        }
        if getattr(artifacts, "lineage", None) is not None:
            payload["lineage"] = artifacts.lineage
        payload.update(exposure)
        if artifacts.requirements_txt is not None:
            payload["requirements_txt"] = artifacts.requirements_txt
        result = self._request("POST", "/api/deployments", json=payload)
        deadline = time.monotonic() + timeout
        while True:
            current = self.get_deployment(name)
            if current["status"] == "ready":
                return current
            if time.monotonic() >= deadline:
                raise MlflowException(
                    f"Deployment {name} (instance {result.get('instance_id')}) did not become ready: "
                    f"{current.get('status_detail', current['status'])}"
                )
            time.sleep(min(3.0, max(0.0, deadline - time.monotonic())))

    def export_lineage(self, model_uri, *, tracking_installation_id=None):
        """Export portable identity, without deploying or granting approval."""
        return export_lineage(
            model_uri, tracking_installation_id=tracking_installation_id
        )

    def export_deployment_lineage(self, name):
        """Read persisted provenance through the owner-authorized deployment API."""
        lineage = self.get_deployment(name).get("lineage")
        if lineage is None:
            raise MlflowException(
                "This older deployment has no recorded lineage; origin cannot be inferred from S3"
            )
        return lineage

    def prepare_handoff(
        self,
        model_uri,
        *,
        s3_profile_name,
        revision,
        title,
        tracking_installation_id=None,
        dataset=None,
        baseline=None,
        expected_behavior=None,
        acceptance_criteria=None,
        previous_handoff_id=None,
    ):
        """Resolve once and retain this JSON request for retries, even if aliases move."""
        artifacts = resolve_deployment_artifacts(
            model_uri,
            tracking_installation_id=tracking_installation_id,
            require_lineage=True,
        )
        profile_id = self._resolve_s3_profile({"s3_profile_name": s3_profile_name})
        profiles = self._request("GET", "/api/s3-profiles")
        profile = next((p for p in profiles if p["id"] == profile_id), None)
        location = urlsplit(artifacts.repository_uri)
        # Capture the repository root, retaining model-name/config.pbtxt layout.
        repository = location.path.strip("/").rsplit("/", 1)[0]
        prefix = (profile.get("prefix") or "").strip("/") if profile else ""
        if (
            profile is None
            or location.netloc != profile["bucket"]
            or prefix
            and not repository.startswith(prefix + "/")
        ):
            raise MlflowException("Model repository is outside the selected S3 profile")
        relative = repository[len(prefix) + 1 :] if prefix else repository
        if not relative:
            raise MlflowException(
                "Select a profile containing the repository as a child folder"
            )
        return {
            "schema_version": 1,
            "origin": artifacts.lineage["origin"],
            "revision": revision,
            "title": title,
            "previous_handoff_id": previous_handoff_id,
            "source": {
                "kind": "s3_repository",
                "profile_id": profile_id,
                "path": relative,
            },
            "lineage": artifacts.lineage,
            "dataset": dataset,
            "baseline": baseline,
            "expected_behavior": expected_behavior,
            "acceptance_criteria": acceptance_criteria,
        }

    def submit_ready(self, prepared_handoff, *, idempotency_key):
        """Explicitly enqueue preparation; log_model never calls this method."""
        if (
            not isinstance(idempotency_key, str)
            or not 1 <= len(idempotency_key) <= 128
            or idempotency_key != idempotency_key.strip()
        ):
            raise MlflowException(
                "A stable Idempotency-Key of 1 to 128 characters is required"
            )
        return self._request(
            "POST",
            "/api/preparation/handoff-captures",
            json=prepared_handoff,
            headers={"Idempotency-Key": idempotency_key},
        )

    def get_deployment(self, name):
        deployment = self._request(
            "GET", f"/api/deployments/mlflow/{quote(name, safe='')}"
        )
        return {**deployment, "flavor": "triton"}

    def list_deployments(self):
        return [
            {**deployment, "flavor": "triton"}
            for deployment in self._request("GET", "/api/deployments/mlflow")
        ]

    def delete_deployment(self, name):
        return self._request(
            "DELETE", f"/api/deployments/mlflow/{quote(name, safe='')}"
        )

    def update_deployment(
        self, name, model_uri=None, flavor=None, config=None, endpoint=None
    ):
        raise MlflowException("Updating deployments is not supported")

    def predict(self, deployment_name=None, inputs=None, endpoint=None):
        raise MlflowException("Prediction is not supported")
