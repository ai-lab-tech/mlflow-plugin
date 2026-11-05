"""Translate MLflow config into the shared deployment exposure contract."""

import re

from mlflow.exceptions import MlflowException


def _boolean(value, name):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise MlflowException(f"{name} must be true or false")


def access_payload(config, request):
    if "ingress_class_name" in config:
        raise MlflowException("Ingress class is configured by the administrator")
    host = str(config.get("ingress_host") or "").strip()
    mode = config.get("access_mode") or ("ingress" if host else "internal")
    auth = _boolean(config.get("basic_auth_enabled", False), "basic_auth_enabled")
    username, password = (
        config.get("basic_auth_username"),
        config.get("basic_auth_password"),
    )
    if mode not in {"internal", "ingress"}:
        raise MlflowException("access_mode must be internal or ingress")
    if mode == "internal":
        if host or auth or username or password:
            raise MlflowException(
                "Cluster-only access cannot include ingress or authentication settings"
            )
        return {"access_mode": "internal"}
    scheme = config.get("ingress_scheme") or "https"
    host = host.lower().rstrip(".")
    if (
        not host
        or len(host) > 253
        or any(
            not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label)
            for label in host.split(".")
        )
    ):
        raise MlflowException("Enter a hostname only, without a protocol, port or path")
    if not host or scheme not in {"http", "https"}:
        raise MlflowException("Ingress requires a hostname and HTTP or HTTPS")
    if auth and (scheme != "https" or not username or not password):
        raise MlflowException(
            "Basic authentication requires HTTPS and both username and password"
        )
    if not auth and (username or password):
        raise MlflowException(
            "Enable basic authentication before supplying credentials"
        )
    policy = request("GET", "/api/deployments/policy")
    if policy.get("base_domain") and "." in host:
        raise MlflowException(
            "Enter only the short hostname; the configured base domain is appended automatically"
        )
    if policy.get("ingress_enabled") is not True:
        raise MlflowException("Model Ingress is disabled by the administrator")
    if scheme == "http" and policy.get("allow_http") is not True:
        raise MlflowException(
            "HTTP endpoints are disabled by the administrator. Use HTTPS."
        )
    result = {
        "access_mode": "ingress",
        "ingress_host": host.lower().rstrip("."),
        "ingress_scheme": scheme,
        "basic_auth_enabled": auth,
    }
    if auth:
        result.update(basic_auth_username=username, basic_auth_password=password)
    return result
