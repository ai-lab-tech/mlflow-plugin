"""Capture prepared files and measure the S05-A inventory without executing them."""

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from unicodedata import normalize


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def digest(value):
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


def safe_path(value):
    if (
        not value
        or value != normalize("NFC", value)
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or any(c in value for c in "\\:")
        or any(ord(c) < 32 for c in value)
    ):
        raise ValueError("Expected an NFC relative POSIX artifact path")
    return value


def capture(source, destination, *, max_bytes=10 * 1024**3, max_files=10000):
    """Copy a frozen caller-owned tree; reject links, special files and changes."""
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or not source.is_dir():
        raise ValueError("Artifact root must be a real directory")
    files, total = [], 0
    destination.mkdir(parents=True, exist_ok=True)
    for root, directories, names in os.walk(source, followlinks=False):
        for name in directories:
            if (Path(root) / name).is_symlink():
                raise ValueError("Symbolic links are not permitted")
        for name in names:
            path = Path(root) / name
            relative = safe_path(path.relative_to(source).as_posix())
            before = path.lstat()
            if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                raise ValueError(
                    "Only regular, non-linked artifact files are permitted"
                )
            total += before.st_size
            if total > max_bytes or len(files) >= max_files:
                raise ValueError("Artifact capture limit exceeded")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            checksum, size = hashlib.sha256(), 0
            with path.open("rb") as stream, target.open("xb") as output:
                if os.fstat(stream.fileno()) != before:
                    raise ValueError("Artifact changed while capturing")
                while chunk := stream.read(1024 * 1024):
                    size += len(chunk)
                    if size > before.st_size:
                        raise ValueError("Artifact changed while capturing")
                    checksum.update(chunk)
                    output.write(chunk)
            after = path.lstat()
            if size != before.st_size or (
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise ValueError("Artifact changed while capturing")
            files.append(
                {
                    "path": relative,
                    "size": size,
                    "digest": "sha256:" + checksum.hexdigest(),
                }
            )
    if not files:
        raise ValueError("Artifact tree is empty")
    return {
        "schema_version": 1,
        "files": sorted(files, key=lambda f: f["path"].encode("utf-8")),
    }


def validate_repository(root, inventory, expected_backend=None):
    """V1 supports one prepared model, with no ensemble or repository agents."""
    try:
        from tritonclient.grpc import model_config_pb2
    except ImportError as error:
        raise ImportError(
            "Install mlflow-triton[serving] to validate prepared configs"
        ) from error
    from google.protobuf import text_format

    paths = {entry["path"] for entry in inventory["files"]}
    names = {path.split("/")[0] for path in paths}
    if len(names) != 1:
        raise ValueError(
            "V1 requires exactly one model; multi-source repositories are unsupported"
        )
    name = names.pop()
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", name):
        raise ValueError("Invalid Triton model name")
    config_path = root / name / "config.pbtxt"
    if not config_path.is_file() or config_path.stat().st_size > 1024 * 1024:
        raise ValueError("A bounded config.pbtxt is required")
    config = text_format.Parse(config_path.read_text(), model_config_pb2.ModelConfig())
    platforms = {
        "onnxruntime_onnx": "onnxruntime",
        "tensorrt_plan": "tensorrt",
        "pytorch_libtorch": "pytorch",
        "tensorflow_graphdef": "tensorflow",
        "tensorflow_savedmodel": "tensorflow",
    }
    platform_backend = platforms.get(config.platform)
    backend = config.backend or platform_backend
    if config.name and config.name != name:
        raise ValueError("Configured model name disagrees with directory")
    if (
        not backend
        or (config.platform and platform_backend is None)
        or (config.backend and platform_backend and config.backend != platform_backend)
        or (expected_backend and backend != expected_backend)
    ):
        raise ValueError("Backend/platform declaration is absent or inconsistent")
    if config.HasField("ensemble_scheduling") or config.HasField(
        "model_repository_agents"
    ):
        raise ValueError("Ensembles and repository agents are unsupported")
    versions = set()
    for path in paths - {name + "/config.pbtxt"}:
        parts = path.split("/")
        if (
            len(parts) < 3
            or not re.fullmatch(r"[1-9][0-9]*", parts[1])
            or int(parts[1]) > 2**63 - 1
        ):
            raise ValueError(
                "Prepared model files require positive numeric version directories"
            )
        versions.add(parts[1])
    if len(versions) != 1:
        raise ValueError("V1 requires exactly one prepared version")
    version = versions.pop()
    if config.version_policy.HasField("specific") and set(
        config.version_policy.specific.versions
    ) != {int(version)}:
        raise ValueError("Version policy selects absent versions")
    filename = config.default_model_filename or {
        "python": "model.py",
        "onnxruntime": "model.onnx",
        "tensorrt": "model.plan",
        "pytorch": "model.pt",
        "tensorflow": "model.savedmodel"
        if config.platform == "tensorflow_savedmodel"
        else "model.graphdef",
    }.get(backend)
    if not filename:
        raise ValueError("Custom backend requires default_model_filename")
    safe_path(filename)
    expected = f"{name}/{version}/{filename}"
    if expected not in paths and not any(
        path.startswith(expected + "/") for path in paths
    ):
        raise ValueError("Prepared model artifact is missing")
    return {
        "data": "model_repository/" + name,
        "backend": backend,
        "model_version": int(version),
    }
