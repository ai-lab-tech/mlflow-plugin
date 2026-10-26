#!/usr/bin/env python3
"""Run all twelve examples in isolated CPU Docker environments."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

from build_triton_runtime import build_image, dockerfile
from real_models import CASES
from serving_validation import validate_serving

GROUPS = {
    "sklearn": (
        "python:3.11-slim",
        "trained21/sk_mixed_columns/bundle/requirements.txt",
    ),
    "pytorch": (
        "python:3.14-slim",
        "legacy59/results/pytorch_cac1a3dc99/artifact/requirements.txt",
    ),
    "transformers": ("python:3.11-slim", "trained21/hf_glue/bundle/requirements.txt"),
}
DEPENDENCIES = {
    "sklearn": {"numpy", "pandas", "scipy", "scikit-learn", "joblib"},
    "pytorch": {"numpy", "torch"},
    "transformers": {
        "numpy",
        "pandas",
        "torch",
        "transformers",
        "tokenizers",
        "accelerate",
        "huggingface_hub",
        "safetensors",
        "sentencepiece",
    },
}


def runtime_requirements(data, family):
    """Use saved inference versions; exclude unrelated training-only packages."""
    source = data / GROUPS[family][1]
    lines = source.read_text().splitlines()
    pins = {line.split("==", 1)[0]: line for line in lines if "==" in line and not line.startswith("--")}
    missing = DEPENDENCIES[family] - pins.keys()
    if missing:
        raise ValueError(f"Missing saved dependency pins in {source}: {sorted(missing)}")
    return (
        "\n".join(
            [
                "--extra-index-url https://download.pytorch.org/whl/cpu",
                "mlflow==3.17.2",
                "httpx>=0.27,<1",
                "cloudpickle==3.1.2",
                *(["torchvision==0.24.0+cpu"] if family == "transformers" else []),
                *[pins[name] for name in sorted(DEPENDENCIES[family])],
            ]
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="New directory; default: /mnt/container-data/mlflow-tests/<timestamp>-<id>",
    )
    parser.add_argument("--case", action="append", choices=CASES, help="Run selected cases only; repeatable")
    parser.add_argument("--family", action="append", choices=GROUPS, help="Default: all three")
    parser.add_argument("--timeout", type=int, default=3600, help="Seconds per family including install")
    parser.add_argument(
        "--triton-images",
        type=Path,
        help="Existing JSON image mapping; otherwise build matching CPU Triton images",
    )
    parser.add_argument("--triton-base-image", default="nvcr.io/nvidia/tritonserver:26.06-py3")
    parser.add_argument("--startup-timeout", type=int, default=180)
    parser.add_argument("--inference-timeout", type=int, default=180)
    args = parser.parse_args()
    if min(args.timeout, args.startup_timeout, args.inference_timeout) <= 0:
        parser.error("--timeout must be positive")
    if not shutil.which("docker"):
        parser.error("Docker is required; install/start Docker and retry")
    subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL)
    data = args.data_root.resolve(strict=True)
    if args.output is None:
        storage = Path("/mnt/container-data")
        if not storage.is_mount():
            parser.error("/mnt/container-data is not mounted; refusing to write tests onto the system disk")
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        output = storage / "mlflow-tests" / run_name
    else:
        output = args.output.resolve()
    plugin = Path(__file__).resolve().parents[1]
    requested = list(dict.fromkeys(args.case or CASES))
    if args.family:
        requested = [name for name in requested if CASES[name][0] in args.family]
    if not requested:
        parser.error("No cases match --case and --family")
    families = list(dict.fromkeys(CASES[name][0] for name in requested))
    images = json.loads(args.triton_images.read_text()) if args.triton_images else {}
    for family in families:
        version = "3.14" if family == "pytorch" else "3.11"
        if args.triton_images and (not isinstance(images.get(version), str) or not images[version]):
            parser.error(
                f"--triton-images needs a Triton image with Python {version} and matching backend stub"
            )
    # Fail preflight before installing anything or creating partial results.
    requirements = {family: runtime_requirements(data, family) for family in families}
    for name, (family, relative) in CASES.items():
        if name in requested and not (data / relative).is_dir():
            parser.error(f"Missing artifact for {name}: {data / relative}")
    try:
        output.mkdir(parents=True, exist_ok=False)
    except PermissionError:
        parser.error(f"No write permission for {output.parent}; create a user-owned test directory first")
    print(f"Test output: {output}", flush=True)
    if not args.triton_images:
        for version in sorted({"3.14" if f == "pytorch" else "3.11" for f in families}):
            folder = output / ("triton-runtime-python-" + version)
            folder.mkdir()
            (folder / "Dockerfile").write_text(dockerfile(args.triton_base_image, version))
            tag = "mlflow-triton-runtime:" + args.triton_base_image.split(":")[-1] + "-python" + version
            available = (
                subprocess.run(
                    ["docker", "image", "inspect", tag],
                    check=False,
                    capture_output=True,
                ).returncode
                == 0
            )
            if not available:
                print(
                    f"Building CPU Triton runtime for Python {version}; log: {folder / 'build.log'}",
                    flush=True,
                )
                if build_image(tag, folder, folder / "build.log"):
                    raise RuntimeError(f"Triton runtime build failed; see {folder / 'build.log'}")
            images[version] = tag
        (output / "triton-images.json").write_text(json.dumps(images, indent=2) + "\n")
    cache = output / "pip-cache"
    cache.mkdir()
    results = []
    for family in families:
        selected = [name for name in requested if CASES[name][0] == family]
        scratch = output / "work" / family
        scratch.mkdir(parents=True)
        requirements_file = output / f"{family}-requirements.txt"
        requirements_file.write_text(requirements[family])
        log = output / f"{family}.log"
        container = f"mlflow-examples-{os.getpid()}-{family}"
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            container,
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-e",
            "HOME=/tmp",
            "-e",
            "TMPDIR=/tmp",
            "-e",
            "USER=mlflow",
            "-e",
            "LOGNAME=mlflow",
            "-e",
            "PIP_CACHE_DIR=/pip-cache",
            "-e",
            "HF_HUB_OFFLINE=1",
            "-e",
            "TRANSFORMERS_OFFLINE=1",
            "-e",
            "HF_HOME=/tmp/huggingface",
            "-e",
            "OMP_NUM_THREADS=2",
            "-e",
            "PYTHONUNBUFFERED=1",
            "-e",
            "PYTHONDONTWRITEBYTECODE=1",
            "-e",
            f"PYTHONPATH={plugin}",
            # Keep output paths identical so SQLite artifact URIs work on the host.
            "-v",
            f"{plugin}:{plugin}:ro",
            "-v",
            f"{data}:{data}:ro",
            "-v",
            f"{output}:{output}:rw",
            "-v",
            f"{cache}:/pip-cache:rw",
            "-v",
            f"{scratch}:/tmp:rw",
            "-w",
            "/tmp",
            GROUPS[family][0],
            "sh",
            "-ec",
            (
                "python -m venv /tmp/runtime; "
                '/tmp/runtime/bin/python -m pip install -r "$1"; '
                'shift; exec /tmp/runtime/bin/python "$@"'
            ),
            "runner",
            str(requirements_file),
            str(plugin / "examples/real_models.py"),
            "--data-root",
            str(data),
            "--output",
            str(output / family),
        ]
        for name in selected:
            command.extend(["--case", name])
        print(f"Running {family}: {len(selected)} cases; log: {log}", flush=True)
        failure = None
        try:
            with log.open("w") as stream:
                process = subprocess.run(
                    command,
                    check=False,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=args.timeout,
                )
            if process.returncode:
                failure = f"Container exited {process.returncode}; see {log}"
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "rm", "-f", container], check=False, capture_output=True)
            failure = f"Family timed out after {args.timeout}s; see {log}"
        except KeyboardInterrupt:
            subprocess.run(["docker", "rm", "-f", container], check=False, capture_output=True)
            raise
        report = output / family / "summary.json"
        recorded = {row["case"]: row for row in json.loads(report.read_text())} if report.exists() else {}
        for name in selected:
            results.append(
                recorded.get(
                    name,
                    {
                        "case": name,
                        "local_validation": "not_run",
                        "triton_validation": "not_run",
                        "error": failure or f"Missing case result; see {log}",
                    },
                )
            )
        for row in results:
            if row["case"] not in selected:
                continue
            if row["local_validation"] != "passed":
                row["triton_validation"] = "blocked_by_local_failure"
                continue
            version = "3.14" if family == "pytorch" else "3.11"
            print(f"Triton serving: {row['case']}", flush=True)
            row.update(
                validate_serving(
                    row,
                    images[version],
                    version,
                    requirements_file,
                    output / "triton" / row["case"],
                    args,
                )
            )
        (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"Finished {family}: {failure or 'completed'}", flush=True)
    passed = sum(row["local_validation"] == "passed" for row in results)
    served = sum(row["triton_validation"] == "passed" for row in results)
    print(
        f"Local packaging/validation: {passed}/{len(results)} passed. Triton serving: {served}/{len(results)} passed."
    )
    print(f"Report: {output / 'summary.json'}")
    return int(passed != len(results) or served != len(results))


if __name__ == "__main__":
    raise SystemExit(main())
