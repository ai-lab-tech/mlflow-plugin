#!/usr/bin/env python3
"""Run saved-model examples in isolated Docker environments."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Keep imports from creating __pycache__ in the source checkout.
sys.dont_write_bytecode = True

from build_triton_runtime import build_image, dockerfile  # noqa: E402 - disable bytecode writes before importing local modules
from real_models import CASES, DEFAULT_DATA_ROOT  # noqa: E402 - disable bytecode writes before importing local modules
from serving_validation import validate_serving  # noqa: E402 - disable bytecode writes before importing local modules

GROUPS = {
    "sklearn": (
        "python:3.11-slim",
        "sklearn_mixed_columns/bundle/requirements.txt",
    ),
    "pytorch": (
        "python:3.14-slim",
        "pytorch_mnist_lstm/artifact/requirements.txt",
    ),
    "transformers": ("python:3.11-slim", "huggingface_classification/bundle/requirements.txt"),
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


def runtime_requirements(data, family, device="cpu"):
    """Use saved inference versions; exclude unrelated training-only packages."""
    source = data / GROUPS[family][1]
    lines = source.read_text().splitlines()
    pins = {
        line.split("==", 1)[0]: line
        for line in lines
        if "==" in line and not line.startswith("--")
    }
    if device == "gpu":
        pins = {name: pin.replace("+cpu", "") for name, pin in pins.items()}
    missing = DEPENDENCIES[family] - pins.keys()
    if missing:
        raise ValueError(
            f"Missing saved dependency pins in {source}: {sorted(missing)}"
        )
    return (
        "\n".join(
            [
                *(
                    ["--extra-index-url https://download.pytorch.org/whl/cpu"]
                    if device == "cpu"
                    else []
                ),
                "mlflow==3.17.2",
                "httpx>=0.27,<1",
                "cloudpickle==3.1.2",
                *(
                    ["torchvision==0.24.0" + ("+cpu" if device == "cpu" else "")]
                    if family == "transformers"
                    else []
                ),
                *[pins[name] for name in sorted(DEPENDENCIES[family])],
            ]
        )
        + "\n"
    )


def test_output_path(output=None, output_root=None):
    """Choose explicitly configured storage without assuming a host layout."""
    if output is not None:
        return Path(output).expanduser().resolve()
    base = output_root or os.environ.get("MLFLOW_TRITON_TEST_ROOT")
    if not base:
        raise ValueError(
            "Choose test storage with --output-root, --output or MLFLOW_TRITON_TEST_ROOT"
        )
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    return Path(base).expanduser().resolve() / run_name


def check_docker_storage():
    root = subprocess.check_output(
        ["docker", "info", "--format", "{{.DockerRootDir}}"], text=True, timeout=30
    ).strip()
    if not root:
        raise ValueError("Docker did not report its storage directory")
    return Path(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Saved artifacts; default: examples/data in this checkout",
    )
    destinations = parser.add_mutually_exclusive_group()
    destinations.add_argument(
        "--output",
        type=Path,
        help="Exact new output directory (never overwritten)",
    )
    destinations.add_argument(
        "--output-root",
        type=Path,
        help="Base directory for timestamped runs; alternatively set MLFLOW_TRITON_TEST_ROOT",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=CASES,
        help="Run selected cases only; repeatable",
    )
    parser.add_argument(
        "--family", action="append", choices=GROUPS, help="Default: all three"
    )
    parser.add_argument(
        "--timeout", type=int, default=3600, help="Seconds per family including install"
    )
    parser.add_argument(
        "--triton-images",
        type=Path,
        help="Existing JSON image mapping; otherwise build matching Triton images for --device",
    )
    parser.add_argument(
        "--triton-base-image", default="nvcr.io/nvidia/tritonserver:26.06-py3"
    )
    parser.add_argument("--startup-timeout", type=int, default=180)
    parser.add_argument("--inference-timeout", type=int, default=180)
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Triton serving device; GPU defaults to PyTorch/Hugging Face cases",
    )
    args = parser.parse_args()
    if min(args.timeout, args.startup_timeout, args.inference_timeout) <= 0:
        parser.error("--timeout must be positive")
    if not shutil.which("docker"):
        parser.error("Docker is required; install/start Docker and retry")
    subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL)
    data = args.data_root.resolve(strict=True)
    try:
        output = test_output_path(args.output, args.output_root)
        docker_storage = check_docker_storage()
    except (ValueError, OSError) as error:
        parser.error(str(error))
    plugin = Path(__file__).resolve().parents[1]
    requested = list(dict.fromkeys(args.case or CASES))
    if args.family:
        requested = [name for name in requested if CASES[name][0] in args.family]
    if args.device == "gpu":
        if (args.case or args.family) and any(
            CASES[name][0] == "sklearn" for name in requested
        ):
            parser.error(
                "sklearn supports CPU only; select PyTorch/Hugging Face for --device gpu"
            )
        requested = [name for name in requested if CASES[name][0] != "sklearn"]
    if not requested:
        parser.error("No cases match --case and --family")
    families = list(dict.fromkeys(CASES[name][0] for name in requested))
    images = json.loads(args.triton_images.read_text()) if args.triton_images else {}
    for family in families:
        version = "3.14" if family == "pytorch" else "3.11"
        if args.triton_images and (
            not isinstance(images.get(version), str) or not images[version]
        ):
            parser.error(
                f"--triton-images needs a Triton image with Python {version} and matching backend stub"
            )
    # Fail preflight before installing anything or creating partial results.
    requirements = {
        family: runtime_requirements(data, family, args.device) for family in families
    }
    for name, (family, relative) in CASES.items():
        if name in requested and not (data / relative).is_dir():
            parser.error(f"Missing artifact for {name}: {data / relative}")
    try:
        output.mkdir(parents=True, exist_ok=False)
    except PermissionError:
        parser.error(
            f"No write permission for {output.parent}; create a user-owned test directory first"
        )
    print(f"Test output: {output}; Docker storage: {docker_storage}", flush=True)
    host_tmp = output / "host-tmp"
    host_tmp.mkdir()
    os.environ["TMPDIR"] = str(host_tmp)
    os.environ["TMP"] = str(host_tmp)
    os.environ["TEMP"] = str(host_tmp)
    if not args.triton_images:
        for version in sorted({"3.14" if f == "pytorch" else "3.11" for f in families}):
            folder = output / ("triton-runtime-python-" + version)
            folder.mkdir()
            (folder / "Dockerfile").write_text(
                dockerfile(args.triton_base_image, version, args.device)
            )
            tag = (
                "mlflow-triton-runtime:"
                + args.triton_base_image.split(":")[-1]
                + "-python"
                + version
            )
            if args.device == "gpu":
                tag += "-gpu"
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
                    f"Building {args.device.upper()} Triton runtime for Python {version}; log: {folder / 'build.log'}",
                    flush=True,
                )
                if build_image(tag, folder, folder / "build.log"):
                    raise RuntimeError(
                        f"Triton runtime build failed; see {folder / 'build.log'}"
                    )
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
            f"PYTHONPATH={plugin / 'packages/mlflow-triton/src'}",
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
            "--device",
            args.device,
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
            subprocess.run(
                ["docker", "rm", "-f", container], check=False, capture_output=True
            )
            failure = f"Family timed out after {args.timeout}s; see {log}"
        except KeyboardInterrupt:
            subprocess.run(
                ["docker", "rm", "-f", container], check=False, capture_output=True
            )
            raise
        report = output / family / "summary.json"
        recorded = (
            {row["case"]: row for row in json.loads(report.read_text())}
            if report.exists()
            else {}
        )
        for name in selected:
            results.append(
                recorded.get(
                    name,
                    {
                        "case": name,
                        "device": args.device,
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
