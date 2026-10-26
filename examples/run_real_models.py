#!/usr/bin/env python3
"""Run all twelve examples in isolated CPU Docker environments."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from real_models import CASES

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
    pins = {
        line.split("==", 1)[0]: line
        for line in lines
        if "==" in line and not line.startswith("--")
    }
    missing = DEPENDENCIES[family] - pins.keys()
    if missing:
        raise ValueError(
            f"Missing saved dependency pins in {source}: {sorted(missing)}"
        )
    return (
        "\n".join(
            [
                "--extra-index-url https://download.pytorch.org/whl/cpu",
                "mlflow==3.17.2",
                "httpx>=0.27,<1",
                "cloudpickle==3.1.2",
                *[pins[name] for name in sorted(DEPENDENCIES[family])],
            ]
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument(
        "--output", required=True, type=Path, help="New directory; never overwritten"
    )
    parser.add_argument(
        "--family", action="append", choices=GROUPS, help="Default: all three"
    )
    parser.add_argument(
        "--timeout", type=int, default=3600, help="Seconds per family including install"
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if not shutil.which("docker"):
        parser.error("Docker is required; install/start Docker and retry")
    subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL)
    data = args.data_root.resolve(strict=True)
    output = args.output.resolve()
    plugin = Path(__file__).resolve().parents[1]
    families = list(dict.fromkeys(args.family or GROUPS))
    # Fail preflight before installing anything or creating partial results.
    requirements = {family: runtime_requirements(data, family) for family in families}
    for name, (family, relative) in CASES.items():
        if family in families and not (data / relative).is_dir():
            parser.error(f"Missing artifact for {name}: {data / relative}")
    output.mkdir(parents=True, exist_ok=False)
    cache = output / "pip-cache"
    cache.mkdir()
    results = []
    for family in families:
        selected = [name for name, (kind, _) in CASES.items() if kind == family]
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
                        "local_validation": "not_run",
                        "triton_validation": "not_run",
                        "error": failure or f"Missing case result; see {log}",
                    },
                )
            )
        (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"Finished {family}: {failure or 'completed'}", flush=True)
    passed = sum(row["local_validation"] == "passed" for row in results)
    print(
        f"Local packaging/validation: {passed}/{len(results)} passed. Triton: not run."
    )
    print(f"Report: {output / 'summary.json'}")
    return int(passed != len(results))


if __name__ == "__main__":
    raise SystemExit(main())
