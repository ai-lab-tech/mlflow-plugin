"""Actual Triton CPU/GPU serving checks; fixtures stay outside the model package."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import uuid
from pathlib import Path


def compare_outputs(actual, expected, specs):
    import numpy as np

    names = {spec["name"] for spec in specs}
    if set(actual) != names or set(expected) != names:
        raise AssertionError("Output names differ from the complete declared contract")
    for spec in specs:
        name = spec["name"]
        observed, reference = actual[name], expected[name]
        if observed is None or observed.shape != reference.shape:
            raise AssertionError(f"Output shape mismatch: {name}")
        if spec["datatype"] == "BYTES":
            observed = np.asarray(
                [
                    item.decode("utf-8") if isinstance(item, bytes) else item
                    for item in observed.ravel()
                ]
            ).reshape(observed.shape)
            np.testing.assert_array_equal(observed, reference, err_msg=name)
        else:
            if observed.dtype != reference.dtype:
                raise AssertionError(f"Output dtype mismatch: {name}")
            if np.issubdtype(reference.dtype, np.floating):
                np.testing.assert_allclose(
                    observed,
                    reference,
                    rtol=1e-5,
                    atol=1e-6,
                    equal_nan=True,
                    err_msg=name,
                )
            else:
                np.testing.assert_array_equal(observed, reference, err_msg=name)


def comparison_result(actual, expected, specs):
    """Keep the existing tolerances and record differences without hiding them."""
    import numpy as np

    report = {
        "status": "passed",
        "rtol": 1e-5,
        "atol": 1e-6,
        "equal_nan": True,
        "outputs": {},
    }
    try:
        compare_outputs(actual, expected, specs)
    except AssertionError as error:
        report.update(status="failed", error=str(error))
    for spec in specs:
        name = spec["name"]
        a, b = actual.get(name), expected.get(name)
        if a is None or b is None or a.shape != b.shape:
            continue
        if np.issubdtype(a.dtype, np.floating) and np.issubdtype(b.dtype, np.floating):
            finite = np.isfinite(a) & np.isfinite(b)
            difference = np.abs(
                a[finite].astype(np.float64) - b[finite].astype(np.float64)
            )
            report["outputs"][name] = {
                "elements": int(a.size),
                "mismatched_elements": int(
                    np.count_nonzero(
                        ~np.isclose(a, b, rtol=1e-5, atol=1e-6, equal_nan=True)
                    )
                ),
                "max_absolute_finite_difference": float(difference.max())
                if difference.size
                else None,
            }
    return report


def latest_model_version(model_directory):
    """Select the numeric version served by the generated default latest policy."""
    versions = [
        path
        for path in Path(model_directory).iterdir()
        if path.is_dir()
        and path.name.isascii()
        and path.name.isdecimal()
        and int(path.name) > 0
        and str(int(path.name)) == path.name
    ]
    if not versions:
        raise ValueError("No numeric Triton model versions found")
    return max(versions, key=lambda path: int(path.name))


def _load_tensor_mapping(version):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "tensor_mapping", version / "model.py"
    )
    mapping = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mapping)
    return mapping


def read_model_contract(model_directory):
    """Read the generated config using Triton's protobuf schema (test tooling only)."""
    from google.protobuf import json_format, text_format
    from tritonclient.grpc import model_config_pb2

    root = Path(model_directory)
    config = model_config_pb2.ModelConfig()
    text_format.Parse((root / "config.pbtxt").read_text(), config)
    mapping = _load_tensor_mapping(latest_model_version(root))
    return mapping.contract_from_config(
        json_format.MessageToDict(config, preserving_proto_field_name=True)
    )


def native_prediction(model_directory, inputs, contract):
    """Predict with the latest artifact and its own tensor adapter."""
    import mlflow.pyfunc

    version = latest_model_version(model_directory)
    mapping = _load_tensor_mapping(version)
    model = mlflow.pyfunc.load_model(str(version / "artifact"))
    return mapping.encode_outputs(
        model.predict(mapping.decode_inputs(inputs, contract)), contract
    )


def native_gpu_reference(contract_path=None):
    """Independent native MLflow prediction in its own process, before Triton starts."""
    import os

    import numpy as np

    contract = (
        json.loads(Path(contract_path).read_text())
        if contract_path is not None
        else read_model_contract("/model")
    )
    with np.load("/fixture.npz", allow_pickle=False) as fixture:
        inputs = {k[6:]: fixture[k] for k in fixture.files if k.startswith("input:")}
        cpu = {k[7:]: fixture[k] for k in fixture.files if k.startswith("output:")}
    for entry in contract["inputs"]:
        if entry["datatype"] == "BYTES":
            inputs[entry["name"]] = inputs[entry["name"]].astype(object)
    os.environ["MLFLOW_DEFAULT_PREDICTION_DEVICE"] = (
        "0" if contract["flavor"] == "transformers" else "cuda:0"
    )
    os.environ["MLFLOW_HUGGINGFACE_USE_DEVICE_MAP"] = "false"
    # Reuse schema conversion only. Do not instantiate TritonPythonModel or execute().
    predicted = native_prediction("/model", inputs, contract)
    saved = {}
    for name, value in predicted.items():
        if value.dtype == object:
            value = np.asarray(
                [
                    item.decode("utf-8") if isinstance(item, bytes) else item
                    for item in value.ravel()
                ],
                dtype=str,
            ).reshape(value.shape)
        saved[name] = value
    np.savez("/tmp/native-gpu.npz", **saved)
    report = {"native_gpu_vs_cpu": comparison_result(saved, cpu, contract["outputs"])}
    Path("/tmp/comparison-details.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def client_main():
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--startup-timeout", type=int, default=180)
    parser.add_argument("--inference-timeout", type=int, default=180)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--native-gpu-reference", action="store_true")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--write-contract", type=Path)
    args = parser.parse_args()
    if args.write_contract is not None:
        args.write_contract.write_text(
            json.dumps(read_model_contract("/model"), indent=2) + "\n"
        )
        return
    if args.native_gpu_reference:
        native_gpu_reference(args.contract)
        return
    import tritonclient.http as http

    contract = read_model_contract("/model")
    with np.load("/fixture.npz", allow_pickle=False) as fixture:
        inputs = {k[6:]: fixture[k] for k in fixture.files if k.startswith("input:")}
        expected = {k[7:]: fixture[k] for k in fixture.files if k.startswith("output:")}
    cpu_expected = expected
    details = {}
    if args.device == "gpu":
        with np.load("/tmp/native-gpu.npz", allow_pickle=False) as reference:
            expected = {key: reference[key] for key in reference.files}
        details = json.loads(Path("/tmp/comparison-details.json").read_text())
    with http.InferenceServerClient(
        "localhost:8000",
        connection_timeout=5,
        network_timeout=args.inference_timeout,
    ) as client:
        deadline = time.monotonic() + args.startup_timeout
        while True:
            try:
                if client.is_server_ready() and client.is_model_ready(args.model):
                    break
            except Exception:  # Startup may transiently refuse connections.
                pass
            if time.monotonic() >= deadline:
                raise TimeoutError("Triton server/model did not become ready")
            time.sleep(1)
        config = client.get_model_config(args.model)
        groups = config.get("instance_group", [])
        expected_kind = "KIND_GPU" if args.device == "gpu" else "KIND_CPU"
        if not groups or any(group.get("kind") != expected_kind for group in groups):
            raise AssertionError(
                f"Triton instance configuration does not match {args.device}"
            )
        request = []
        for spec in contract["inputs"]:
            array = inputs[spec["name"]]
            if spec["datatype"] == "BYTES":
                array = array.astype(object)
            tensor = http.InferInput(spec["name"], list(array.shape), spec["datatype"])
            tensor.set_data_from_numpy(array, binary_data=True)
            request.append(tensor)
        response = client.infer(
            args.model,
            request,
            outputs=[
                http.InferRequestedOutput(s["name"], binary_data=True)
                for s in contract["outputs"]
            ],
        )
        names = [item["name"] for item in response.get_response()["outputs"]]
        if len(names) != len(set(names)):
            raise AssertionError("Duplicate Triton output names")
        actual = {name: response.as_numpy(name) for name in names}
        comparison_key = (
            "triton_vs_native_gpu" if args.device == "gpu" else "triton_vs_cpu"
        )
        details[comparison_key] = comparison_result(
            actual, expected, contract["outputs"]
        )
        if args.device == "gpu":
            details["triton_vs_cpu"] = comparison_result(
                actual, cpu_expected, contract["outputs"]
            )
        Path("/tmp/comparison-details.json").write_text(
            json.dumps(details, indent=2) + "\n"
        )
        compare_outputs(actual, expected, contract["outputs"])
        print(
            json.dumps(
                {
                    "triton_validation": "passed",
                    "requests": 1,
                    "comparison": "saved_native_artifact_gpu"
                    if args.device == "gpu"
                    else "saved_native_artifact_cpu",
                    "rtol": 1e-5,
                    "atol": 1e-6,
                    "equal_nan": True,
                }
            )
        )


def validate_serving(row, image, python_version, requirements, directory, options):
    directory.mkdir(parents=True, exist_ok=False)
    model = Path(row["serving_path"]).resolve()
    fixture = Path(row["fixture_path"]).resolve()
    helper = Path(__file__).resolve()
    name = "mlflow-serving-" + uuid.uuid4().hex[:12]
    device = getattr(options, "device", "cpu")
    requirements_only = getattr(options, "requirements_only", False)
    model_python = "/tmp/model-runtime/bin/python" if requirements_only else "python3"
    model_site = f"/tmp/model-runtime/lib/python{python_version}/site-packages"
    client_site = f"/tmp/client-runtime/lib/python{python_version}/site-packages"
    result = {
        "device": device,
        "triton_validation": "failed",
        "triton_stage": "container_start",
        "triton_image": image,
        "dependency_policy": (
            "generated_requirements_only" if requirements_only else "family_and_model"
        ),
        "client_environment": "separate_virtualenv" if requirements_only else "shared",
    }

    def run(command, filename, timeout=60):
        with (directory / filename).open("a") as log:
            subprocess.run(
                command,
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )

    def client_command(*arguments):
        if requirements_only:
            return [
                "docker",
                "exec",
                "-e",
                f"PYTHONPATH={client_site}",
                name,
                "/tmp/client-runtime/bin/python",
                *arguments,
            ]
        return ["docker", "exec", name, "python3", *arguments]

    try:
        run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                name,
                *(
                    [
                        "--gpus",
                        "all",
                        "-e",
                        "NVIDIA_VISIBLE_DEVICES=all",
                        "-e",
                        "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
                        "-e",
                        "CUDA_VISIBLE_DEVICES=0",
                    ]
                    if device == "gpu"
                    else [
                        "--runtime",
                        "runc",
                        "-e",
                        "CUDA_VISIBLE_DEVICES=-1",
                        "-e",
                        "NVIDIA_VISIBLE_DEVICES=void",
                    ]
                ),
                "--entrypoint",
                "sleep",
                "--shm-size",
                "1g",
                "-e",
                "PYTHONPATH="
                + (model_site if requirements_only else "/tmp/model-deps"),
                "-e",
                "PYTHONNOUSERSITE=1",
                "-e",
                "PYTHONDONTWRITEBYTECODE=1",
                "-e",
                "HF_HUB_OFFLINE=1",
                "-e",
                "TRANSFORMERS_OFFLINE=1",
                "-e",
                "OMP_NUM_THREADS=2",
                "-e",
                "TOKENIZERS_PARALLELISM=false",
                "-v",
                f"{model}:/models/{row['case']}:ro",
                "-v",
                f"{model}:/model:ro",
                "-v",
                f"{fixture}:/fixture.npz:ro",
                *(
                    []
                    if requirements_only
                    else ["-v", f"{requirements}:/family-requirements.txt:ro"]
                ),
                "-v",
                f"{helper}:/validation.py:ro",
                image,
                "infinity",
            ],
            "runtime.log",
            300,
        )
        result["triton_stage"] = "python_compatibility"
        version = subprocess.check_output(
            [
                "docker",
                "exec",
                name,
                "python3",
                "-c",
                "import sys;print(str(sys.version_info.major)+'.'+str(sys.version_info.minor))",
            ],
            text=True,
            timeout=30,
        ).strip()
        libraries = subprocess.check_output(
            [
                "docker",
                "exec",
                name,
                "ldd",
                "/opt/tritonserver/backends/python/triton_python_backend_stub",
            ],
            text=True,
            timeout=30,
        )
        result["runtime_python"] = version
        (directory / "stub-libraries.txt").write_text(libraries)
        linked = re.search(r"libpython(\d+\.\d+)", libraries)
        if version != python_version or not linked or linked[1] != version:
            raise ValueError(
                f"Need Python {python_version} and matching Triton stub; found interpreter {version}"
            )
        result["triton_stage"] = "dependency_install"
        if requirements_only:
            (directory / "model-requirements.txt").write_bytes(
                (model / "requirements.txt").read_bytes()
            )
            run(
                [
                    "docker",
                    "exec",
                    name,
                    "python3",
                    "-m",
                    "venv",
                    "/tmp/model-runtime",
                ],
                "install.log",
                options.timeout,
            )
            run(
                [
                    "docker",
                    "exec",
                    name,
                    model_python,
                    "-m",
                    "pip",
                    "install",
                    "--report",
                    "/tmp/model-install-report.json",
                    "-r",
                    "/model/requirements.txt",
                ],
                "install.log",
                options.timeout,
            )
            run(
                ["docker", "exec", name, model_python, "-m", "pip", "check"],
                "dependency-check.log",
                options.timeout,
            )
            run(
                ["docker", "exec", name, model_python, "-m", "pip", "freeze"],
                "requirements-lock.txt",
                options.timeout,
            )
            result["dependency_validation"] = "passed"
        else:
            run(
                [
                    "docker",
                    "exec",
                    name,
                    "python3",
                    "-m",
                    "pip",
                    "install",
                    "--target",
                    "/tmp/model-deps",
                    "--upgrade",
                    *(
                        ["--extra-index-url", "https://download.pytorch.org/whl/cpu"]
                        if device == "cpu"
                        else []
                    ),
                    "-r",
                    "/family-requirements.txt",
                    "-r",
                    "/model/requirements.txt",
                    "tritonclient[http,grpc]",
                ],
                "install.log",
                options.timeout,
            )
        if device == "gpu":
            result["triton_stage"] = "gpu_availability"
            run(
                [
                    "docker",
                    "exec",
                    name,
                    model_python,
                    "-c",
                    "import torch; assert torch.version.cuda is not None, 'CPU-only PyTorch wheel'; "
                    "assert torch.cuda.is_available(), 'CUDA unavailable'; "
                    "x=torch.ones(1, device='cuda:0'); torch.cuda.synchronize(); "
                    "print(torch.version.cuda, torch.cuda.get_device_name(0), x.device)",
                ],
                "gpu.log",
                options.startup_timeout,
            )
        if requirements_only:
            result["triton_stage"] = "client_dependency_install"
            run(
                [
                    "docker",
                    "exec",
                    "-e",
                    "PYTHONPATH=",
                    name,
                    "python3",
                    "-m",
                    "venv",
                    "/tmp/client-runtime",
                ],
                "client-install.log",
                options.timeout,
            )
            run(
                client_command(
                    "-m",
                    "pip",
                    "install",
                    "--report",
                    "/tmp/client-install-report.json",
                    "tritonclient[http,grpc]",
                    "pandas",
                ),
                "client-install.log",
                options.timeout,
            )
            run(
                client_command(
                    "/validation.py",
                    "--model",
                    row["case"],
                    "--write-contract",
                    "/tmp/model-contract.json",
                ),
                "client-contract.log",
                options.startup_timeout,
            )
        result["triton_stage"] = "server_start"
        # Inference needs no network access outside the container.
        run(["docker", "network", "disconnect", "bridge", name], "runtime.log")
        if device == "gpu":
            result["triton_stage"] = "native_gpu_reference"
            run(
                [
                    "docker",
                    "exec",
                    name,
                    model_python,
                    "/validation.py",
                    "--model",
                    row["case"],
                    "--device",
                    "gpu",
                    "--native-gpu-reference",
                    *(
                        ["--contract", "/tmp/model-contract.json"]
                        if requirements_only
                        else []
                    ),
                ],
                "native-gpu.log",
                options.inference_timeout,
            )
        result["triton_stage"] = "server_start"
        run(
            [
                "docker",
                "exec",
                "-d",
                name,
                "sh",
                "-ec",
                "exec /opt/tritonserver/bin/tritonserver --model-repository=/models "
                "--strict-model-config=true --allow-grpc=false --allow-metrics=false "
                "--pinned-memory-pool-byte-size=0 --exit-timeout-secs=10 "
                "> /tmp/triton.log 2>&1",
            ],
            "runtime.log",
        )
        result["triton_stage"] = "inference_comparison"
        run(
            client_command(
                "/validation.py",
                "--model",
                row["case"],
                "--device",
                device,
                "--startup-timeout",
                str(options.startup_timeout),
                "--inference-timeout",
                str(options.inference_timeout),
            ),
            "comparison.log",
            options.startup_timeout + options.inference_timeout + 30,
        )
        result.update(
            triton_validation="passed",
            triton_stage="complete",
            requests=1,
            comparison="saved_native_artifact_gpu"
            if device == "gpu"
            else "saved_native_artifact_cpu",
            rtol=1e-5,
            atol=1e-6,
            equal_nan=True,
        )
    except Exception as error:
        result["triton_error"] = f"{type(error).__name__}: {error}; see {directory}"
    finally:
        try:
            details = json.loads(
                subprocess.check_output(
                    ["docker", "exec", name, "cat", "/tmp/comparison-details.json"],
                    text=True,
                    timeout=30,
                )
            )
            required_key = (
                "triton_vs_native_gpu" if device == "gpu" else "triton_vs_cpu"
            )
            if (
                result["triton_validation"] == "passed"
                and details.get(required_key, {}).get("status") != "passed"
            ):
                raise ValueError("Missing or unsuccessful primary comparison report")
            if (
                result["triton_validation"] == "passed"
                and device == "gpu"
                and "native_gpu_vs_cpu" not in details
            ):
                raise ValueError("Missing CPU/GPU diagnostic comparison")
            for key in ["native_gpu_vs_cpu", "triton_vs_native_gpu", "triton_vs_cpu"]:
                if key in details:
                    result[key] = details[key]
            (directory / "comparison-details.json").write_text(
                json.dumps(details, indent=2) + "\n"
            )
        except Exception as error:
            result["comparison_details_error"] = str(error)
            if result["triton_validation"] == "passed":
                result["triton_validation"] = "failed"
        try:
            run(["docker", "exec", name, "cat", "/tmp/triton.log"], "server.log", 30)
        except Exception:
            pass
        finally:
            if requirements_only:
                for artifact in [
                    "model-install-report.json",
                    "client-install-report.json",
                ]:
                    try:
                        run(
                            ["docker", "exec", name, "cat", "/tmp/" + artifact],
                            artifact,
                            30,
                        )
                    except Exception as error:
                        result.setdefault("dependency_audit_errors", []).append(
                            str(error)
                        )
                        result["triton_validation"] = "failed"
            try:
                subprocess.run(
                    ["docker", "rm", "-f", name],
                    check=False,
                    capture_output=True,
                    timeout=30,
                )
            except Exception as error:
                result["cleanup_error"] = str(error)
                result["triton_validation"] = "failed"
            (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    client_main()
