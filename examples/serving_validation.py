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
                [item.decode("utf-8") if isinstance(item, bytes) else item for item in observed.ravel()]
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


def client_main():
    import numpy as np
    import tritonclient.http as http

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--startup-timeout", type=int, required=True)
    parser.add_argument("--inference-timeout", type=int, required=True)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()
    contract = json.loads(Path("/model/1/contract.json").read_text())
    with np.load("/fixture.npz", allow_pickle=False) as fixture:
        inputs = {k[6:]: fixture[k] for k in fixture.files if k.startswith("input:")}
        expected = {k[7:]: fixture[k] for k in fixture.files if k.startswith("output:")}
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
            raise AssertionError(f"Triton instance configuration does not match {args.device}")
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
            outputs=[http.InferRequestedOutput(s["name"], binary_data=True) for s in contract["outputs"]],
        )
        names = [item["name"] for item in response.get_response()["outputs"]]
        if len(names) != len(set(names)):
            raise AssertionError("Duplicate Triton output names")
        actual = {name: response.as_numpy(name) for name in names}
        compare_outputs(actual, expected, contract["outputs"])
        print(
            json.dumps(
                {
                    "triton_validation": "passed",
                    "requests": 1,
                    "comparison": "saved_native_artifact",
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
    result = {
        "device": device,
        "triton_validation": "failed",
        "triton_stage": "container_start",
        "triton_image": image,
    }

    def run(command, filename, timeout=60):
        with (directory / filename).open("a") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)

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
                "PYTHONPATH=/tmp/model-deps",
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
                "-v",
                f"{requirements}:/family-requirements.txt:ro",
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
                *(["--extra-index-url", "https://download.pytorch.org/whl/cpu"] if device == "cpu" else []),
                "-r",
                "/family-requirements.txt",
                "-r",
                "/model/requirements.txt",
                "tritonclient[http]",
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
                    "python3",
                    "-c",
                    "import torch; assert torch.version.cuda is not None, 'CPU-only PyTorch wheel'; "
                    "assert torch.cuda.is_available(), 'CUDA unavailable'; "
                    "x=torch.ones(1, device='cuda:0'); torch.cuda.synchronize(); "
                    "print(torch.version.cuda, torch.cuda.get_device_name(0), x.device)",
                ],
                "gpu.log",
                options.startup_timeout,
            )
        result["triton_stage"] = "server_start"
        # Inference needs no network access outside the container.
        run(["docker", "network", "disconnect", "bridge", name], "runtime.log")
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
            [
                "docker",
                "exec",
                name,
                "python3",
                "/validation.py",
                "--model",
                row["case"],
                "--device",
                device,
                "--startup-timeout",
                str(options.startup_timeout),
                "--inference-timeout",
                str(options.inference_timeout),
            ],
            "comparison.log",
            options.startup_timeout + options.inference_timeout + 30,
        )
        result.update(
            triton_validation="passed",
            triton_stage="complete",
            requests=1,
            comparison="saved_native_artifact",
            rtol=1e-5,
            atol=1e-6,
            equal_nan=True,
        )
    except Exception as error:
        result["triton_error"] = f"{type(error).__name__}: {error}; see {directory}"
    finally:
        try:
            run(["docker", "exec", name, "cat", "/tmp/triton.log"], "server.log", 30)
        except Exception:
            pass
        finally:
            try:
                subprocess.run(["docker", "rm", "-f", name], check=False, capture_output=True, timeout=30)
            except Exception as error:
                result["cleanup_error"] = str(error)
                result["triton_validation"] = "failed"
            (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    client_main()
