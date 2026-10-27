"""Build reusable CPU/GPU Triton images with matching Python interpreters and stubs."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def dockerfile(image, version, device="cpu"):
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be cpu or gpu")
    release = re.fullmatch(r"nvcr\.io/nvidia/tritonserver:(\d{2}\.\d{2})-py3", image)
    if not release or not re.fullmatch(r"3\.\d+", version):
        raise ValueError("Use a versioned NVIDIA Triton py3 image and a Python 3.x version")
    branch = "r" + release[1]
    gpu = "ON" if device == "gpu" else "OFF"
    visibility = (
        "ENV CUDA_VISIBLE_DEVICES=-1 NVIDIA_VISIBLE_DEVICES=void"
        if device == "cpu"
        else "ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility"
    )
    return f'''FROM mambaorg/micromamba:2.3.3 AS mamba
FROM {image}
USER root
COPY --from=mamba /bin/micromamba /usr/local/bin/micromamba
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential cmake git ca-certificates rapidjson-dev libarchive-dev zlib1g-dev \\
    && rm -rf /var/lib/apt/lists/*
ENV MAMBA_ROOT_PREFIX=/opt/mamba
RUN micromamba create -y -p /opt/triton-test-python -c conda-forge python={version} pip numpy \\
    && micromamba clean --all --yes
ENV PATH=/opt/triton-test-python/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/triton-test-python/lib:$LD_LIBRARY_PATH
# Build tools use the base interpreter, independently of the legacy model Python.
# Use the wheel's native CMake executable so it never imports into Python {version}.
RUN /usr/bin/python3 -m pip install --no-cache-dir --only-binary=:all: --timeout 120 --retries 5 \\
    --target /opt/triton-build-tools cmake==3.31.10 \\
    && /opt/triton-build-tools/cmake/data/bin/cmake --version
RUN git clone --depth 1 --branch {branch} https://github.com/triton-inference-server/python_backend /tmp/python_backend \\
    && /opt/triton-build-tools/cmake/data/bin/cmake -S /tmp/python_backend -B /tmp/stub-build \\
       -DTRITON_ENABLE_GPU={gpu} -DTRITON_ENABLE_METRICS=OFF \\
       -DTRITON_BACKEND_REPO_TAG={branch} -DTRITON_COMMON_REPO_TAG={branch} -DTRITON_CORE_REPO_TAG={branch} \\
       -DPYTHON_EXECUTABLE=/opt/triton-test-python/bin/python \\
       -DPython_EXECUTABLE=/opt/triton-test-python/bin/python \\
       -DPython3_EXECUTABLE=/opt/triton-test-python/bin/python \\
    && /opt/triton-build-tools/cmake/data/bin/cmake --build /tmp/stub-build --target triton-python-backend-stub -j2 \\
    && cp /tmp/stub-build/triton_python_backend_stub /opt/tritonserver/backends/python/triton_python_backend_stub \\
    && ldd /opt/tritonserver/backends/python/triton_python_backend_stub | grep 'libpython{version}' \\
    && rm -rf /tmp/python_backend /tmp/stub-build
RUN python3 -c 'import sys; assert sys.version_info[:2] == tuple(map(int, "{version}".split(".")))'
{visibility}
'''


def build_image(tag, folder, logfile):
    """Show and preserve complete BuildKit diagnostics, including the first error."""
    with logfile.open("w") as log:
        process = subprocess.Popen(
            ["docker", "build", "--progress=plain", "--tag", tag, str(folder)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return process.wait()
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", nargs="+", required=True, dest="versions")
    parser.add_argument("--image", default="nvcr.io/nvidia/tritonserver:26.06-py3")
    parser.add_argument("--output", type=Path, default=Path("generated/triton-runtime-images.json"))
    parser.add_argument(
        "--prepare-only", action="store_true", help="Write Dockerfiles without running Docker"
    )
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mapping = json.loads(args.output.read_text()) if args.output.exists() else {}
    for version in args.versions:
        source = dockerfile(args.image, version, args.device)
        folder = args.output.parent / ("triton-runtime-python-" + version)
        folder.mkdir(exist_ok=True)
        (folder / "Dockerfile").write_text(source)
        release = args.image.split(":")[1]
        tag = "triton-generator-runtime:{}-python{}".format(release, version)
        if args.device == "gpu":
            tag += "-gpu"
        if args.prepare_only:
            print("Prepared Dockerfile: " + str(folder / "Dockerfile"), flush=True)
            continue
        logfile = folder / "build.log"
        if build_image(tag, folder, logfile):
            print("Runtime build failed; full output: " + str(logfile), file=sys.stderr)
            return 1
        mapping[version] = tag
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(mapping, indent=2) + "\n")
        temporary.replace(args.output)
    if not args.prepare_only:
        print("Runtime mapping: " + str(args.output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
