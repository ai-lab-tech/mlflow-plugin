"""Generate runnable Triton HTTP examples from validated native inputs."""

import json
import struct

import numpy as np

from . import _runtime


def write_request_example(repository, example, contract):
    tensors = _runtime.example_tensors(example, contract)
    binary = any(
        (spec["datatype"] == "BYTES" and spec["logical_type"] == "binary")
        or (
            np.issubdtype(tensors[spec["name"]].dtype, np.floating)
            and not np.isfinite(tensors[spec["name"]]).all()
        )
        for spec in contract["inputs"]
    )
    payload = {
        "inputs": [],
        "outputs": [{"name": spec["name"]} for spec in contract["outputs"]],
    }
    blocks = []
    for spec in contract["inputs"]:
        array = tensors[spec["name"]]
        entry = {
            "name": spec["name"],
            "shape": list(array.shape),
            "datatype": spec["datatype"],
        }
        if binary:
            if spec["datatype"] == "BYTES":
                parts = []
                for value in array.ravel():
                    value = (
                        value.encode("utf-8")
                        if isinstance(value, str)
                        else bytes(value)
                    )
                    parts.append(struct.pack("<I", len(value)) + value)
                block = b"".join(parts)
            else:
                block = array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes(
                    order="C"
                )
            entry["parameters"] = {"binary_data_size": len(block)}
            blocks.append(block)
        else:
            entry["data"] = array.ravel().tolist()
        payload["inputs"].append(entry)

    if binary:
        for output in payload["outputs"]:
            output["parameters"] = {"binary_data": True}
        header = (
            json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        (repository / "request.headers").write_text(
            "Content-Type: application/octet-stream\n"
            f"Inference-Header-Content-Length: {len(header)}\n"
        )
        with (repository / "request.bin").open("wb") as stream:
            stream.write(header)
            for block in blocks:
                stream.write(block)
    else:
        (repository / "request.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
        )
