"""Deterministic conversion of MLflow signatures to Triton tensor contracts."""

import json
import re

import numpy as np
from mlflow.types.schema import ColSpec, DataType, TensorSpec

_NUMPY = {
    "bool": "BOOL",
    "uint8": "UINT8",
    "uint16": "UINT16",
    "uint32": "UINT32",
    "uint64": "UINT64",
    "int8": "INT8",
    "int16": "INT16",
    "int32": "INT32",
    "int64": "INT64",
    "float16": "FP16",
    "float32": "FP32",
    "float64": "FP64",
}
_COLUMNS = {
    DataType.boolean: "bool",
    DataType.integer: "int32",
    DataType.long: "int64",
    DataType.float: "float32",
    DataType.double: "float64",
    DataType.string: "str",
    DataType.binary: "bytes",
}


def _schema(schema, direction):
    if schema is None or len(schema.inputs) == 0:
        raise ValueError("A complete input and output signature is required")
    entries = schema.inputs
    tensor_based = all(isinstance(s, TensorSpec) for s in entries)
    if not tensor_based and not all(isinstance(s, ColSpec) for s in entries):
        raise ValueError("Mixed column/tensor signatures are unsupported")
    if tensor_based and len(entries) > 1 and any(s.name is None for s in entries):
        raise ValueError(
            "Multiple tensors require explicit names in the MLflow signature"
        )
    specs, used = [], set()
    for index, item in enumerate(entries):
        source = item.name if item.name is not None else index
        name = str(source) if item.name is not None else f"{direction}_{index}"
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or name in used:
            name = f"{direction}_{index}"
            while name in used:
                name += "_"
        used.add(name)
        if tensor_based:
            dtype = np.dtype(item.type)
            dims = list(item.shape)
            logical = "binary" if dtype.kind == "S" else "string"
        else:
            if not item.required:
                raise ValueError("Optional columns require an explicit serving adapter")
            if item.type not in _COLUMNS:
                raise ValueError(
                    "Nested/object/datetime columns require an explicit serving adapter"
                )
            dtype = np.dtype(_COLUMNS[item.type])
            dims = [-1]
            logical = "binary" if item.type == DataType.binary else "string"
        if not dims or any(type(d) is not int or d == 0 or d < -1 for d in dims):
            raise ValueError(
                "Triton requires non-scalar tensor shapes with positive dimensions or -1"
            )
        datatype = "BYTES" if dtype.kind in "OUS" else _NUMPY.get(dtype.name)
        if datatype is None:
            raise ValueError("Unsupported Triton dtype: " + str(dtype))
        specs.append(
            {
                "name": name,
                "source_name": source,
                "named": item.name is not None,
                "datatype": datatype,
                "numpy_dtype": dtype.name,
                "logical_type": logical,
                "dims": dims,
            }
        )
    mode = (
        ("tensor" if len(entries) == 1 and entries[0].name is None else "named_tensors")
        if tensor_based
        else "columns"
    )
    return specs, mode


def contract_from_signature(signature, flavor):
    if signature is None:
        raise ValueError(
            "MLflow did not infer a signature; supply an explicit signature and valid input_example"
        )
    inputs, input_mode = _schema(signature.inputs, "input")
    outputs, output_mode = _schema(signature.outputs, "output")
    return {
        "flavor": flavor,
        "inputs": inputs,
        "outputs": outputs,
        "input_mode": input_mode,
        "output_mode": output_mode,
    }


def validate_serving_options(name, flavor, device):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", name):
        raise ValueError(
            "triton_model_name must start with a letter/underscore and contain letters, digits, _ or -"
        )
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be cpu or gpu")
    if flavor == "sklearn" and device != "cpu":
        raise ValueError("The sklearn wrapper requires device=cpu")


def model_config(name, contract, device):
    """Build Triton's configuration, retaining only MLflow-specific mapping data."""
    validate_serving_options(name, contract["flavor"], device)
    mapping = {
        "version": 1,
        "flavor": contract["flavor"],
        "input_mode": contract["input_mode"],
        "output_mode": contract["output_mode"],
    }
    config = {"name": name, "backend": "python", "max_batch_size": 0}
    for direction in ("input", "output"):
        specs = contract[direction + "s"]
        config[direction] = [
            {
                "name": spec["name"],
                "data_type": "TYPE_"
                + ("STRING" if spec["datatype"] == "BYTES" else spec["datatype"]),
                "dims": spec["dims"],
            }
            for spec in specs
        ]
        mapping[direction + "s"] = [
            {key: spec[key] for key in ("name", "source_name", "named", "logical_type")}
            for spec in specs
        ]
    config["parameters"] = {
        "mlflow_contract": {"string_value": json.dumps(mapping, separators=(",", ":"))}
    }
    config["instance_group"] = [{"count": 1, "kind": "KIND_" + device.upper()}]
    return config


def config_text(name, contract, device):
    config = model_config(name, contract, device)
    lines = [f"name: {json.dumps(name)}", 'backend: "python"', "max_batch_size: 0"]
    for direction in ("input", "output"):
        lines.append(direction + " [")
        for index, item in enumerate(config[direction]):
            lines.append(
                "  { name: "
                + json.dumps(item["name"])
                + " data_type: "
                + item["data_type"]
                + " dims: "
                + json.dumps(item["dims"])
                + " }"
                + ("," if index < len(config[direction]) - 1 else "")
            )
        lines.append("]")
    lines.append("instance_group [{ count: 1 kind: KIND_" + device.upper() + " }]")
    value = config["parameters"]["mlflow_contract"]["string_value"]
    lines.append(
        'parameters { key: "mlflow_contract" value { string_value: '
        + json.dumps(value)
        + " } }"
    )
    return "\n".join(lines) + "\n"
