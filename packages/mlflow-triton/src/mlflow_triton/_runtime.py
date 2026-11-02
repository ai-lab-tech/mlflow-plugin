"""Reusable Triton Python backend, copied into each logged repository.

Only MLflow's saved prediction interface is executed; training code is not run.
"""

import json
import os
import threading
from pathlib import Path

import numpy as np
import pandas as pd

# Serializes this module's temporary environment overrides. External callers that
# mutate os.environ or load models directly must coordinate separately.
_LOAD_LOCK = threading.RLock()


def contract_from_config(config):
    """Decode MLflow mappings; tensor types/shapes come only from Triton fields."""
    try:
        mapping = json.loads(config["parameters"]["mlflow_contract"]["string_value"])
        if mapping["version"] != 1:
            raise ValueError("Unsupported mlflow_contract version")
        if int(config.get("max_batch_size", 0)) != 0:
            raise ValueError("MLflow runtime requires max_batch_size=0")
        if mapping["flavor"] not in {"sklearn", "pytorch", "transformers"}:
            raise ValueError("Unsupported MLflow flavor")
        dtypes = {
            "BOOL": "bool",
            "UINT8": "uint8",
            "UINT16": "uint16",
            "UINT32": "uint32",
            "UINT64": "uint64",
            "INT8": "int8",
            "INT16": "int16",
            "INT32": "int32",
            "INT64": "int64",
            "FP16": "float16",
            "FP32": "float32",
            "FP64": "float64",
            "STRING": "object",
        }
        contract = {"flavor": mapping["flavor"]}
        for direction in ("input", "output"):
            mode = mapping[direction + "_mode"]
            if mode not in {"columns", "tensor", "named_tensors"}:
                raise ValueError("Unsupported MLflow mapping mode")
            entries = mapping[direction + "s"]
            tensors = config[direction]
            by_name = {tensor["name"]: tensor for tensor in tensors}
            names = [entry["name"] for entry in entries]
            if (
                not names
                or len(set(names)) != len(names)
                or len(by_name) != len(tensors)
                or set(names) != set(by_name)
            ):
                raise ValueError(
                    "MLflow mapping must match every declared " + direction
                )
            if mode == "tensor" and len(entries) != 1:
                raise ValueError("Single tensor mode requires exactly one tensor")
            specs = []
            for entry in entries:
                tensor = by_name[entry["name"]]
                if (
                    tensor.get("reshape") is not None
                    or tensor.get("is_shape_tensor")
                    or tensor.get("optional")
                ):
                    raise ValueError(
                        "Reshaped, shape or optional tensors are unsupported"
                    )
                dtype = tensor["data_type"].removeprefix("TYPE_")
                dims = [int(dim) for dim in tensor["dims"]]
                if not dims or any(dim == 0 or dim < -1 for dim in dims):
                    raise ValueError("Invalid tensor dimensions")
                if mode == "columns" and len(dims) != 1:
                    raise ValueError("Column mappings require rank-one tensors")
                if (
                    entry["logical_type"] not in {"binary", "string"}
                    or type(entry["named"]) is not bool
                ):
                    raise ValueError("Invalid MLflow field mapping")
                specs.append(
                    {
                        "name": entry["name"],
                        "source_name": entry["source_name"],
                        "named": entry["named"],
                        "logical_type": entry["logical_type"],
                        "datatype": "BYTES" if dtype == "STRING" else dtype,
                        "numpy_dtype": dtypes[dtype],
                        "dims": dims,
                    }
                )
            contract[direction + "s"] = specs
            contract[direction + "_mode"] = mode
        return contract
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        raise ValueError(
            "Invalid Triton MLflow configuration: " + str(error)
        ) from error


def checked_array(value, spec):
    array = np.asarray(value)
    if spec["datatype"] == "BYTES":
        if any(not isinstance(v, (str, bytes)) for v in array.ravel()):
            raise ValueError("Expected strings/bytes for " + spec["name"])
        if spec["logical_type"] != "binary":
            array = np.asarray(
                [
                    v.decode("utf-8") if isinstance(v, bytes) else v
                    for v in array.ravel()
                ],
                dtype=object,
            ).reshape(array.shape)
        else:
            array = array.astype(object)
    elif array.dtype != np.dtype(spec["numpy_dtype"]):
        raise ValueError(
            "Unexpected dtype for " + spec["name"] + ": " + str(array.dtype)
        )
    if array.ndim != len(spec["dims"]) or any(
        expected != -1 and expected != actual
        for expected, actual in zip(spec["dims"], array.shape)
    ):
        raise ValueError(
            "Unexpected shape for " + spec["name"] + ": " + str(array.shape)
        )
    return array


def decode_inputs(tensors, contract):
    specs = contract["inputs"]
    if set(tensors) != {s["name"] for s in specs}:
        raise ValueError("Request must contain every declared input exactly once")
    values = [checked_array(tensors[s["name"]], s) for s in specs]
    if contract["input_mode"] == "columns":
        return pd.DataFrame({s["source_name"]: v for s, v in zip(specs, values)})
    if contract["input_mode"] == "tensor":
        return values[0]
    return {s["source_name"]: v for s, v in zip(specs, values)}


def encode_outputs(result, contract):
    specs = contract["outputs"]
    if contract["output_mode"] == "columns":
        if isinstance(result, pd.DataFrame):
            if len(result.columns) != len(specs):
                raise ValueError("Prediction contains unexpected output columns")
            values = [
                result[s["source_name"]].to_numpy()
                if s["named"]
                else result.iloc[:, i].to_numpy()
                for i, s in enumerate(specs)
            ]
        elif isinstance(result, dict):
            if set(result) != {s["source_name"] for s in specs}:
                raise ValueError("Prediction contains unexpected output keys")
            values = [result[s["source_name"]] for s in specs]
        else:
            array = np.asarray(result)
            if array.ndim == 1 and len(specs) == 1:
                values = [array]
            elif array.ndim == 2 and array.shape[1] == len(specs):
                values = [array[:, i] for i in range(len(specs))]
            else:
                raise ValueError("Prediction does not match the column signature")
    elif contract["output_mode"] == "tensor":
        values = [result]
    else:
        if not isinstance(result, dict) or set(result) != {
            s["source_name"] for s in specs
        }:
            raise ValueError("Prediction does not match the named tensor signature")
        values = [result[s["source_name"]] for s in specs]
    return {s["name"]: checked_array(v, s) for s, v in zip(specs, values)}


def example_tensors(example, contract):
    """Convert the author's example using the same mapping as serving."""
    specs = contract["inputs"]
    if contract["input_mode"] == "columns":
        if isinstance(example, pd.DataFrame):
            values = [example[s["source_name"]].to_numpy() for s in specs]
        else:
            frame = pd.DataFrame(example)
            values = [frame[s["source_name"]].to_numpy() for s in specs]
    elif contract["input_mode"] == "tensor":
        values = [example]
    else:
        values = [example[s["source_name"]] for s in specs]
    return {s["name"]: checked_array(v, s) for s, v in zip(specs, values)}


def load_predictor(path, flavor, device="cpu"):
    import mlflow.pyfunc

    # MLflow's HF and PyTorch adapters interpret this setting differently.
    target = (
        ("-1" if device == "cpu" else device.split(":")[-1])
        if flavor == "transformers"
        else device
    )
    variables = {
        "MLFLOW_DEFAULT_PREDICTION_DEVICE": target,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MLFLOW_HUGGINGFACE_USE_DEVICE_MAP": "false",
    }
    with _LOAD_LOCK:
        previous = {key: os.environ.get(key) for key in variables}
        try:
            os.environ.update(variables)
            return mlflow.pyfunc.load_model(str(path))
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class TritonPythonModel:
    def initialize(self, args):
        import triton_python_backend_utils as pb_utils

        self.pb_utils = pb_utils
        version = Path(__file__).resolve().parent
        self.contract = contract_from_config(json.loads(args["model_config"]))
        device = (
            "cuda:" + str(args["model_instance_device_id"])
            if args["model_instance_kind"] == "GPU"
            else "cpu"
        )
        if self.contract["flavor"] == "sklearn" and device != "cpu":
            raise ValueError("The sklearn wrapper requires a CPU instance")
        self.model = load_predictor(
            version / "artifact", self.contract["flavor"], device
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                tensors = {}
                for spec in self.contract["inputs"]:
                    tensor = self.pb_utils.get_input_tensor_by_name(
                        request, spec["name"]
                    )
                    if tensor is None:
                        raise ValueError("Missing input: " + spec["name"])
                    tensors[spec["name"]] = tensor.as_numpy()
                inputs = decode_inputs(tensors, self.contract)
                outputs = encode_outputs(self.model.predict(inputs), self.contract)
                responses.append(
                    self.pb_utils.InferenceResponse(
                        output_tensors=[
                            self.pb_utils.Tensor(name, value)
                            for name, value in outputs.items()
                        ]
                    )
                )
            except Exception as error:  # noqa: BLE001 - Triton needs one error response per failed request.
                responses.append(
                    self.pb_utils.InferenceResponse(
                        error=self.pb_utils.TritonError(str(error))
                    )
                )
        return responses
