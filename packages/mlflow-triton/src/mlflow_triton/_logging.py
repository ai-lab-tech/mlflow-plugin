"""Native MLflow packaging followed by deterministic Triton Python serving."""

from __future__ import annotations

import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import mlflow
from mlflow.models import Model, infer_signature

from . import _runtime
from ._signature import config_text, contract_from_signature, validate_serving_options


def _validate_inputs(
    flavor, model, input_example, signature, triton_model_name, device, save_kwargs
):
    validate_serving_options(triton_model_name, flavor, device)
    if input_example is None:
        raise ValueError("input_example is required and must not be None")
    try:
        if len(input_example) == 0:
            raise ValueError("input_example must not be empty")
    except TypeError:
        raise ValueError("input_example must contain at least one request") from None
    if signature is False:
        raise ValueError("Signature inference cannot be disabled for Triton packaging")
    if flavor == "pytorch":
        import torch

        if not isinstance(model, torch.nn.Module):
            raise ValueError(
                "Pass an instantiated torch.nn.Module with trained weights, not a state_dict"
            )
        if any(module.training for module in model.modules()):
            raise ValueError("Set the PyTorch model to eval() before logging")
        # Do not silently export arbitrary Python/LLM computation into a graph.
        save_kwargs.setdefault("serialization_format", "pickle")
    if flavor == "transformers" and save_kwargs.get("save_pretrained") is False:
        raise ValueError(
            "Triton packages require local model/tokenizer assets; save_pretrained=False is unsupported"
        )


def _validate_saved_model(native, flavor, input_example, signature):
    metadata = Model.load(native / "MLmodel")
    if "python_function" not in metadata.flavors:
        raise ValueError(
            "This model has no supported MLflow Python prediction interface"
        )
    # Validate the saved artifact, not just the original in-memory object.
    predictor = _runtime.load_predictor(native, flavor)
    output = predictor.predict(input_example)
    selected = signature if signature is not None else metadata.signature
    if selected is None or (
        signature is None and (selected.inputs is None or selected.outputs is None)
    ):
        selected = infer_signature(input_example, output)
    contract = contract_from_signature(selected, flavor)
    tensors = _runtime.example_tensors(input_example, contract)
    restored = _runtime.decode_inputs(tensors, contract)
    # Round-trip the request without a second stochastic generation call.
    import numpy as np
    import pandas as pd

    if isinstance(restored, pd.DataFrame):
        original = (
            input_example
            if isinstance(input_example, pd.DataFrame)
            else pd.DataFrame(input_example)
        )
        pd.testing.assert_frame_equal(
            restored.reset_index(drop=True),
            original.reset_index(drop=True),
            check_dtype=False,
        )
    elif isinstance(restored, dict):
        if set(restored) != set(input_example):
            raise ValueError("Tensor mapping does not preserve all example inputs")
        for key in restored:
            np.testing.assert_array_equal(restored[key], input_example[key])
    else:
        np.testing.assert_array_equal(restored, input_example)
    _runtime.encode_outputs(output, contract)
    metadata.signature = selected
    metadata.save(native / "MLmodel")
    if any(p.is_symlink() for p in native.rglob("*")):
        raise ValueError("Saved model assets must not contain symbolic links")
    return metadata, contract


def _assemble_repository(
    native, staging, metadata, contract, triton_model_name, device, signature
):
    # Make only the independent serving copy. Move directories on the same disk
    # instead of recopying the complete package into its final location.
    artifact = staging / "artifact"
    shutil.copytree(native, artifact)
    package = staging / "package"
    native.rename(package)
    repository = package / "model_repository" / triton_model_name
    version = repository / "1"
    version.mkdir(parents=True)
    artifact.rename(version / "artifact")
    config = config_text(triton_model_name, contract, device)
    shutil.copyfile(Path(_runtime.__file__), version / "model.py")
    (repository / "config.pbtxt").write_text(config)
    requirements = (
        (package / "requirements.txt").read_text()
        if (package / "requirements.txt").exists()
        else ""
    )
    if not any(line.strip().startswith("mlflow") for line in requirements.splitlines()):
        requirements += f"\nmlflow=={mlflow.__version__}\n"
    (repository / "requirements.txt").write_text(requirements)
    (repository / "validation.json").write_text(
        json.dumps(
            {
                "local_validation": "saved_artifact_prediction_and_tensor_mapping_passed",
                "triton_validation": "not_run",
                "signature_origin": "supplied"
                if signature is not None
                else "mlflow_inferred",
                "shape_policy": "signature_dimensions; no extra dynamic axes guessed",
                "validation_device": "cpu",
            },
            indent=2,
        )
        + "\n"
    )
    metadata.add_flavor(
        "triton",
        data=f"model_repository/{triton_model_name}",
        model_name=triton_model_name,
    )
    metadata.save(package / "MLmodel")
    return package


def prepare_model(
    flavor,
    model,
    path,
    *,
    input_example,
    signature=None,
    triton_model_name="model",
    device="cpu",
    **save_kwargs,
):
    """Validate and atomically publish a native model plus serving repository."""
    _validate_inputs(
        flavor, model, input_example, signature, triton_model_name, device, save_kwargs
    )
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise ValueError("Destination already exists: " + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    native_flavor = importlib.import_module("mlflow." + flavor)
    # A sibling staging directory guarantees rename stays on the same filesystem.
    # TemporaryDirectory removes incomplete copies after any packaging failure.
    with tempfile.TemporaryDirectory(
        prefix=".mlflow-triton-prepare-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary)
        native = staging / "native"
        native_flavor.save_model(
            model,
            str(native),
            signature=signature,
            input_example=input_example,
            **save_kwargs,
        )
        metadata, contract = _validate_saved_model(
            native, flavor, input_example, signature
        )
        package = _assemble_repository(
            native, staging, metadata, contract, triton_model_name, device, signature
        )
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Destination already exists: " + str(destination))
        package.rename(destination)


def save_model(path, prepared_path, mlflow_model=None):
    """MLflow Model.log flavor hook; all checks precede logging/registration."""
    shutil.copytree(prepared_path, path, dirs_exist_ok=True)
    saved = Model.load(Path(path) / "MLmodel")
    target = mlflow_model or Model()
    target.flavors.update(saved.flavors)
    target.signature = saved.signature
    target.saved_input_example_info = saved.saved_input_example_info
    target.metadata = {**(saved.metadata or {}), **(target.metadata or {})}
    target.save(Path(path) / "MLmodel")


def log_native_model(
    flavor,
    model,
    *,
    input_example,
    name="model",
    signature=None,
    triton_model_name=None,
    device="cpu",
    registered_model_name=None,
    await_registration_for=300,
    run_id=None,
    metadata=None,
    tags=None,
    params=None,
    **save_kwargs,
):
    with tempfile.TemporaryDirectory(prefix="mlflow-triton-log-") as temporary:
        prepared = Path(temporary) / "prepared"
        prepare_model(
            flavor,
            model,
            prepared,
            input_example=input_example,
            signature=signature,
            triton_model_name=name if triton_model_name is None else triton_model_name,
            device=device,
            metadata=metadata,
            **save_kwargs,
        )
        return Model.log(
            artifact_path=None,
            flavor=sys.modules[__name__],
            name=name,
            registered_model_name=registered_model_name,
            await_registration_for=await_registration_for,
            run_id=run_id,
            metadata=metadata,
            tags=tags,
            params=params,
            prepared_path=str(prepared),
        )
