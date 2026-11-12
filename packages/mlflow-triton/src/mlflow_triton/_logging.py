"""Native MLflow packaging followed by deterministic Triton Python serving."""

from __future__ import annotations

import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import mlflow
from mlflow.entities import LoggedModelOutput, LoggedModelStatus, Metric
from mlflow.models import Model, infer_signature

from . import _runtime
from ._architecture import validate_init_kwargs, write_architecture
from ._dependencies import PredictionDependencies
from ._request import write_request_example
from ._requirements import serving_requirements
from ._signature import config_text, contract_from_signature, validate_serving_options
from ._versions import append_shared_version, extend_registered_repository


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


def _in_memory_predictor(native_flavor, flavor, model, signature, save_kwargs):
    """Use MLflow's serving adapters without loading or copying model weights."""
    predict_fn = "predict"
    validation_device = "cpu"
    if flavor == "sklearn":
        implementation = native_flavor._SklearnModelWrapper(model)
        predict_fn = save_kwargs.get("pyfunc_predict_fn", "predict")
    elif flavor == "pytorch":
        tensor = next(model.parameters(), None)
        if tensor is None:
            tensor = next(model.buffers(), None)
        validation_device = str(tensor.device) if tensor is not None else "cpu"
        implementation = native_flavor._PyTorchWrapper(model, validation_device)
    else:
        if isinstance(model, dict):
            native_flavor._validate_transformers_model_dict(model)
            components = dict(model)
            # Pipeline construction otherwise chooses the first available GPU.
            # Keep the supplied weights on their existing device.
            if getattr(model["model"], "hf_device_map", None) is None:
                components["device"] = getattr(model["model"], "device", -1)
            model = native_flavor._build_pipeline_from_model_input(
                components, task=save_kwargs.get("task")
            )
        model_config = dict(save_kwargs.get("model_config") or {})
        prompt_template = save_kwargs.get("prompt_template")
        if prompt_template is not None and model.task == "text-generation":
            model_config.setdefault("return_full_text", False)
        implementation = native_flavor._TransformersWrapper(
            model,
            model_config=model_config,
            prompt_template=prompt_template,
        )
        validation_device = str(model.device)
    predictor = mlflow.pyfunc.PyFuncModel(
        Model(signature=signature), implementation, predict_fn=predict_fn
    )
    return model, predictor, validation_device


def _set_default_requirements(native_flavor, flavor, model, save_kwargs, observed=()):
    """Avoid native dependency inference, which reloads models in a subprocess."""
    if (
        save_kwargs.get("pip_requirements") is not None
        or save_kwargs.get("conda_env") is not None
    ):
        return
    if flavor == "transformers":
        requirements = native_flavor.get_default_pip_requirements(model.model)
    elif flavor == "sklearn":
        serialization = save_kwargs.get(
            "serialization_format", native_flavor.SERIALIZATION_FORMAT_CLOUDPICKLE
        )
        options = {"include_cloudpickle": serialization == "cloudpickle"}
        if serialization == "skops":
            options["include_skops"] = True
        requirements = native_flavor.get_default_pip_requirements(**options)
    else:
        requirements = native_flavor.get_default_pip_requirements()
    requirements = list(dict.fromkeys([*requirements, *observed]))
    extra = save_kwargs.pop("extra_pip_requirements", None)
    if extra is not None:
        requirements = [
            *requirements,
            *(["-r " + extra] if isinstance(extra, str) else extra),
        ]
    save_kwargs["pip_requirements"] = requirements


def _validate_prediction(native, flavor, input_example, signature, output):
    metadata = Model.load(native / "MLmodel")
    if "python_function" not in metadata.flavors:
        raise ValueError(
            "This model has no supported MLflow Python prediction interface"
        )
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
    native,
    staging,
    metadata,
    contract,
    triton_model_name,
    device,
    signature,
    input_example,
    validate_saved_model,
    validation_device,
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
    write_request_example(repository, input_example, contract)
    (version / "requirements.txt").write_text(
        serving_requirements(package, mlflow.__version__), encoding="utf-8"
    )
    (repository / "validation.json").write_text(
        json.dumps(
            {
                "local_validation": (
                    "saved_artifact_prediction_and_tensor_mapping_passed"
                    if validate_saved_model
                    else "in_memory_prediction_and_tensor_mapping_passed"
                ),
                "saved_artifact_validation": "passed"
                if validate_saved_model
                else "not_run",
                "triton_validation": "not_run",
                "signature_origin": "supplied"
                if signature is not None
                else "mlflow_inferred",
                "shape_policy": "signature_dimensions; no extra dynamic axes guessed",
                "validation_device": validation_device,
            },
            indent=2,
        )
        + "\n"
    )
    metadata.add_flavor(
        "triton",
        data=f"model_repository/{triton_model_name}",
        model_name=triton_model_name,
        model_version=1,
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
    architecture_init_kwargs=None,
    triton_model_name="model",
    device="cpu",
    validate_saved_model=False,
    **save_kwargs,
):
    """Validate and atomically publish a native model plus serving repository."""
    _validate_inputs(
        flavor, model, input_example, signature, triton_model_name, device, save_kwargs
    )
    if not isinstance(validate_saved_model, bool):
        raise ValueError("validate_saved_model must be a bool")
    validate_init_kwargs(architecture_init_kwargs)
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
        selected = signature
        validation_device = "cpu"
        if not validate_saved_model:
            model, predictor, validation_device = _in_memory_predictor(
                native_flavor, flavor, model, signature, save_kwargs
            )
            dependencies = PredictionDependencies()
            if (
                save_kwargs.get("pip_requirements") is None
                and save_kwargs.get("conda_env") is None
            ):
                with dependencies.capture():
                    output = predictor.predict(input_example)
                observed = dependencies.requirements()
            else:
                output = predictor.predict(input_example)
                observed = ()
            if selected is None:
                selected = infer_signature(input_example, output)
            _set_default_requirements(
                native_flavor, flavor, model, save_kwargs, observed
            )
        native_flavor.save_model(
            model,
            str(native),
            signature=selected,
            input_example=input_example,
            **save_kwargs,
        )
        if validate_saved_model:
            predictor = _runtime.load_predictor(native, flavor)
            output = predictor.predict(input_example)
        metadata, contract = _validate_prediction(
            native, flavor, input_example, selected, output
        )
        write_architecture(native, flavor, model, architecture_init_kwargs)
        package = _assemble_repository(
            native,
            staging,
            metadata,
            contract,
            triton_model_name,
            device,
            signature,
            input_example,
            validate_saved_model,
            validation_device,
        )
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Destination already exists: " + str(destination))
        package.rename(destination)


def save_model(path, prepared_path, mlflow_model=None, validate_serving_input=False):
    """MLflow hook; validate_serving_input is consumed by Model.log itself."""
    shutil.copytree(prepared_path, path, dirs_exist_ok=True)
    saved = Model.load(Path(path) / "MLmodel")
    target = mlflow_model or Model()
    target.flavors.update(saved.flavors)
    target.signature = saved.signature
    target.saved_input_example_info = saved.saved_input_example_info
    target.metadata = {**(saved.metadata or {}), **(target.metadata or {})}
    target.save(Path(path) / "MLmodel")


def _log_update(prepared, update, run_id):
    """Merge new artifacts without duplicating an existing run/model association."""
    client = mlflow.MlflowClient()
    logged = client.get_logged_model(update.model_id)
    if run_id is None and (active := mlflow.active_run()) is not None:
        run_id = active.info.run_id
    run = client.get_run(run_id) if run_id else None
    metadata = Model.load(prepared / "MLmodel")
    metadata.model_id = logged.model_id
    metadata.model_uuid = logged.model_id
    metadata.artifact_path = logged.artifact_location
    metadata.run_id = run_id
    metadata.save(prepared / "MLmodel")

    # log_model_artifacts merges files: older Triton directories are not touched.
    # Model.log(model_id=...) cannot be used here because MLflow 3.14 attempts
    # to insert the same run/model output twice when updating within one run.
    try:
        client.log_model_artifacts(logged.model_id, str(prepared))
    except Exception:
        client.finalize_logged_model(logged.model_id, LoggedModelStatus.FAILED)
        raise
    client.finalize_logged_model(logged.model_id, LoggedModelStatus.READY)
    if run is not None:
        outputs = run.outputs.model_outputs if run.outputs else []
        if not any(output.model_id == logged.model_id for output in outputs):
            client.log_outputs(run_id, [LoggedModelOutput(logged.model_id, step=0)])
        # Match normal Model.log's association of step-zero run metrics.
        metrics = [
            Metric(
                key=metric.key,
                value=metric.value,
                timestamp=metric.timestamp,
                step=metric.step,
                dataset_name=metric.dataset_name,
                dataset_digest=metric.dataset_digest,
                run_id=run_id,
                model_id=logged.model_id,
            )
            for name in run.data.metrics
            for metric in client.get_metric_history(run_id, name)
            if metric.step == 0 and metric.model_id is None
        ]
        if metrics:
            client.log_batch(run_id=run_id, metrics=metrics)
    info = metadata.get_model_info(client.get_logged_model(logged.model_id))
    info.registered_model_version = update.registered_model_version
    return info


def log_native_model(
    flavor,
    model,
    *,
    input_example,
    name="model",
    signature=None,
    triton_model_name=None,
    device="cpu",
    validate_saved_model=False,
    registered_model_name=None,
    model_uri=None,
    await_registration_for=300,
    run_id=None,
    metadata=None,
    tags=None,
    params=None,
    **save_kwargs,
):
    if registered_model_name is not None and model_uri is not None:
        raise ValueError("registered_model_name and model_uri are mutually exclusive")
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
            validate_saved_model=validate_saved_model,
            metadata=metadata,
            **save_kwargs,
        )
        update = None
        if registered_model_name is not None or model_uri is not None:
            update = extend_registered_repository(
                prepared, registered_model_name, model_uri=model_uri
            )
        append_shared_version(prepared, update)
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
            # Model.log otherwise reloads the model to validate its serving input.
            # prepare_model has already validated the requested prediction mode.
            validate_serving_input=False,
        )
