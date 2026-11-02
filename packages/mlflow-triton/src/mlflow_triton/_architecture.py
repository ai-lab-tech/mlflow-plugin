"""Human-readable architecture evidence; never a model reconstruction recipe."""

from __future__ import annotations

import json


def class_name(value):
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def validate_init_kwargs(value):
    if value is not None:
        if not isinstance(value, dict):
            raise ValueError("architecture_init_kwargs must be a JSON object")
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "architecture_init_kwargs must contain JSON values"
            ) from error


def write_architecture(native, flavor, model, init_kwargs):
    if flavor not in {"pytorch", "transformers"}:
        return
    components = model if isinstance(model, dict) else {}
    network = (
        (components.get("model") if components else getattr(model, "model", model))
        if flavor == "transformers"
        else model
    )
    files = sorted(
        p.relative_to(native).as_posix() for p in native.rglob("*") if p.is_file()
    )
    report = {
        "schema_version": 1,
        "flavor": flavor,
        "model_class": class_name(network),
        "init_kwargs": init_kwargs,
        "init_kwargs_origin": "user_supplied" if init_kwargs is not None else "unknown",
        "code_files": [name for name in files if name.endswith(".py")],
        "config_files": [name for name in files if name.endswith("config.json")],
        "limitations": [
            "Descriptive evidence only; not an executable reconstruction recipe.",
            "Module structure does not describe forward control flow or preprocessing.",
            "Constructor arguments are not inferred. Custom code and dependencies may be required.",
        ],
    }
    if flavor == "transformers":
        report["components"] = {
            name: class_name(value)
            for name in (
                "tokenizer",
                "feature_extractor",
                "image_processor",
                "processor",
            )
            if (
                value := components.get(name)
                if components
                else getattr(model, name, None)
            )
            is not None
        }
    if hasattr(network, "named_modules"):
        report["modules"] = [
            {"name": name, "class": class_name(module)}
            for name, module in network.named_modules()
        ]
        report["tensors"] = []
        for kind, tensors in (
            ("parameter", network.named_parameters()),
            ("buffer", network.named_buffers()),
        ):
            for name, tensor in tensors:
                try:
                    shape = list(tensor.shape)
                except RuntimeError:
                    shape = None
                report["tensors"].append(
                    {
                        "name": name,
                        "kind": kind,
                        "shape": shape,
                        "dtype": str(tensor.dtype),
                    }
                )
    (native / "architecture.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
