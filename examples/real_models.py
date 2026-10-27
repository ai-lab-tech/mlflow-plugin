"""Saved-model examples. No training, downloads or live Triton calls."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path

from data_catalog import CASES, DEFAULT_DATA_ROOT, select_cases


def load_example(name, data_root):
    """Load existing trained state and supply a new, independent example request."""
    import numpy as np

    flavor, relative = CASES[name]
    path = Path(data_root) / relative
    if not path.is_dir():
        raise FileNotFoundError(f"Missing saved artifact directory: {path}")
    kwargs = {}
    if flavor == "sklearn":
        import joblib
        import pandas as pd
        import sklearn
        from sklearn.pipeline import Pipeline

        requirements = (path / "requirements.txt").read_text().splitlines()
        saved_version = next(
            (
                line.split("==", 1)[1]
                for line in requirements
                if line.startswith("scikit-learn==")
            ),
            None,
        )
        if saved_version and sklearn.__version__ != saved_version:
            raise RuntimeError(
                f"Saved sklearn model requires scikit-learn=={saved_version}; "
                f"current version is {sklearn.__version__}. Use the artifact's environment."
            )
        # Preserve trusted, already loaded Python pipelines without skops conversion.
        kwargs["serialization_format"] = "cloudpickle"
        model = joblib.load(path / "model.joblib")
        if name == "sklearn_mixed_columns":
            example = pd.DataFrame(
                {
                    "pclass": [3, 1],
                    "sex": ["male", "female"],
                    "age": [28.0, 42.0],
                    "fare": [12.5, 80.0],
                    "embarked": ["S", "C"],
                }
            )
            example = example[list(model.feature_names_in_)]
        elif name == "sklearn_gradient_boosting":
            example = np.linspace(-1.0, 1.0, 20).reshape(2, 10)
        else:
            scaler_file = (
                "StandardScaler.joblib"
                if name == "sklearn_scaled_svm"
                else "MinMaxScaler.joblib"
            )
            # Compose already fitted components; never call fit().
            model = Pipeline(
                [
                    ("preprocessing", joblib.load(path / scaler_file)),
                    ("estimator", model),
                ]
            )
            example = np.array([[5.1, 3.5], [6.2, 2.8]], dtype=np.float64)
    elif flavor == "pytorch":
        import cloudpickle
        import torch

        # This saved object contains the original architecture, including custom
        # classes. It is not a module imported from the training script.
        with (path / "reference-model.pkl").open("rb") as stream:
            model = cloudpickle.load(stream)
        filename = "netG_epoch_0.pth" if name == "pytorch_dcgan" else "weights.pt"
        state = torch.load(
            path / "artifact" / filename, map_location="cpu", weights_only=True
        )
        model.load_state_dict(state, strict=True)
        model.cpu().eval()
        if name == "pytorch_polynomial":
            example = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
        elif name == "pytorch_custom_network":
            example = np.linspace(
                -0.5, 0.5, 2 * model.linear1.in_features, dtype=np.float32
            )
            example = example.reshape(2, model.linear1.in_features)
        elif name == "pytorch_mnist_lstm":
            example = np.zeros((2, 1, 28, 28), dtype=np.float32)
            example[:, :, 5:23, 13:15] = 1.0
        else:
            example = (
                np.random.default_rng(42).normal(size=(2, 100, 1, 1)).astype(np.float32)
            )
        # MLflow defaults to its torch-compatible CloudPickle adapter, which
        # supplies both Pickler and Unpickler. Raw cloudpickle does not.
    else:
        import pandas as pd
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForQuestionAnswering,
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )

        task, model_class = {
            "huggingface_custom": (
                "text-classification",
                AutoModelForSequenceClassification,
            ),
            "huggingface_classification": (
                "text-classification",
                AutoModelForSequenceClassification,
            ),
            "huggingface_ner": (
                "token-classification",
                AutoModelForTokenClassification,
            ),
            "huggingface_question_answering": (
                "question-answering",
                AutoModelForQuestionAnswering,
            ),
            "huggingface_generation": ("text-generation", AutoModelForCausalLM),
        }[name]
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        network = (
            model_class.from_pretrained(
                path,
                local_files_only=True,
                trust_remote_code=(name == "huggingface_custom"),
            )
            .cpu()
            .eval()
        )
        model = pipeline(task, model=network, tokenizer=tokenizer, device=-1)
        kwargs["task"] = task
        if task == "question-answering":
            example = pd.DataFrame(
                {
                    "question": ["Where does Alice live?"],
                    "context": ["Alice lives in Berlin and works as an engineer."],
                }
            )
        else:
            example = pd.DataFrame({"text": ["Alice lives in Berlin."]})
        if task == "text-generation":
            kwargs["model_config"] = {"max_new_tokens": 8, "do_sample": False}
    return flavor, model, example, kwargs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Saved artifacts; defaults to MLFLOW_TRITON_DATA_ROOT or public examples/data",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Serving device; local reference predictions use CPU",
    )
    args = parser.parse_args()
    if args.list:
        for name, (flavor, relative) in CASES.items():
            print(f"{name}: {flavor} — {relative}")
        return 0
    if args.output is None:
        parser.error("--output is required unless --list is used")
    try:
        selected = select_cases(args.data_root, args.case, device=args.device)
    except ValueError as error:
        parser.error(str(error))
    # Make running examples explicit and independent of configured remote tracking.
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import mlflow

    mlflow.set_tracking_uri("sqlite:///" + str(output / "mlflow.db"))
    experiment = mlflow.create_experiment(
        "saved-model-examples", artifact_location=(output / "artifacts").as_uri()
    )
    results = []
    for name in selected:
        result = {
            "case": name,
            "device": args.device,
            "local_validation": "failed",
            "triton_validation": "not_run",
        }
        try:
            flavor, model, example, kwargs = load_example(name, args.data_root)
            wrapper = importlib.import_module("mlflow_triton." + flavor)
            with mlflow.start_run(experiment_id=experiment, run_name=name):
                info = wrapper.log_model(
                    model,
                    name=name,
                    input_example=example,
                    device=args.device,
                    **kwargs,
                )
            import shutil

            import numpy as np
            from mlflow_triton import _runtime, triton

            repository = triton.load_model(info.model_uri)
            serving = output / "serving" / name
            serving.parent.mkdir(exist_ok=True)
            shutil.copytree(repository, serving)
            from serving_validation import read_model_contract

            contract = read_model_contract(serving)
            # Independent prediction through the saved native MLflow artifact.
            native = _runtime.load_predictor(serving / "1/artifact", flavor)
            expected = _runtime.encode_outputs(native.predict(example), contract)
            inputs = _runtime.example_tensors(example, contract)
            fixture = {}
            for prefix, values in [("input", inputs), ("output", expected)]:
                for key, value in values.items():
                    array = np.asarray(value)
                    if array.dtype == object:
                        array = np.asarray(
                            [
                                item.decode("utf-8")
                                if isinstance(item, bytes)
                                else item
                                for item in array.ravel()
                            ],
                            dtype=str,
                        ).reshape(array.shape)
                    fixture[prefix + ":" + key] = array
            fixture_path = output / (name + "-fixture.npz")
            np.savez(fixture_path, **fixture)
            result.update(
                local_validation="passed",
                model_uri=info.model_uri,
                serving_path=str(serving),
                fixture_path=str(fixture_path),
            )
        except Exception as error:  # noqa: BLE001 — retain every case's failure
            result["error"] = f"{type(error).__name__}: {error}"
        results.append(result)
        (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    return int(any(result["local_validation"] != "passed" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
