# mlflow-triton

Log a trained scikit-learn, PyTorch or Hugging Face Transformers model to MLflow
with the files needed to serve it using NVIDIA Triton.

> Intended for demos and proofs of concept. Before production use, an ML engineer
> must review, validate and optimize the generated serving implementation.

## Installation

```bash
pip install mlflow-triton
```

Requires Python 3.10+ and MLflow >=3.14,<4. Install your model's framework and
other dependencies separately. Configure your MLflow tracking URI and experiment
before logging if you use a remote server.

## Log your trained model

Pass an already-trained model and a representative prediction input. These
examples assume the model and input variables already exist.
For a complete runnable example, see [quickstart.py](https://github.com/ai-lab-tech/mlflow-plugin/blob/main/examples/quickstart.py).
You can also [run the example in Google Colab](https://colab.research.google.com/github/ai-lab-tech/mlflow-plugin/blob/main/examples/quickstart_colab.ipynb),
including installation and an optional download of the Triton serving files.

### scikit-learn

```python
from mlflow_triton import sklearn

info = sklearn.log_model(
    fitted_pipeline,
    name="classifier",
    input_example=example_dataframe,
    registered_model_name="classifier",  # Optional: create a Registry version
)
print(info.model_uri)
```

Include preprocessing in the fitted estimator or `Pipeline`.

### PyTorch

```python
from mlflow_triton import pytorch

info = pytorch.log_model(
    loaded_model.eval(),
    name="network",
    input_example=example_numpy_array,
)
```

Pass an instantiated `torch.nn.Module` with trained weights loaded and all
submodules in evaluation mode. For GPU serving, add `device="gpu"`.
If your model needs custom architecture code or dependencies, supply them with
`code_paths=["inference_architecture"]` and
`pip_requirements="requirements-serving.txt"`, replacing these paths with your own.

### Hugging Face Transformers

```python
from mlflow_triton import transformers

info = transformers.log_model(
    loaded_generation_pipeline,
    name="text_generator",
    input_example=["Explain what a model signature describes."],
    task="text-generation",
    model_config={"max_new_tokens": 32, "do_sample": False},
)
```

Pass a supported pipeline or a component dictionary such as
`{"model": model, "tokenizer": tokenizer}`, including the task's processor assets.
Use `model_config` for prediction settings.

All three calls return MLflow `ModelInfo`. The model and generated Triton files
are stored at `info.model_uri`; no additional save or load call is needed.
`info.registered_model_version` identifies the Registry version when requested.
Logging does not start Triton or deploy a service.

## Log API reference

Pass the model positionally or by the parameter name below. All other arguments
must be keyword arguments.

### Required arguments

| Module | Model parameter | `input_example` |
| --- | --- | --- |
| `sklearn` | `sk_model`: fitted estimator or Pipeline | DataFrame or NumPy array accepted by the model. Preserve column names, order and dtypes. |
| `pytorch` | `pytorch_model`: trained module in evaluation mode | NumPy array or named tensor input accepted by MLflow's PyTorch prediction interface. Match input dtype and dimensions. |
| `transformers` | `transformers_model`: pipeline or component dictionary | Task input accepted by MLflow's prediction interface, such as a list of strings or a DataFrame. |

`input_example` must be non-empty and is required even with an explicit signature.
It is prediction input, not a Triton HTTP request. Its values are saved in the
artifact and generated request files, so use non-sensitive examples.

### Common optional arguments

| Argument | Default | Purpose |
| --- | --- | --- |
| `name` | `"model"` | MLflow model name; also the Triton model name unless overridden. |
| `triton_model_name` | `None` | Override the Triton name. Start with an ASCII letter or `_`; remaining characters may also include digits and `-`. |
| `device` | `"cpu"` | Triton serving target: `"cpu"` or `"gpu"`. sklearn supports CPU only. Does not move the supplied model or select a GPU index. |
| `signature` | `None` | MLflow `ModelSignature` with input and output schemas. Inferred from the example and prediction when omitted; `False` is unsupported. |
| `validate_saved_model` | `False` | When `True`, reload the saved model on CPU for validation. See [validation](#validation-and-limitations). |
| `registered_model_name` | `None` | Create a version in the MLflow Model Registry. |
| `await_registration_for` | `300` | Registration wait timeout in seconds. |
| `run_id` | `None` | Associate the model with an MLflow run; otherwise MLflow applies its default run behavior. |
| `metadata` | `None` | Model metadata dictionary. |
| `tags` | `None` | Logged-model tags dictionary. |
| `params` | `None` | Tracking parameters dictionary. For Transformers generation settings, use `model_config`. |

The Triton directory always uses version `1`, independently of the MLflow
Registry version. Use `name` for the logged model; `artifact_path` is unsupported.

### Model saving options

Pass these options to the same `log_model()` call to include code, specify
dependencies or configure serialization.

| Argument | Framework | Default / purpose |
| --- | --- | --- |
| `code_paths` | All | `None`. Local inference-code files or directories to bundle. Preserve their importable package structure. |
| `pip_requirements` | All | `None`. Requirements list or file path; otherwise uses MLflow's default framework dependencies. |
| `extra_pip_requirements` | All | `None`. Add dependencies to the default or inferred requirements. |
| `conda_env` | All | `None`. Environment dictionary or file path. Follow MLflow's restrictions on combining environment arguments. |
| `pyfunc_predict_fn` | sklearn | `"predict"`. Prediction method, for example `"predict_proba"`. Also determines the output used to infer the signature. |
| `serialization_format` | sklearn | Uses the installed MLflow version's default format. |
| `serialization_format` | PyTorch | `"pickle"`. Alternatives require MLflow support; use `validate_saved_model=True` to check reloading. |
| `pickle_module` | PyTorch | `None`. Python module used for serialization. |
| `task` | Transformers | `None`. Task such as `"text-generation"`; provide it if MLflow cannot infer the task. |
| `processor` | Transformers | `None`. Task-specific processor, when required. |
| `model_config` | Transformers | `None`. Prediction settings dictionary, such as `{"max_new_tokens": 32}`. |
| `save_pretrained` | Transformers | `True`. Must remain enabled to include local model and tokenizer assets. |
| `architecture_init_kwargs` | PyTorch, Transformers | `None`. Constructor settings recorded as documentation; not used to reconstruct the model. See the [architecture report](ADVANCED.md#inspect-model-architecture-without-loading-weights). |

Include custom dependencies explicitly: default logging skips automatic dependency
detection to avoid reloading weights. With `validate_saved_model=True`, MLflow's
dependency detection is enabled when no explicit environment is supplied.
`code_paths` bundles local code; requirement arguments record installable dependencies.

Additional saving options and version-specific defaults follow the installed
MLflow version's `mlflow.<framework>.save_model` API, which the library uses
internally to write model files. Options exclusive to MLflow's `log_model` are
not automatically supported.

## Validation and limitations

By default, logging runs an example prediction with your model on its current
device, infers or checks its signature, and validates the Triton input/output
mapping. Invalid inputs, unsupported schemas and failed predictions raise errors.

Set `validate_saved_model=True` to instead reload the saved artifact on CPU and
validate it. This requires memory for an additional model copy. The result is
recorded in `validation.json`. Test actual Triton serving and GPU execution separately.

- Serving uses Triton's Python backend and MLflow's prediction interface. PyTorch
  models are not automatically exported to a graph backend; streaming is unsupported.
- Preprocessing and postprocessing must be included in the supplied prediction
  interface. Custom architecture code must be bundled or installed.
- Supported schemas contain NumPy-compatible numeric/string tensors or required
  scalar DataFrame columns. Nested/object schemas, optional or datetime columns,
  and scalar tensors need a separate adapter.
- Generated configuration uses `max_batch_size: 0`. Supply an explicit signature
  for dynamic dimensions that cannot be inferred from the example.
- Sharded/device-mapped models and tasks unsupported by MLflow's prediction
  interface need a separate adapter.

## Retrieve Triton files (optional)

Use this only when you need a local directory for inspection or deployment:

```python
from mlflow_triton import triton

model_directory = triton.load_model(info.model_uri)
print(model_directory / "config.pbtxt")
```

Despite its name, `triton.load_model()` returns a `pathlib.Path`, downloading files
as needed. It does not load model weights or load a model into a running server.
The returned directory contains one Triton model:

```text
<triton_model_name>/
  config.pbtxt
  requirements.txt
  request.json        # Or request.bin + request.headers for binary inputs
  validation.json
  1/
    model.py
    artifact/         # Self-contained MLflow model, code and assets
```

Install the recorded dependencies in a serving environment compatible with
Triton's Python backend. Logging stores two model copies so the Triton directory
can be deployed independently; allow enough disk space for both and temporary files.
MLflow's framework and `pyfunc` loaders remain usable for the logged model.

## Log an existing Triton model directory

If you already have Triton serving files, store them in MLflow directly:

```python
from mlflow_triton import triton

info = triton.log_model(
    triton_model_path="/tmp/model-repository/iris_classifier",
    name="triton-model",
    registered_model_name="iris-serving",
)
```

| Argument | Default | Purpose |
| --- | --- | --- |
| `triton_model_path` | Required | One model directory with at least one positive numbered version directory. Symbolic links are rejected. |
| `name` | `"triton-model"` | MLflow logged-model name; preserves the existing Triton directory name and configuration. |
| `registered_model_name` | `None` | Optional Registry name. |
| `await_registration_for` | `300` | Registration wait timeout in seconds. |
| `run_id` | `None` | Optional MLflow run association. |
| `metadata` | `None` | Model metadata dictionary. |

Returns `ModelInfo`. This API preserves your existing files; it does not generate
serving code, signatures or requests. It accepts only the arguments above.
Retrieve the files with `triton.load_model(info.model_uri)` when needed.

## Further documentation

- [Advanced serving guide](ADVANCED.md): validation details, device behavior,
  configuration mapping, architecture reports and resource requirements.
- [Examples and validation](https://github.com/ai-lab-tech/mlflow-plugin/blob/main/examples/README.md).
- [Send a generated request](https://github.com/ai-lab-tech/mlflow-plugin#send-an-example-request) after deploying your model to Triton.
- [Optional Triton Control deployment](https://github.com/ai-lab-tech/mlflow-plugin/blob/main/packages/mlflow-triton-control/README.md).
