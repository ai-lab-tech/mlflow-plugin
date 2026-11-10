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
including installation and a listing of the generated Triton artifact paths.

### scikit-learn

```python
from mlflow_triton import sklearn

info = sklearn.log_model(
    fitted_pipeline,
    name="classifier",
    input_example=example_dataframe,
    registered_model_name="classifier",  # Optional: select/register this model
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

## Register a model after training

Use the plugin's `register_model()` to apply the same Triton version rules as
`log_model(..., registered_model_name=...)` after logging and selecting a model:

```python
from mlflow_triton import register_model, sklearn

info = sklearn.log_model(
    fitted_pipeline,
    name="classifier",
    input_example=X_sample,
)
version = register_model(info.model_uri, "classifier")
print(version.version)  # MLflow Registry version, not the Triton directory number
```

- First registration: register the logged model with Triton directory `1/`.
- Matching `config.pbtxt`: append `2/`, `3/`, ... to the existing registered
  model, reusing its MLflow model ID and Registry version. Its root native model
  is updated to the selected weights; earlier Triton directories stay intact.
- Changed configuration: create a new Registry version starting at `1/`.

The return value is MLflow `ModelVersion`. Arguments are `model_uri`, `name`,
`await_registration_for=300`, and optional keyword-only `tags`. Models without
a Triton flavor use standard MLflow registration. Calls for one registered name
must run sequentially. Source model references support logged-model IDs,
Registry versions/aliases, and run URIs. Registration copies the selected source
artifacts without loading or retraining the model. Earlier source Triton
versions and predecessor weights are not downloaded or copied.

**Use `mlflow_triton.register_model`, not `mlflow.register_model`, for this behavior.**
MLflow's own function and **Register Model in the UI** create Registry versions
without running this plugin's Triton config comparison or incrementing Triton
numbers. This package does not install a server-side registration hook.

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
| `registered_model_name` | `None` | Select the latest Registry version; register a new version only for a new/incompatible model. |
| `model_uri` | `None` | Explicit predecessor (`models:/<model_id>`, Registry version/alias, or `runs:/...`); cannot be combined with `registered_model_name`. |
| `await_registration_for` | `300` | Registration wait timeout in seconds. |
| `run_id` | `None` | Associate the model with an MLflow run; otherwise MLflow applies its default run behavior. |
| `metadata` | `None` | Model metadata dictionary. |
| `tags` | `None` | Logged-model tags dictionary. |
| `params` | `None` | Tracking parameters dictionary. For Transformers generation settings, use `model_config`. |

Use `name` for the logged model; `artifact_path` is unsupported.

### Automatic Triton versions

When `model_uri` is supplied, logging compares the generated
`config.pbtxt` with that model. Otherwise, `registered_model_name` selects the
latest version of that registered model:

- Equal configuration: reuse the **same MLflow model ID**, append the next Triton
  directory (`1/`, `2/`, `3/`, ...), and retain earlier directories. When selected
  by `registered_model_name`, also reuse the existing Registry version.
- Changed configuration: create a **new MLflow model ID** with Triton directory
  `1/`. With `registered_model_name`, register it as a new version of that name.
- No predecessor, or no Triton flavor: create a new model starting at `1/`.

Only metadata and the previous `config.pbtxt` are downloaded. Earlier Triton
weights, backend code, and requirements remain in storage without being copied.
The native MLflow package at the artifact root is updated to the latest weights,
so loading the same model URI (including its Registry URI) returns the new model.
Without either history parameter, logging always creates a new model at `1/`.

```python
info = mlflow_triton.sklearn.log_model(
    updated_model,
    input_example=X_sample,
    name="model",
    model_uri="models:/<model_id>",
)
```

An explicit predecessor does not require registration. `model_uri` and
`registered_model_name` are mutually exclusive: supplying both raises `ValueError`
before model preparation or logging. Each Triton version carries its own serving
requirements at `<version>/requirements.txt`. Calls updating the same model must
run sequentially. A matching predecessor must have a logged-model ID (MLflow 3).

See [model version updates](ADVANCED.md#model-version-updates) for the activity
diagram, configuration comparison, stored metadata, and concurrency details.

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

## Generated Triton files

`log_model()` stores the generated Triton directory in the MLflow artifact store
alongside the native model. Inspect it through the MLflow artifact browser.

List the generated files in the artifact store without downloading them:

```python
from mlflow import MlflowClient

triton_path = info.flavors["triton"]["data"]
for artifact in MlflowClient().list_logged_model_artifacts(
    info.model_id, path=triton_path
):
    print(artifact.path, "directory" if artifact.is_dir else "file")
```

This lists the direct entries. The quickstart examples also walk subdirectories
to show the complete generated file list. Paths are relative to the logged model;
they are not local filesystem paths or direct S3 addresses.

Generated directory layout:

```text
<triton_model_name>/
  config.pbtxt
  request.json        # Or request.bin + request.headers for binary inputs
  validation.json
  1/
    requirements.txt
    model.py
    artifact/         # Self-contained MLflow model, code and assets
```

The serving `requirements.txt` is standalone: requirements and constraints from
MLflow's saved environment are combined into package requirements, without `-r`
or `-c` file references. Constraint-only packages become explicit serving
dependencies and will also be installed. Version restrictions and environment
markers are retained. Unsupported pip options or local file dependencies fail
packaging rather than producing an unusable deployment file. MLflow's native
environment files inside the saved artifact remain unchanged.
If MLflow is specified only conditionally, the serving file also requires
`mlflow>=3.14,<4` unconditionally. Applicable conditional version restrictions
still apply, so MLflow remains available on other serving Python versions too.

Install the recorded dependencies in a serving environment compatible with
Triton's Python backend. Logging stores two model copies so the Triton directory
can be deployed independently; allow enough disk space for both and temporary files.
MLflow's framework and `pyfunc` loaders remain usable for the logged model.

## Further documentation

- [Advanced serving guide](ADVANCED.md): validation details, device behavior,
  configuration mapping, architecture reports and resource requirements.
- [Examples and validation](https://github.com/ai-lab-tech/mlflow-plugin/blob/main/examples/README.md).
- [Send a generated request](https://github.com/ai-lab-tech/mlflow-plugin#send-an-example-request) after deploying your model to Triton.
- [Optional Triton Control deployment](https://github.com/ai-lab-tech/mlflow-plugin/blob/main/packages/mlflow-triton-control/README.md).
