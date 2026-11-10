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

## Register a model

Both registration paths below create a new MLflow Registry version on every
call and apply the same [Registry and Triton version rules](#registry-and-triton-versions).
They work with the sklearn, PyTorch, and Transformers wrappers.

### Register while logging

Pass `registered_model_name` to log and register the trained model in one call:

```python
from mlflow_triton import sklearn

info = sklearn.log_model(
    fitted_pipeline,
    name="classifier",
    input_example=X_sample,
    registered_model_name="classifier",
)
print(info.model_uri)
print(info.registered_model_version)
```

The return value is MLflow `ModelInfo`. Without `registered_model_name`, logging
stores a model without registering it.

### Register a model after training

Log model candidates first, then register the selected one with the plugin's
`register_model()`:

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

The return value is MLflow `ModelVersion`. Arguments are `model_uri`, `name`,
`await_registration_for=300`, and optional keyword-only `tags`. Models without
a Triton flavor use standard MLflow registration. Source references support
logged-model IDs, Registry versions/aliases, and run URIs. The selected source
artifacts are copied without loading or retraining the model. Earlier source
Triton versions and predecessor weights are not downloaded or copied.

For a fresh, unregistered training output, this separate registration step
renumbers its serving directory before registering it, retaining its model ID
and source run. This requires artifact deletion support and permission to
remove the old numbered directory. If the source is already registered
elsewhere, a new logged snapshot is created instead, so existing Registry
references keep their original native weights. Registration while logging
prepares the directory number before upload and does not require this deletion.

Use **`mlflow_triton.register_model`** for these Triton version rules.
MLflow's own `mlflow.register_model()` and **Register Model in the UI** create
Registry versions without running the plugin's config comparison or incrementing
Triton numbers. This package does not install a server-side registration hook.

### Registry and Triton versions

These rules apply to **both** `log_model(..., registered_model_name="classifier")`
and `mlflow_triton.register_model(model_uri, "classifier")`.
Every registration creates a **new MLflow Registry version** with its own native
weights. Compatible versions share one growing Triton repository. Calls for one
registered name must run sequentially.

Starting with a new registered model:

- First registration: Registry version 1 with Triton directory `1/`.
- Matching `config.pbtxt`: Registry version 2 adds Triton directory `2/` to the
  shared repository; the next matching registration adds `3/`, and so on.
- Changed configuration: a new Registry version starts a separate Triton
  repository at `1/`. Registry versions and Triton numbers can then differ.

The first compatible model's Triton directory contains the complete history:

```text
First model's artifacts/model_repository/classifier/
├── config.pbtxt
├── 1/    # Run 1
└── 2/    # Run 2
```

Only the new numbered directory is appended there. The first model's native
weights and root `MLmodel` are never replaced, so MLflow Registry version 1 still
loads Run 1. New snapshots also retain their own serving copy. Their Triton
flavor records `repository_model_id`, pointing deployment to the shared location.

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
| `registered_model_name` | `None` | Create a Registry version on every call; use the latest version for Triton numbering. |
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
latest version of that registered model.

Every call creates a **new MLflow model ID**. With `registered_model_name`,
every call also creates a **new Registry version**, even for identical models.

- Equal configuration: the new snapshot contains the next Triton directory
  (`2/`, `3/`, ...).
- Changed configuration, no predecessor, or no Triton flavor: start at `1/`.

Only predecessor metadata and `config.pbtxt` are downloaded. Earlier native
weights and serving directories are never replaced or copied into the new model.
Only its new numbered serving directory is also uploaded to the shared repository
at the first compatible model. The first model's Triton directory accumulates
compatible versions; later snapshots retain their own serving copy and point to
that shared repository.
Registry versions and Triton numbers can differ after configuration changes.
Without either history parameter, logging creates a new model at `1/`.

```python
info = mlflow_triton.sklearn.log_model(
    updated_model,
    input_example=X_sample,
    name="model",
    model_uri="models:/<model_id>",
)
```

An explicit predecessor's native weights stay unchanged; its shared Triton
repository gains the new version. Registration is optional. `model_uri` and
`registered_model_name` are mutually exclusive: supplying both raises `ValueError`
before model preparation or logging. Each Triton version carries its own serving
requirements at `<version>/requirements.txt`. Calls updating the same shared
repository must run sequentially. A matching predecessor must have a logged-model ID (MLflow 3).

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
