# mlflow-triton

Core MLflow logging wrappers for sklearn, PyTorch and Hugging Face. Generate a
portable Triton repository for Data Scientist demos and PoCs, without requiring
Triton Control, deployment credentials or an LLM. Production optimization remains
a separate ML Engineering task.

```bash
pip install mlflow-triton
```

Requires Python 3.10+ and MLflow >=3.14,<4. Install the framework dependencies
appropriate to your saved model separately. This package registers no deployment
entry point and makes no Triton Control API calls.

## Log native models with a required example request

Use the plugin's framework module instead of `mlflow.<framework>.log_model`:

```python
from mlflow_triton import sklearn, pytorch, transformers, triton

# Already-fitted Pipeline, including its original preprocessing.
info = sklearn.log_model(
    fitted_pipeline,
    name="classifier",
    input_example=example_dataframe,
    registered_model_name="classifier",
)

# Instantiated custom architecture with its saved trained weights loaded.
# Include inference-only class/helper modules, not a training entry point.
torch_info = pytorch.log_model(
    loaded_model.eval(),
    name="network",
    input_example=example_numpy_array,
    code_paths=["inference_architecture"],
    pip_requirements=["torch==<your-version>", "numpy==<your-version>"],
)

# A supported Transformers pipeline containing the model and tokenizer/processor.
hf_info = transformers.log_model(
    loaded_pipeline,
    name="text_model",
    input_example=["Example input"],
    model_config={"max_new_tokens": 32, "do_sample": False, "return_full_text": True},
)

repository = triton.load_model(info.model_uri)
print(repository / "config.pbtxt")
```

`input_example` is a mandatory keyword argument even when `signature=` is supplied.
It is the model's prediction input (DataFrame, NumPy array, named tensor dictionary,
or supported Transformers task input), **not** a Triton HTTP request envelope.
MLflow stores the example with the model: use appropriate non-sensitive examples.
No training is run. Logging executes inference and reloads the saved artifact in
the local environment, so logging requires the framework dependencies and memory
needed to load the model. Framework packages such as torch/transformers remain
optional plugin dependencies; install the versions required by your model.

The wrappers preserve an explicit signature. Otherwise they use MLflow's inferred
signature and fill a missing signature from the example and saved model prediction.
They reject missing/incompatible schemas before logging or registration. Native
flavor options such as `code_paths`, requirements, sklearn's `pyfunc_predict_fn`,
and Transformers' `task`, `processor` and `model_config` are forwarded to native
packaging. `name`, `registered_model_name`, `await_registration_for`, `run_id`,
`metadata`, `tags` and logging `params` are supported. Logging `params` are tracking
metadata, not generation parameters: use `model_config` for Transformers settings.
This is an explicit wrapper API; it does not monkey-patch MLflow or autologging.

The logged artifact contains the native MLflow flavors and an additional `triton`
flavor pointing to:

```text
model_repository/<name>/
  config.pbtxt
  requirements.txt
  validation.json
  1/
    model.py
    contract.json
    artifact/          # Complete local MLflow package, including code and assets
```

Native `mlflow.pyfunc.load_model(info.model_uri)` and native framework loaders
remain usable. The embedded package is a separate copy so the Triton directory
can be deployed independently; this increases artifact storage size. Dependencies
are recorded, not installed into the running Triton server automatically. Build
an environment from `requirements.txt` with a Python version matching Triton's
Python backend stub; custom code dependencies must be included or installed.

The generated configuration uses `backend: "python"` and `max_batch_size: 0`.
Tensor shapes include their full batch dimension. Column schemas map to separate
rank-one tensors; tensor schemas retain their declared shapes. `contract.json`
records mappings between original field names and Triton-safe tensor names.
Only dynamic axes already declared by the signature are retained; a single
example does not prove that other sequence/image dimensions are variable.
Supply an explicit signature when those dimensions need to be dynamic.

`device="cpu"` is the default. `device="gpu"` emits `KIND_GPU` for PyTorch or
Transformers; `model.py` uses the assigned Triton instance device when loading.
Local validation currently loads on CPU even for a GPU target. Models that cannot
load on CPU, sharded/device-mapped models, or tasks unsupported by MLflow pyfunc
need a separate adapter; the wrapper does not claim universal model support.

PyTorch uses Python/pickle packaging by default, **not** an implicit LibTorch/PT2
export. Supply required architecture modules with `code_paths` or dependencies.
Pre/postprocessing outside the supplied model/Pipeline is not captured. Transformers
packages preserve their saved tokenizer, chat template and model configuration,
and expose the MLflow task interface (which can differ from raw `forward()` or
pipeline outputs). Serving is unary prediction, not streaming or a dedicated
high-throughput LLM engine. No GGUF/Q4_K_M compatibility is implied.

Supported signatures contain NumPy-compatible numeric/string tensors or required
scalar DataFrame columns. Optional columns, nested/object schemas, datetime columns,
and scalar tensors require an explicit serving adapter and fail clearly rather than
being flattened or cast speculatively.

`validation.json` distinguishes saved-artifact/local mapping validation from
actual Triton validation. Logging does **not** launch Triton, deploy a service,
or establish GPU correctness. The direct `triton.log_model()` API below remains available. Deployment is a separate concern.

## Log API reference

The three framework wrappers have these public entry points (common keyword
options are abbreviated below; the Python signatures expose them explicitly):

```python
sklearn.log_model(sk_model, *, input_example, **kwargs)
pytorch.log_model(pytorch_model, *, input_example, **kwargs)
transformers.log_model(transformers_model, *, input_example, **kwargs)
```

Pass the model positionally or by its exact parameter name. `input_example` and
all other options must be keyword arguments. The tables below describe the
options handled by this plugin and commonly used native packaging options.

### Required arguments

| Function | Model argument | Required `input_example` |
| --- | --- | --- |
| `sklearn.log_model` | `sk_model`: already-fitted estimator or complete preprocessing `Pipeline`. | A non-empty DataFrame or NumPy array accepted by the saved prediction interface. Preserve column names, order and dtypes. |
| `pytorch.log_model` | `pytorch_model`: instantiated `torch.nn.Module` with trained weights loaded; call `.eval()` first. A `state_dict` or filename alone is insufficient. | A non-empty NumPy input, or named tensor input supported by MLflow's PyTorch prediction interface. Match the model's input dtype and dimensions. |
| `transformers.log_model` | `transformers_model`: supported Transformers pipeline, or component dictionary such as `{"model": model, "tokenizer": tokenizer}`. Include the task's required tokenizer/processor assets. | A non-empty task input supported by MLflow pyfunc, such as a list of text strings or a DataFrame with the task's input fields. |

`input_example=None` and empty examples are rejected. The example remains
mandatory with an explicit signature. It is saved in the artifact and used for
prediction and tensor-mapping validation; it is not training data or a Triton
HTTP request. PyTorch models with any submodule still in training mode are rejected.

### Common optional arguments

These arguments are accepted by all three framework wrappers through `**kwargs`.

| Argument | Type / default | Meaning |
| --- | --- | --- |
| `name` | `str`, `"model"` | MLflow logged-model name. Also supplies the Triton model name unless `triton_model_name` is set. |
| `triton_model_name` | `str` or `None`, `None` | Triton directory and `config.pbtxt` model name. Defaults to `name`; must start with an ASCII letter or underscore, followed by ASCII letters, digits, `_` or `-`. |
| `device` | `"cpu"` or `"gpu"`, `"cpu"` | Target Triton instance kind. sklearn accepts only `"cpu"`; PyTorch and Transformers accept both. Local packaging validation still runs on CPU. This does not choose a GPU index or start Triton. |
| `signature` | MLflow `ModelSignature` or `None`, `None` | Explicit input **and output** schemas, or inference from the saved model and example. Explicit schemas must be compatible with the generated tensor mapping. `False` is rejected. |
| `registered_model_name` | `str` or `None`, `None` | Create a version under this MLflow Registry name. Omit to log without registering. |
| `await_registration_for` | seconds, `300` | MLflow wait timeout for registration, when registration is requested. |
| `run_id` | `str` or `None`, `None` | Optional MLflow run association, forwarded to `Model.log`. Without it, MLflow applies its active-run/default behavior. |
| `metadata` | `dict` or `None`, `None` | Model metadata, passed to native packaging and MLflow logging. |
| `tags` | `dict` or `None`, `None` | Tags for the MLflow logged model. |
| `params` | `dict` or `None`, `None` | Parameters recorded with the MLflow logged model. These do not configure inference; use Transformers `model_config` for generation settings. |

The MLflow Registry version and Triton filesystem version are independent:
logging always generates `<triton_model_name>/1/`, even when the Registry creates
version 2 or higher. There is no `triton_version` argument.

### Native packaging arguments

Remaining keyword arguments are forwarded to **`mlflow.<framework>.save_model`**,
not to that framework's `log_model`. Availability and defaults therefore depend on
the installed MLflow version. This wrapper does not accept every argument exposed
by native `log_model`; for example, use `name`, not `artifact_path`. Unknown or
incompatible options fail during native packaging.

| Argument | Applies to | Default / purpose |
| --- | --- | --- |
| `code_paths` | All three | Native default `None`. List of local inference-code files/directories to package, e.g. custom architecture modules. Include their importable package structure; do not pass training entry points. |
| `pip_requirements` | All three | Native default `None` (native dependency inference/defaults). Supply a requirements list or requirements-file path for a reproducible serving environment. |
| `extra_pip_requirements` | All three | Native default `None`. Add requirements to native defaults/inference. |
| `conda_env` | All three | Native default `None`. Explicit environment dictionary or environment-file path. Follow native MLflow restrictions on combining it with pip requirement arguments. |
| `pyfunc_predict_fn` | sklearn | `"predict"`. Select the estimator method used by the MLflow prediction interface, e.g. `"predict_proba"` when supported by the estimator and MLflow. This also determines the output used to infer the signature. |
| `serialization_format` | sklearn | Native MLflow default. Choose a format supported by the installed sklearn flavor. |
| `serialization_format` | PyTorch | Plugin default `"pickle"`. An explicit alternative must be supported by the installed native flavor and pass saved-model validation. The plugin does not automatically export or guarantee graph compatibility. |
| `pickle_module` | PyTorch | Native default `None`. Optional serialization module for native PyTorch packaging. |
| `task` | Transformers | Native default `None`. Task identifier such as `"text-classification"` or `"text-generation"`. Supply it when native packaging cannot infer the task, especially with custom models/component dictionaries. |
| `processor` | Transformers | Native default `None`. Task-specific processor where required; model/tokenizer components otherwise belong in the pipeline or component dictionary. |
| `model_config` | Transformers | Native default `None`. Saved inference configuration, e.g. `{"max_new_tokens": 32, "do_sample": False}`. Options must be supported by the task and MLflow prediction interface. |
| `save_pretrained` | Transformers | Native default `True`. Must not be `False`: Triton needs the model and tokenizer assets saved locally. |

Custom architecture code and third-party dependencies must be available when the
saved artifact is reloaded. `code_paths` bundles local code; requirement arguments
record installable dependencies. Neither reconstructs a missing architecture.
External preprocessing must already be part of the supplied prediction interface.

### Examples with explicit options

The objects below are already loaded/fitted; these calls do not train them.
Configure the MLflow tracking URI/experiment using MLflow's normal API before
logging, if required for your environment.

```python
from mlflow_triton import sklearn, pytorch, transformers, triton

# sklearn: named DataFrame columns and preprocessing travel with the Pipeline.
sk_info = sklearn.log_model(
    sk_model=fitted_pipeline,
    input_example=example_dataframe.iloc[:2],
    name="classifier",
    device="cpu",
    pyfunc_predict_fn="predict_proba",
    registered_model_name="demo-classifier",
)

# PyTorch: custom architecture instance, trained weights already loaded.
pt_info = pytorch.log_model(
    pytorch_model=loaded_model.eval(),
    input_example=example_numpy_array[:2],
    name="network",
    device="gpu",
    code_paths=["inference_architecture"],
    pip_requirements="requirements-serving.txt",
)

# Hugging Face: a pipeline includes the tokenizer; use saved generation options.
hf_info = transformers.log_model(
    transformers_model=loaded_generation_pipeline,
    input_example=["Explain what a model signature describes."],
    name="text-generator",
    triton_model_name="text_generator",
    device="gpu",
    task="text-generation",
    model_config={"max_new_tokens": 32, "do_sample": False},
)

print(hf_info.model_uri)
repository = triton.load_model(hf_info.model_uri)
print(repository / "config.pbtxt")
print(repository / "1/model.py")
```

### Return value and validation

All three calls return MLflow `ModelInfo`. Use `info.model_uri` to load the saved
model or resolve its Triton repository, and `info.registered_model_version` when
registration was requested. `triton.load_model(info.model_uri)` returns a local
`pathlib.Path` to the model-name directory, downloading artifacts as necessary.

Before logging/registration, the wrapper saves the native model, reloads it,
executes the example prediction, derives/checks the signature and validates the
input/output tensor mapping. Unsupported schemas, missing custom code, invalid
devices or failed predictions raise an exception. A successful return means local
packaging validation passed; actual Triton serving and GPU execution must be
tested separately. There are no `deploy`, `validate_triton`, `max_batch_size` or
GPU-index arguments in these wrappers.

## Log a Triton model in MLflow

Like NVIDIA's Triton MLflow plugin, this package provides a `triton` model
flavor for a complete Triton model directory. The directory must contain at
least one numbered version; `config.pbtxt` may be included at
its root. The MLflow Model contains an `MLmodel` file with a `triton` flavor
entry and a copy of the entire directory under `model/`.

```python
from mlflow_triton import triton

model_info = triton.log_model(
    triton_model_path="/tmp/model-repository/iris_classifier",
    name="triton-model",
    registered_model_name="iris-serving",
)
print(model_info.registered_model_version)
```

`triton.load_model("models:/iris-serving/1")` downloads the registered model
and returns the local Triton model directory. Omit `registered_model_name` to
save a model without creating a Registry version. This API is in the plugin
package, not in MLflow's `mlflow.triton` namespace.


### Arguments for an existing Triton model directory

This separate API packages an **existing** Triton model directory. It does not
create `model.py`, infer a signature or change its configuration.

```python
triton.log_model(
    triton_model_path,
    name="triton-model",
    registered_model_name=None,
    await_registration_for=300,
    run_id=None,
    metadata=None,
)
```

| Argument | Required / default | Meaning |
| --- | --- | --- |
| `triton_model_path` | Required | Path to one model-name directory containing at least one positive numbered version directory. Symbolic links are rejected. Pass the model directory, not the parent repository containing multiple models. |
| `name` | `"triton-model"` | MLflow logged-model name. The existing Triton directory name and configuration are preserved. |
| `registered_model_name` | `None` | Optional MLflow Registry name. |
| `await_registration_for` | `300` | Registration wait timeout in seconds. |
| `run_id` | `None` | Optional MLflow run association. |
| `metadata` | `None` | Optional model metadata dictionary. |

Returns MLflow `ModelInfo`. This API has no `input_example`, `signature`, `device`,
`tags`, `params` or arbitrary `**kwargs` parameters. Configure serving directly in
the supplied Triton directory before logging it.

## Examples and optional deployment

See [shared examples and validation](https://github.com/ai-lab-tech/mlflow-plugin/blob/main/examples/README.md) and the
[repository overview](https://github.com/ai-lab-tech/mlflow-plugin/blob/main/README.md). CPU/GPU validation is performed by the
example runner, not by starting a server inside log_model().

For optional Triton Control deployment, install
[mlflow-triton-control](https://github.com/ai-lab-tech/mlflow-plugin/blob/main/packages/mlflow-triton-control/README.md). A complete Triton
model repository is the handoff to any chosen deployment tooling.

## Packaging and concurrent loading

Packaging checks the target name and device before saving or predicting. It builds
in a temporary sibling of the destination and renames the completed directory into
place, cleaning up staging on failure. The native artifact and independent serving
copy remain separate; intermediate full-package copies are avoided. MLflow may
still copy/upload the completed artifact during logging. `TMPDIR` selects the outer
logging workspace; choose a disk with enough room for both model copies and MLflow's
own temporary storage.

Calls through the runtime's `load_predictor` are serialized while MLflow device and
offline environment settings are temporarily applied, including restoration after
failure. This lock does not protect unrelated code that modifies `os.environ`,
direct MLflow loads, or separately imported runtime copies. Coordinate those callers
or use separate processes. Prediction requests are not serialized by this lock.
