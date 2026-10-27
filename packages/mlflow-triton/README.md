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


## Examples and optional deployment

See [shared examples and validation](https://github.com/ai-lab-tech/mlflow-plugin/blob/main/examples/README.md) and the
[repository overview](https://github.com/ai-lab-tech/mlflow-plugin/blob/main/README.md). CPU/GPU validation is performed by the
example runner, not by starting a server inside log_model().

For optional Triton Control deployment, install
[mlflow-triton-control](https://github.com/ai-lab-tech/mlflow-plugin/blob/main/packages/mlflow-triton-control/README.md). A complete Triton
model repository is the handoff to any chosen deployment tooling.
