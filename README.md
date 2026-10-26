# mlflow-triton-control

MLflow logging wrappers for scikit-learn, PyTorch and Hugging Face, plus a
deployment target for Triton Control. The logging wrappers require an example
request, retain or infer the MLflow signature, and generate a complete Triton
Python model repository without an LLM. Install this package in the environment
running model logging. Triton Control manages deployment of published repositories.

Requires Python 3.10+ and MLflow `>=3.14,<4`. Automatic CI tests the version
pinned in [`charts/triton-control/values.yaml`](https://github.com/ai-lab-tech/triton-control/blob/main/charts/triton-control/values.yaml)
on `triton-control/main`. CI reads that version on every run, including the
weekly scheduled run. Manual runs can test an additional MLflow version.

```bash
python -m build
python -m pip install dist/mlflow_triton_control-*.whl
mlflow deployments help -t triton-control
```

From this directory, run the tests after installing the wheel:

```bash
python -m pip check
python -m unittest discover -s tests -v
```

The smoke tests verify installed plugin discovery, deployment requests with a
mock HTTP transport, and real model logging, registration, and retrieval with
a temporary SQLite tracking database.

## Log native models with a required example request

Use the plugin's framework module instead of `mlflow.<framework>.log_model`:

```python
from mlflow_triton_control import sklearn, pytorch, transformers, triton

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
or establish GPU correctness. Existing S3 deployment APIs and the direct
`triton.log_model()` API below remain available. To deploy via Triton Control,
publish the generated Triton directory to the selected S3 repository as before.

## Smoke tests

Run from the repository root; each command uses a fresh temporary environment:

```bash
# Test a specific MLflow version
bash smoke-latest.sh 3.14.0

# Test the latest supported MLflow version
bash smoke-latest.sh
```

Replace `3.14.0` with the version to test. No live Kubernetes cluster is needed.
Manual CI runs accept an additional version through `mlflow_version`.

## How MLflow finds the plugin

`mlflow deployments create` is a command provided by MLflow. This package
registers the deployment target through its Python package metadata:

```toml
[project.entry-points."mlflow.deployments"]
triton-control = "mlflow_triton_control.deployment_client"
```

For `--target triton-control://triton-control:8000`, MLflow looks up the
`triton-control` entry point among installed packages, loads this module, and
calls `TritonControlDeploymentClient.create_deployment()`. That class inherits
from MLflow's `BaseDeploymentClient` and sends the deployment request to the
Triton Control API. MLflow reads entry-point metadata; it does not search
through every package's source files. The package must be installed in the
same Python environment as the `mlflow` CLI.

The short hostname works when the client runs in the Service's namespace.
For another namespace, use `triton-control://<service>.<namespace>.svc.cluster.local:8000`,
for example `triton-control://triton-control.blabla.svc.cluster.local:8000`.
Use the actual Service name created by your Helm release.

## Log a Triton model in MLflow

Like NVIDIA's Triton MLflow plugin, this package provides a `triton` model
flavor for a complete Triton model directory. The directory must contain at
least one numbered version; `config.pbtxt` may be included at
its root. The MLflow Model contains an `MLmodel` file with a `triton` flavor
entry and a copy of the entire directory under `model/`.

```python
from mlflow_triton_control import triton

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

The current deployment client still accepts an already-published `s3://` Triton
repository URI. Deploying directly from `models:/...` and copying that version
to the selected S3 profile are separate planned changes. The current Iris
workflow therefore continues to use the S3 URI.

## Argo Workflows

Submit the workflow through Triton Control's authenticated Argo proxy. Mark the
deployment template with these Workflow annotations:

```yaml
metadata:
  name: train-iris
  annotations:
    triton-control.ai/mlflow-deploy-template: deploy
    triton-control.ai/mlflow-deployment-name: iris-classifier
    triton-control.ai/mlflow-s3-profile-name: workflow-training
```

The backend resolves the profile name and its linked Argo artifact repository
for the submitting user. It sets `spec.artifactRepositoryRef` and injects
`TRITON_CONTROL_TOKEN`, `TRITON_CONTROL_S3_PROFILE_ID`, and
`TRITON_CONTROL_S3_BUCKET` into the named `script` or `container` template.
Use the injected ID with `-C s3_profile_id="$TRITON_CONTROL_S3_PROFILE_ID"`.
The token is valid for 60 minutes and limited
to the specified deployment and S3 profile. The Workflow receives a five-minute
completion TTL unless it already defines one, and the Secret receives a
Workflow owner reference for garbage collection.

The opted-in Workflow must have a fixed `metadata.name` and run in the Triton
Control namespace. Direct submissions to Argo bypass this token injection.

## S3 deployment of logged models (0.3.0)

The Triton flavor now stores data under `model/<original-model-name>/` inside
the MLflow Model. Pass the concrete S3 URI of that subdirectory to the deployment
plugin, using an S3 profile that can read the artifact bucket. Resolve proxy
artifact locations against the managed server's S3 root; the deployment client
continues to accept only S3 URIs. Existing models using `data: model` remain
loadable. See the Iris auto-deployment workflow for the complete example.

## Publishing

Publish manually from the repository root, using a fresh output directory for
each release:

```bash
python -m pip install --upgrade build twine
python -m build --outdir dist/mlflow-triton-control-0.3.0
python -m twine check dist/mlflow-triton-control-0.3.0/*
python -m twine upload dist/mlflow-triton-control-0.3.0/*
```

When prompted, enter `__token__` as the username and a PyPI API token scoped to
`mlflow-triton-control` as the password.

## Real saved-model examples

See [the twelve examples](examples/README.md) for four sklearn, four PyTorch and
four Hugging Face cases using existing trained artifacts and independent input
examples. The full runner records local packaging failures and then validates
successful packages with real Triton CPU servers and saved-artifact comparisons.
