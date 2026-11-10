# Advanced serving guide

Details for inspecting, validating and deploying models logged with `mlflow-triton`.
For installation and Python arguments, see the [package README](README.md).

## Validation details

Before logging/registration, the wrapper executes the example prediction using
MLflow's prediction adapter around the supplied model, derives/checks the signature,
saves the native model and validates the input/output tensor mapping. It reuses
the existing weights. Unsupported schemas, invalid devices or failed predictions
raise an exception. A successful default return does not establish that the saved
artifact can be reloaded in another environment.

Use `validate_saved_model=True` to instead validate a CPU reload of the saved
artifact, including whether its custom code can be imported locally. The original
model remains in memory during this check. `validation.json` records the selected
mode and validation device; actual Triton serving and GPU execution must be
tested separately. There are no `deploy`, `validate_triton`, `max_batch_size` or
GPU-index arguments in these wrappers.

## Generated files and serving behavior

The logged artifact contains the native MLflow flavors and an additional `triton`
flavor pointing to:

```text
model_repository/<triton_model_name>/
  config.pbtxt
  request.json        # Or request.bin + request.headers for binary inputs
  validation.json
  1/
    requirements.txt
    model.py
    artifact/          # Complete local MLflow package, including code and assets
```

`mlflow.pyfunc.load_model(info.model_uri)` and MLflow framework loaders
remain usable for models logged with the framework wrappers. The embedded package is a separate copy so the Triton directory
can be deployed independently; this increases artifact storage size. Dependencies
are recorded, not installed into the running Triton server automatically. Build
an environment from `<version>/requirements.txt` with a Python version matching
Triton's Python backend stub; custom code dependencies must be included or installed.

The generated configuration uses `backend: "python"` and `max_batch_size: 0`.
Tensor shapes include their full batch dimension. Column schemas map to separate
rank-one tensors; tensor schemas retain their declared shapes. The `mlflow_contract` parameter in `config.pbtxt`
records mappings between original field names and Triton-safe tensor names.
Only dimensions marked as dynamic in the signature are retained; a single
example does not prove that other sequence/image dimensions are variable.
Supply an explicit signature when those dimensions need to be dynamic.

`device="cpu"` is the default. `device="gpu"` emits `KIND_GPU` for PyTorch or
Transformers; `model.py` uses the assigned Triton instance device when loading.
Default validation uses the supplied model's current device, independently of the
Triton target. Optional saved-artifact validation loads on CPU even for a GPU
target. Sharded/device-mapped models or tasks unsupported by MLflow pyfunc
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

`validation.json` identifies in-memory or saved-artifact prediction validation,
records whether the saved artifact was checked, and distinguishes local mapping
validation from actual Triton validation. Logging does **not** launch Triton, deploy a service,
or establish GPU correctness.

## Model version updates

Every logging call creates a separate logged-model ID and artifact location.
`registered_model_name` also creates a Registry version on every call. Existing
native packages are never overwritten, so loading Registry version 1 still
returns the model originally registered as version 1.

`model_uri` selects a predecessor with unchanged native weights (`models:/<model_id>`, a Registry
version/alias, or `runs:/<run_id>/<path>`). Alternatively,
`registered_model_name` selects the latest Registry version for config comparison.
The two parameters are mutually exclusive.

| Predecessor config | Logged model | Triton directory | Registry with name |
| --- | --- | --- | --- |
| Matching | New ID | Next number, also appended to shared repository | New version |
| Different | New ID | `1/` | New version |
| None, or no Triton flavor | New ID | `1/` | New version |

For two compatible calls, Registry versions 1 and 2 point to separate snapshots
containing `1/` and `2/`, respectively. The first model's Triton repository
also gains `2/`, so Triton can load both from a single location. A third call with changed configuration
creates Registry version 3 with directory `1/`. No previous weights are copied.
Only predecessor metadata, directory listings, and `config.pbtxt` are read.
Only the new numbered serving directory is uploaded to the shared owner; root
`MLmodel`, native weights, config, and previous numbered directories are not
uploaded or overwritten there.

Config comparison ignores protobuf whitespace and comments, preserving values
and field order. Weights, backend code, and requirements are not compared. The
`version` inside `mlflow_contract` describes its format, not a model version.
Each Triton directory has its own requirements. Logging does not deploy or
reload Triton.

The flavor records `model_version`, `repository_group`, and, for matching
configurations, `model_uri` and `repository_model_id`. The latter identifies the
first logged model that owns the shared Triton repository. Deployment resolves
this ID before choosing the repository URI, while native MLflow loading continues
to use the requested snapshot. Deployment reads the requested version's
requirements from that shared directory. Triton's default policy serves the
latest version; clients can request a specific Triton version. Each call applies its own `name`, `params`, and
`tags` and is associated with its own active or supplied run.
Calls using the same registered name must be sequential for sequential Triton
numbering; there is no distributed lock. Native snapshots are preserved on failure, but serving uploads are not
transactional: an interrupted operation can leave an unregistered or partial
numbered directory. Existing version directories are never overwritten. Version
allocation uses the maximum of predecessor metadata and existing numeric
directories, preventing reuse when branching from an older snapshot.

### Separate registration

`mlflow_triton.register_model(model_uri, name)` compares the source's
configuration with the latest registered model. Every call creates a Registry
`ModelVersion`. Matching configs append only the new numbered directory to the
shared repository and advance the Triton number; different configs
start at `1/`. The selected source artifacts are staged without loading weights.
Earlier source directories and predecessor weights are not downloaded or copied.

A fresh unregistered source keeps its model ID and training-run association.
When renumbering is necessary, its new Triton directory and metadata are uploaded
and the old numbered directory is removed before registration. Artifact storage
must support deletion and the caller must have deletion permission. Upload or
deletion failures abort registration; artifact operations are not transactional.
A source already referenced by a visible Registry version is instead copied to
a fresh logged model before renumbering, preserving its native package and
existing serving versions. Its shared serving repository can still gain the
new directory.
Registry access failures abort the operation. Serialize registration with other
operations on the source; there is no lock across clients.

The standard `mlflow.register_model` and UI registration also create Registry
versions, but do not compare Triton configurations or renumber artifacts.
This package does not install a server-side registration hook.

## Disk space and concurrent loading

Logging creates two copies of the model: the MLflow model and a self-contained
copy for Triton serving. Logging reads only predecessor metadata and config;
it never copies earlier model versions. Existing native snapshots and serving versions remain unchanged; the shared
serving repository accumulates one copy of each new compatible version. Allow disk space for the new model's two copies and
MLflow's temporary files. Set `TMPDIR` before logging to choose a temporary
directory on a disk with enough space. Incomplete staging files are cleaned up if packaging fails.

The serving runtime temporarily changes MLflow device and offline environment
settings while loading a model, then restores them. It serializes its own model
loads, but cannot coordinate with unrelated code that changes `os.environ` or
loads models directly through MLflow. Use separate processes for those operations
if they could run concurrently. Prediction requests are not serialized by this lock.

## MLflow mapping in Triton configuration

Generated repositories store the MLflow-specific mapping as JSON in the
`parameters.mlflow_contract.string_value` field of `config.pbtxt`. It contains a
mapping version, framework flavor, input/output structure (columns, single tensor
or named tensors), original field names and text/binary semantics. Tensor names
link mapping entries to the standard `input` and `output` configuration fields;
datatypes and shapes are read exclusively from those fields.

Triton passes the configuration to `model.py.initialize()` as `args["model_config"]`.
The runtime validates and reconstructs its tensor mapping from it. Device placement
still comes from Triton's assigned instance kind/device ID. No separate
`contract.json` is generated or read by the new runtime. Unsupported mapping
versions, missing tensor mappings and nonzero `max_batch_size` fail initialization.

Existing logged artifacts retain their original `model.py` and `contract.json`.
Re-log a model to produce the new layout; do not mix old generated runtime files
with new configuration. Editing a mapping or tensor schema must remain compatible
with the saved model's prediction interface and requires renewed validation.

## Ready-to-send inference example

Each framework wrapper generates one `request.json` beside `config.pbtxt`, using
values from `input_example`, concrete tensor shapes and all configured output names.
See the [central request instructions](https://github.com/ai-lab-tech/mlflow-plugin#send-an-example-request)
for curl commands. No per-model shell script or instruction file is generated.
Logging does not send a request.

For NaN/infinity or binary byte inputs, the example instead consists of
`request.bin` (complete HTTP body, including its JSON prefix) and `request.headers`
(the HTTP content type and JSON-prefix byte length). This uses
[Triton's binary tensor extension](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_binary_data.html)
without changing input values. Outputs are requested as binary in that case.

Example files contain the supplied input values, just like MLflow's stored input
example; use non-sensitive examples.

## Inspect model architecture without loading weights

PyTorch and Transformers wrappers automatically include `architecture.json` in the
MLflow artifact and its self-contained Triton `<version>/artifact/` copy. Read this JSON
without importing custom model code or loading weights. It records the model class,
module classes, parameter/buffer shapes and dtypes (never tensor values), and relative
paths to packaged Python and configuration files. Transformers reports component
classes such as the tokenizer; its saved `config.json` contains model configuration.

| Argument | Required | Meaning |
| --- | --- | --- |
| `architecture_init_kwargs` | No | JSON object documenting constructor arguments, supplied by the caller. Defaults to unknown; never guessed or passed to the constructor. |

```python
from mlflow_triton import pytorch

pytorch.log_model(
    model,
    input_example=example,
    code_paths=["custom_network.py"],
    architecture_init_kwargs={"input_size": 16, "hidden_size": 32},
)
```

The same optional argument is supported by `mlflow_triton.transformers.log_model`.
Include only non-sensitive architecture settings, never credentials or training data.
This is documentation, not an automatic reconstruction recipe: module structure
cannot describe arbitrary `forward()` logic or preprocessing. Constructor settings
are caller-declared and are not verified against the model. Required custom code
must still be included using `code_paths` or installed dependencies. Existing artifacts
are unchanged; the report is generated on subsequent logging. Serving inputs,
outputs, weights and `config.pbtxt` are unaffected.
