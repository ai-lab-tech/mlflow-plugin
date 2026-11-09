# Examples

## Quick starts

Start here to train a small classifier and log it with Triton serving files.
Neither example requires Docker, a GPU or a running Triton server.

| Example | How to run |
| --- | --- |
| [Python quick start](quickstart.py) | Install `mlflow-triton` and `scikit-learn`, then run `python examples/quickstart.py` from the repository root. |
| [Google Colab notebook](https://colab.research.google.com/github/ai-lab-tech/mlflow-plugin/blob/main/examples/quickstart_colab.ipynb) | Run the cells in order. Includes installation and lists generated artifact paths without downloading model weights. |
| [Registry versioning smoke test](smoke_versioning.py) | Run `python examples/smoke_versioning.py --output /tmp/triton-versioning-smoke` with a new output directory. Trains three Iris classifiers, checks compatible and incompatible updates in a real local MLflow registry, and retains the database, artifacts, and `report.json`. |

The `versioning-smoke` job in GitHub Actions runs this check against MLflow
3.14.0 and the latest supported release using the built core wheel. Each run
uploads the report, SQLite registry, model artifacts, and log for seven days,
including partial output on failure. This check must pass before publishing.
The JSON report records failures as well as successes, and smoke checks remain
active when Python optimization is enabled. Serving validation selects the
highest numeric folder, matching the generated default Triton version policy.

## Core smoke test against a real MLflow server

[smoke_core_server.py](smoke_core_server.py) starts an MLflow server on a temporary
loopback port. The core wrappers use its HTTP tracking and Registry APIs, and
model files pass through the server's artifact proxy. Each scenario follows
Arrange–Act–Assert and uses real models, server calls, and artifact downloads.

On Linux or macOS, install the core package, then the CPU framework dependencies
(the PyTorch CPU index below is used by the Linux CI runner):

```bash
python -m pip install -e packages/mlflow-triton
python -m pip install "torch==2.9.0" --index-url https://download.pytorch.org/whl/cpu
python -m pip install "transformers==4.57.1"
python examples/smoke_core_server.py --framework all --output /tmp/core-server-smoke
```

Use a new output directory for each run. `--framework sklearn` runs the sklearn
scenarios without the PyTorch or Transformers dependencies. The other selections
are `pytorch` and `transformers`; selected dependencies are required, and missing
ones fail the run rather than skipping coverage. Tiny Transformers weights and
tokenizers are created locally, with Hub downloads disabled.

The suite covers:

- sklearn prediction and probability logging, inferred and supplied signatures,
  active runs and explicit `run_id`, model tags, parameters, and metadata.
- Saved-model validation, generated Triton config/request/requirements files,
  native MLflow reloads, and predictions from downloaded serving copies.
- Registry versions, compatible and incompatible updates, and alias resolution.
- PyTorch custom code loaded from downloaded artifacts after the original source
  is removed, architecture metadata, and rejection of training-mode modules.
- Transformers pipelines and component dictionaries, saved tokenizer assets,
  architecture reports, and deterministic generation after reloading.
- Invalid inputs rejected before any model is published.

The server stops when the test finishes, including on failure. `report.json`,
`server.log`, the SQLite registry, and all model artifacts remain in the output
directory. This tests MLflow integration and packaging; actual Triton execution
is covered by the serving-validation tooling below.

The `core-server-smoke` GitHub Actions job runs all scenarios against MLflow
3.14.0 and the latest supported release, using the built core wheel. It uploads
results for seven days and must pass before publishing.

### Test a new MLflow release

In GitHub, open **Actions → MLflow packages CI → Run workflow**, choose the branch,
and set **mlflow_version**:

- An exact version, such as `3.17.2`, runs every test job against that version.
- `latest` tests the newest stable release allowed by `mlflow>=3.14,<4`.
- An empty value runs the standard `3.14.0` and `latest` matrix.

Exact prerelease versions are accepted within the supported range. The selected
version applies to the package tests, framework tests, versioning smoke, and real
HTTP server smoke. Manual runs test the packages without publishing a release.
Pull requests, pushes, and scheduled runs continue to use the standard matrix.

The same test can be started with GitHub CLI:

```bash
gh workflow run mlflow-plugin-ci.yml -f mlflow_version=3.17.2
```

Use `--ref <branch>` to test another branch. Download the smoke artifacts from the
completed workflow run to inspect the actual installed version and results.

## Saved-model serving validation

Use the suite below to test existing saved models against actual Triton responses.
It requires model fixtures and Docker for the full serving run.

These examples exercise the new `sklearn.log_model`, `pytorch.log_model` and
`transformers.log_model` wrappers with mandatory `input_example` requests.
They use a shared 13-case catalog: five synthetic-data fixtures in this repository
and eight fixtures in a private repository pending license review. See
[data setup](data/README.md) to prepare a combined local dataset.
Binary artifacts are tracked with Git LFS; see [data provenance](data/README.md).
They do not train, download models, import training scripts, read benchmark request
fixtures or expected answers, or use an LLM. The full runner below starts Triton for validation.

| Family | Example | Coverage |
| --- | --- | --- |
| sklearn | `sklearn_mixed_columns` | Saved column transformer, imputation, categorical encoding and classifier |
| sklearn | `sklearn_gradient_boosting` | Saved gradient boosting classifier, ten numeric features |
| sklearn | `sklearn_scaled_svm` | Separately saved fitted StandardScaler plus SVC |
| sklearn | `sklearn_scaled_mlp` | Separately saved fitted MinMaxScaler plus MLP |
| PyTorch | `pytorch_polynomial` | Saved custom polynomial module |
| PyTorch | `pytorch_custom_network` | Saved custom TwoLayerNet architecture and state_dict |
| PyTorch | `pytorch_mnist_lstm` | Image tensor, recurrent network and saved weights |
| PyTorch | `pytorch_dcgan` | Latent tensor to image tensor, saved generator weights |
| Hugging Face | `huggingface_custom` | Own configuration/model classes; saved BERT weights with custom inference behavior |
| Hugging Face | `huggingface_classification` | Local BERT sequence classification and tokenizer |
| Hugging Face | `huggingface_ner` | Local BERT token classification and tokenizer |
| Hugging Face | `huggingface_question_answering` | Local BERT, question/context inputs and tokenizer |
| Hugging Face | `huggingface_generation` | Local SFT causal LM, tokenizer and deterministic eight-token generation |

## Run all examples, including real Triton serving

After cloning, retrieve the tracked binary artifacts once from the repository root:

```bash
git lfs install --local
git lfs pull
```

Both runners default to `MLFLOW_TRITON_DATA_ROOT` when set, otherwise this
checkout's public `examples/data`. Use `--data-root` for a prepared combined dataset. An alternative data root must use the same
`<case-name>/` layout. Prepared cases are direct children of the chosen data directory. No sibling generator repository is required.
Docker must be running. From the repository root:

```bash
python examples/run_real_models.py
```

Configure a storage directory suitable for your machine. For example, on a
machine with a second SSD mounted at `/mnt/container-data`:

```bash
python examples/run_real_models.py --device gpu --output-root /mnt/container-data/mlflow-tests
```

`--output-root` creates a timestamped subdirectory per run. Alternatively, set
`MLFLOW_TRITON_TEST_ROOT` once in your shell and run without a storage argument:

```bash
export MLFLOW_TRITON_TEST_ROOT=/mnt/container-data/mlflow-tests
python examples/run_real_models.py --device cpu
```

`--output /any/new/directory` selects an exact output directory. CLI options take
precedence over the environment variable. Any writable location is allowed;
there is no hardcoded SSD path or mount requirement. Without an output option
or the environment variable the runner asks you to configure storage instead
of silently writing to the system disk. Logs, MLflow packages, caches, packaging
scratch and host temporary files go under the chosen run directory.

Docker images, build cache and serving-container scratch remain in Docker's
configured data directory, which the runner reports but does not change.
Configure Docker storage separately if its current disk has insufficient space. Previous test results are not moved.

The launcher first prepares CPU Triton images with Python 3.11 and 3.14 and
matching Python backend stubs. It reuses its existing image tags on later runs.
The initial build downloads image/build dependencies and can take considerable
time and disk space. Build logs are retained under `triton-runtime-python-*/`.
The default Triton base is `nvcr.io/nvidia/tritonserver:26.06-py3`; override it with
`--triton-base-image`. You can supply `--triton-images path/to/images.json` instead
of building; the mapping must contain `3.11` and `3.14` image names when running
all families. The old benchmark map containing only 3.7/3.10 is insufficient.
The server checks both the actual interpreter version and linked backend stub.

Three isolated packaging containers install saved inference dependency versions
and MLflow 3.17.2, then run four sklearn, four PyTorch and five Hugging Face cases. Python 3.14 preserves the producer
version of the saved custom PyTorch classes. HF additionally installs the matching
CPU torchvision 0.24.0 for torch 2.9.0. sklearn explicitly uses CloudPickle for
these trusted, already-deserialized Python pipelines; PyTorch uses MLflow's
compatible default pickle adapter, not raw cloudpickle as a pickle module.

Every successful package then gets a fresh **actual Triton CPU server**:

1. Install the recorded model and family dependencies.
2. Start the generated model repository with its original config and model.py.
3. Wait for server and model readiness.
4. Send an HTTP inference request using binary tensor transport.
5. Compare all output names, shapes, dtypes and values with an independent
   prediction from the saved native MLflow artifact. Integers and strings are
   exact; floats use `rtol=1e-5`, `atol=1e-6`, `equal_nan=True`.
6. Save server/comparison logs and remove the temporary server container.

The server has no published host port and its external network is disconnected
before inference. Reference fixtures live outside the deployable model directory.
Models are loaded offline; no training or model download occurs.

The combined `summary.json` reports local and actual Triton validation separately.
Local failures block serving and never count as Triton passes. Any failed or
unexecuted case makes the command exit nonzero. Server details are in
`triton/<case>/result.json`, `server.log`, `install.log`, and `comparison.log`.
Packaging logs remain `sklearn.log`, `pytorch.log` and `transformers.log`.

Use repeatable `--family sklearn` for a subset, `--timeout 7200` for dependency
installation/packaging, `--startup-timeout 300` for server readiness and
`--inference-timeout 300` for a request. Runtime image builds have separate live
build logs. This is one example request per model, not the 88-case accuracy benchmark.

The orchestration and comparisons have unit tests, including mismatched Python
versions and failed inference. The full Docker/Triton run remains user-operated;
unit tests do not establish successful serving results for the catalog.

## Run

Use an environment containing this plugin and the saved artifacts' compatible
framework versions. In particular, these sklearn checkpoints require
`scikit-learn==1.7.2`; the script rejects a mismatching version before deserialization.
Each source bundle includes `requirements.txt`. For older pinned dependencies use
an appropriate Python version (the saved sklearn environment uses NumPy 1.26.4).
Different families can run in separate environments via repeated `--case` flags.
Only load your trusted artifacts: the PyTorch examples load the already saved
`reference-model.pkl` architecture with cloudpickle, then strictly reload the
corresponding saved state_dict. They never read reference inputs or outputs.

From the plugin directory:

```bash
python examples/real_models.py --list

python examples/real_models.py \
  --output /tmp/mlflow-real-examples

# Or select one example, with a new output directory:
python examples/real_models.py \
  --output /tmp/mlflow-custom-network \
  --case pytorch_custom_network
```

The runner creates a fresh local SQLite MLflow store, logs each successful model
and writes `summary.json` with its MLflow model URI. Existing output directories
are refused. A failure is retained and subsequent examples continue; any failure
makes the command exit nonzero. The source artifacts are never modified.

Each successful log contains the native MLflow package plus
`model_repository/<name>/config.pbtxt`, `requirements.txt`, `validation.json`,
`1/model.py` and `1/artifact/`. The MLflow mapping is stored in the
`mlflow_contract` parameter of `config.pbtxt`.
Native and Triton packages both contain the weights, so allow for duplicate storage.

## What this validates

By default, the wrapper predicts the example request using the supplied model
and checks the Triton tensor mapping locally without reloading model weights.
Set `validate_saved_model=True` on a framework wrapper to instead reload and
validate the saved artifact on CPU, requiring memory for another model copy.
`validation.json` records the mode; `triton_validation` remains `not_run`.
This is not a numerical benchmark or a claim of live Triton compatibility.
The self-authored example inputs are schema/inference probes, not accuracy tests.

These thirteen cases are coverage examples, not claimed successes. In particular,
NER returns nested records in some MLflow/Transformers versions; the current
wrapper rejects unsupported nested signatures. Such a result stays a failure;
the example does not silently flatten, drop or stringify predictions to pass.
QA and generation likewise exercise the native MLflow prediction contract.
Framework dependency failures are reported separately in each case's error.

Local loader checks completed for all four PyTorch examples and for SVM/MLP.
The current inspection environment has sklearn 1.9.1: mixed-column and boosting
load/predict checks failed there, so matching the saved environment is required.
Use the full runner above to measure updated packaging and actual serving results.

## Retest selected cases

`--case` is repeatable. For the four failures from the first serving run:

```bash
python examples/run_real_models.py \
  --case sklearn_mixed_columns \
  --case huggingface_classification \
  --case huggingface_question_answering \
  --case huggingface_generation
```

This creates a separate report on the data disk and leaves the eight passed cases
and their original report untouched. It does not merge scores automatically.
Packaging containers bind their `/tmp` (including Python environments and MLflow
staging files) to `<output>/work/<family>`, with `TMPDIR=/tmp`. These work folders
are retained with the run. Docker serving scratch uses Docker's data disk.

## Select CPU or GPU serving

```bash
# All available cases on CPU (13 with the combined dataset)
python examples/run_real_models.py --device cpu

# Available GPU cases (4 PyTorch + 5 Hugging Face with the combined dataset)
python examples/run_real_models.py --device gpu

# One GPU case
python examples/run_real_models.py --device gpu --case huggingface_generation
```

`--device` defaults to `cpu` and is forwarded to the public `log_model` wrappers.
Selecting GPU without case/family filters selects available GPU-capable framework
examples. Set `MLFLOW_TRITON_DATA_ROOT` or pass `--data-root` for private cases. Explicitly selecting sklearn with GPU is an error; it never silently
runs sklearn on CPU in a GPU report.

GPU runs require an NVIDIA GPU, a compatible driver and Docker GPU support.
The runtime builder enables GPU support in the Python backend stub and uses
separate `-gpu` image tags. If supplying `--triton-images`, supply images built
for the requested device, with matching Python versions. GPU dependency lists
retain framework versions but replace CPU-only wheel pins with their public
versions from PyPI. The runtime verifies that the resulting PyTorch build has
CUDA support and can execute a CUDA operation before launching the server.
Missing GPU access or incompatible CUDA dependencies fail clearly.

Logging and native reference predictions still run on CPU. Triton receives
`KIND_GPU` configuration and Docker GPU access; the client also verifies Triton's
reported instance kind. GPU serving is compared with a separate native GPU prediction; CPU/GPU
differences remain separate diagnostics. All comparisons retain the same tolerances. GPU runs use the first CUDA-visible GPU.
Each case records `device` in the report, with GPU preflight details in `gpu.log`.
No GPU inference result has been established by these code/unit-test changes;
run the command above to measure it. Triton/native-GPU differences remain serving failures. CPU/GPU differences
are recorded separately; the tolerances are never automatically relaxed.

## Serve using only each model's generated requirements

Use this launcher for the original eight GPU cases: four PyTorch cases plus
Hugging Face classification, NER, question answering and generation:

```bash
bash examples/run_eight_gpu_requirements.sh \
  --data-root /path/to/combined-test-data \
  --output-root /path/to/test-output
```

The launcher passes `--requirements-only` to `run_real_models.py`. Each model gets
a fresh Triton container and a fresh virtual environment. The serving environment
installs only that generated model's `requirements.txt` and its transitive
dependencies, then runs `pip check`. Family dependency lists and test client
packages are not added to it. The runtime's inherited Python path, including
bundled DALI packages, is replaced by the new model environment's path.

HTTP/protobuf client dependencies are installed in a separate virtual environment.
The client prepares the tensor contract for the native GPU reference, so that
reference runs in the model environment without importing client dependencies.
Actual Triton inference is compared against that independent native GPU prediction;
CPU/GPU differences remain separate diagnostics with unchanged tolerances.
Dependency installation or validation failures remain failures; the runner does
not add missing model dependencies automatically.

For automatic CPU packages, the generated file includes exact CPU Torch-family
pins and package-specific wheel sources. This also works with
`--requirements-only`: no installer-specific CUDA-to-CPU substitution is needed.
Explicit user requirements retain their build choices. A strict test still
creates a fresh model environment to check dependency completeness, so it does
not measure reuse of packages already present in a production image.

Packaging still uses the saved producer dependencies to load and log the model.
The strict dependency check applies to the subsequent serving environment.
The default runtime base remains Triton `26.06-py3`, with the matching Python
backend stub for Python 3.14 (PyTorch) or 3.11 (Hugging Face). Existing compatible
runtime images are reused, while model environments are recreated for every case.

Each `triton/<case>/` directory retains `model-requirements.txt`, `install.log`,
`dependency-check.log`, `requirements-lock.txt`, `model-install-report.json`,
`client-install.log`, `client-install-report.json` and the normal serving reports.
The eight-case launcher excludes `huggingface_custom`; use the main runner with
`--requirements-only --device gpu` to select all available GPU cases, including it.

## Native GPU reference and CPU/GPU differences

GPU runs now load the same saved MLflow artifact on `cuda:0` in a separate process,
perform prediction through MLflow directly, and exit before Triton starts. The
reference reuses tensor/schema conversion helpers, but never calls Triton's model
`initialize()` or `execute()`. It uses the same request and dependency environment.

The report distinguishes:

- `triton_vs_native_gpu`: primary serving comparison on the same device.
- `native_gpu_vs_cpu`: numerical differences between native GPU and the original CPU reference.
- `triton_vs_cpu`: original cross-device serving comparison, still reported.

`triton_validation` for a GPU run now means **Triton versus native GPU**, identified
by `comparison: saved_native_artifact_gpu`. A CPU/GPU mismatch does not change
that serving result, but remains explicitly `failed` in its diagnostic entry.
Earlier reports are untouched and used CPU reference predictions for GPU serving.
CPU tests continue to compare Triton with the CPU reference.

All checks keep `rtol=1e-5`, `atol=1e-6`, and `equal_nan=True`. Numeric diagnostics
include mismatched element counts and the largest absolute difference among finite
values. Details are retained in `triton/<case>/comparison-details.json` and the
combined report; native reference logs are in `native-gpu.log`.

Retest only DCGAN, preserving all earlier results:

```bash
python examples/run_real_models.py --device gpu --case pytorch_dcgan \
  --output-root /mnt/container-data/mlflow-tests
```

## Hugging Face with custom Python code

`huggingface_custom` is an additional (13th) example. It provides its own
`CustomBertConfig` in `configuration_custom.py` and
`CustomBertForSequenceClassification` in `modeling_custom.py`. The custom forward
method divides logits by a configurable temperature (1.7), so loading the standard
BERT implementation instead would change the predictions.

The weights and tokenizer are copied unchanged from the existing locally
fine-tuned `hf_glue` checkpoint. No training or download was performed for this
example. The temperature is an integration-test setting, not a fitted calibration
or an accuracy improvement. Provenance is stored alongside the bundle.

The local config supplies `auto_map`; the example explicitly loads this reviewed
local code with `trust_remote_code=True` and `local_files_only=True`. It then calls
the public `transformers.log_model(..., input_example=..., device=...)` wrapper.
MLflow packages the registered custom source files together with the model.

```bash
python examples/run_real_models.py --case huggingface_custom --device cpu \
  --output-root /mnt/container-data/mlflow-tests

python examples/run_real_models.py --case huggingface_custom --device gpu \
  --output-root /mnt/container-data/mlflow-tests
```

Local packaging passed. A regression test uses tiny synthetic weights to verify
the custom behavior, reloads the packaged artifact in a fresh process with the
original source removed and a fresh HF module cache, and verifies that deleting
the packaged architecture file causes a failure. Actual CPU/GPU Triton execution
for this new example remains to be run; the prior twelve passing cases do not
count as validation of this additional case.

## Package installation

Examples use only the `mlflow-triton` core. For local execution install it with
`pip install -e packages/mlflow-triton` from the repository root. The Docker
runner mounts the core source directory; no Control integration is installed or
needed. See the [core API documentation](../packages/mlflow-triton/README.md).
