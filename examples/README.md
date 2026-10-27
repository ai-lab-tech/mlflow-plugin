# Twelve real saved-model examples

These examples exercise the new `sklearn.log_model`, `pytorch.log_model` and
`transformers.log_model` wrappers with mandatory `input_example` requests.
They use the trained artifacts committed in this repository under `examples/data`.
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
| Hugging Face | `huggingface_classification` | Local BERT sequence classification and tokenizer |
| Hugging Face | `huggingface_ner` | Local BERT token classification and tokenizer |
| Hugging Face | `huggingface_question_answering` | Local BERT, question/context inputs and tokenizer |
| Hugging Face | `huggingface_generation` | Local SFT causal LM, tokenizer and deterministic eight-token generation |

## Run all twelve, including real Triton serving

After cloning, retrieve the tracked binary artifacts once from the repository root:

```bash
git lfs install --local
git lfs pull
```

Both runners default to this checkout's `examples/data`; `--data-root` is only
needed to override that location. No sibling generator repository is required.
Docker must be running. From any working directory:

```bash
python3 /home/ghost/git/mlflow-plugin/examples/run_real_models.py
```

Use the script in this repository, not the old `/tmp/triton-control-log-model-wrapper`
copy. By default, each run creates a new timestamped directory under
`/mnt/container-data/mlflow-tests/`, on the large data disk. Logs, MLflow artifacts,
serving copies, fixtures and pip cache are stored there. The runner refuses this
default if the disk is not mounted. `--output` can override the location; an explicit
output directory must be new. Previous results and running tests are not moved.

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
and MLflow 3.17.2, then run four cases each. Python 3.14 preserves the producer
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
this code change itself does not establish twelve successful serving results.

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
  --output /tmp/mlflow-real-examples-12

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
`1/model.py`, `1/contract.json` and `1/artifact/`.
Native and Triton packages both contain the weights, so allow for duplicate storage.

## What this validates

The wrapper reloads its saved artifact, predicts the example request and checks
the Triton tensor mapping locally. `triton_validation` remains `not_run`.
This is not a numerical benchmark or a claim of live Triton compatibility.
The self-authored example inputs are schema/inference probes, not accuracy tests.

These twelve are coverage examples, not twelve claimed successes. In particular,
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
python3 /home/ghost/git/mlflow-plugin/examples/run_real_models.py \
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
