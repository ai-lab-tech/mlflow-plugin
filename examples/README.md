# Twelve real saved-model examples

These examples exercise the new `sklearn.log_model`, `pytorch.log_model` and
`transformers.log_model` wrappers with mandatory `input_example` requests.
They reuse trained artifacts already present in the sibling generator repository.
They do not train, download models, import training scripts, read benchmark request
fixtures or expected answers, or start Triton. No LLM is involved.

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

## Run all twelve with isolated dependencies

Docker must be running. From any working directory:

```bash
python3 /home/ghost/git/mlflow-plugin/examples/run_real_models.py \
  --data-root /home/ghost/git/triton-model-generator/benchmarks/data \
  --output /home/ghost/git/triton-model-generator/generated/mlflow-real-examples-12
```

The launcher pulls Python container images as needed and installs pinned inference
libraries from the saved artifacts plus MLflow 3.17.2. It uses Python 3.11 for
sklearn/Hugging Face and Python 3.14 for the saved PyTorch custom classes (their
producer Python is 3.14). Package/image downloads require network access; model
loading is offline. No training or live Triton service is started.

There are three temporary CPU containers with four cases each. Source artifacts
and plugin code are mounted read-only. Outputs belong to your host user. The
output path must be new. Follow progress in `sklearn.log`, `pytorch.log`, and
`transformers.log` inside that directory. The combined `summary.json` contains
all twelve outcomes, including installation failures and timeouts. Per-family
SQLite tracking stores and logged packages remain in the corresponding folders.
Use `--family sklearn` (repeatable) for a subset, or `--timeout 7200` to increase
the default one-hour limit per family. A nonzero exit means at least one case
did not pass. The launcher itself was checked with simulated Docker success,
installation failure and timeout; actual container execution is left to you.

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
  --data-root /home/ghost/git/triton-model-generator/benchmarks/data \
  --output /tmp/mlflow-real-examples-12

# Or select one example, with a new output directory:
python examples/real_models.py \
  --data-root /home/ghost/git/triton-model-generator/benchmarks/data \
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
The complete 12-example logging run and actual Triton execution remain unrun.
