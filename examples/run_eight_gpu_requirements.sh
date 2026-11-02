#!/usr/bin/env bash
# Prepare/log saved models, then test each generated repository independently.
set -euo pipefail

task_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$task_repo_root/examples/run_real_models.py" \
  --device gpu \
  --requirements-only \
  --case pytorch_polynomial \
  --case pytorch_custom_network \
  --case pytorch_mnist_lstm \
  --case pytorch_dcgan \
  --case huggingface_classification \
  --case huggingface_ner \
  --case huggingface_question_answering \
  --case huggingface_generation \
  "$@"
