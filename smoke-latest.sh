#!/usr/bin/env bash
# Build both packages, test core alone, then test optional Control integration.
set -euo pipefail
if (( $# > 1 )); then
    echo "Usage: bash smoke-latest.sh [latest|VERSION]" >&2
    exit 2
fi
mlflow_version="${1:-latest}"
if [[ "$mlflow_version" = latest ]]; then
    mlflow_requirement='mlflow>=3.14,<4'
elif [[ "$mlflow_version" =~ ^[0-9]+(\.[0-9]+){1,3}([a-zA-Z0-9.+-]*)?$ ]]; then
    mlflow_requirement="mlflow==$mlflow_version"
else
    echo "Expected latest or an MLflow version" >&2
    exit 2
fi
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
smoke_dir="$(mktemp -d)"
trap 'rm -rf -- "$smoke_dir"' EXIT
"${PYTHON:-python3}" -m venv "$smoke_dir/venv"
smoke_python="$smoke_dir/venv/bin/python"
"$smoke_python" -m pip install --upgrade pip build
"$smoke_python" "$repo_dir/scripts/check_release.py"
for package in mlflow-triton mlflow-triton-control; do
    "$smoke_python" -m build "$repo_dir/packages/$package" --outdir "$smoke_dir/dist/$package"
done
"$smoke_python" -m pip install "$smoke_dir"/dist/mlflow-triton/*.whl "$mlflow_requirement"
export MLFLOW_DISABLE_AGENT_HINT=1
cd "$smoke_dir"
"$smoke_python" -m pip check
"$smoke_python" "$repo_dir/scripts/test_packages.py" --package core --require-core-only
"$smoke_python" -m pip install "$smoke_dir"/dist/mlflow-triton-control/*.whl
"$smoke_python" -m pip check
"$smoke_dir/venv/bin/mlflow" deployments help -t triton-control
"$smoke_python" "$repo_dir/scripts/test_packages.py" --package control
