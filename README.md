# MLflow → Triton

Two separately installable packages in one repository. Data Scientists can log
saved models and demonstrate their predictions in Triton. ML Engineers receive
the generated serving structure as a starting point; this is not a production
readiness or performance guarantee.

| Package | Install | Responsibility |
| --- | --- | --- |
| [mlflow-triton](packages/mlflow-triton/README.md) | `pip install mlflow-triton` | Logging wrappers, signatures, CPU/GPU config and complete Triton repositories |
| [mlflow-triton-control](packages/mlflow-triton-control/README.md) | `pip install mlflow-triton-control` | Optional deployment through Triton Control; installs the matching core |

```python
from mlflow_triton import sklearn

info = sklearn.log_model(fitted_pipeline, input_example=example_dataframe)
```

The core requires no Triton Control service, token or configuration. Installing
the Control package registers the optional MLflow deployment target; no activation
flag is needed. Existing `mlflow_triton_control` logging imports remain compatible
when the integration is installed.

## Repository layout

```text
packages/mlflow-triton/          # Core package and its PyPI README
packages/mlflow-triton-control/  # Deployment package and its PyPI README
examples/                       # Shared runners and Git-LFS test artifacts
tests/                          # Core, integration and compatibility tests
scripts/                        # Build/release validation and test selection
VERSION                         # Shared release version
```

## Develop and test

```bash
pip install -e packages/mlflow-triton
# Optional integration; install core first when developing an unreleased version:
pip install -e packages/mlflow-triton-control
python scripts/test_packages.py --package core
python scripts/test_packages.py --package control
```

[Example documentation](examples/README.md) covers the saved models, custom
architectures, CPU/GPU runs and configurable storage. Model binaries use Git LFS.
The runners use core code only. Tests do not start live Triton or deploy services;
actual serving validation is explicitly run through the examples.

## Releases

Both packages share version `0.4.0` and the tag `v0.4.0`. CI builds two independent
wheels/sdists, tests the core without the Control package, then tests the optional
integration. Supported MLflow versions are tested independently of Triton Control's
Helm chart. See [release instructions](RELEASING.md) for version checks, tag-based
publishing and required PyPI Trusted Publisher setup.
