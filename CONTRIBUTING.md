# Contributing

Bug reports, focused fixes, documentation improvements and new framework examples
are welcome. Use issues for reproducible problems and feature proposals; use pull
requests for reviewable changes. Keep discussions respectful and technical.

## Development setup

Use Python 3.11+ for repository tooling (the packages declare Python 3.10+).
Start from a virtual environment and install core before the optional integration:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e packages/mlflow-triton
pip install -e packages/mlflow-triton-control
pip install ruff build twine
```

Install PyTorch/Transformers separately to run their optional tests. Release CI
requires a CPU framework job with PyTorch, Transformers and tritonclient installed;
`--fail-on-skip` makes skipped tests fail that job. The custom HF
unit test uses small synthetic weights and local architecture source; full example
weights are only needed by the serving runner. Git LFS is required for those weights.

## Checks

```bash
ruff check packages tests scripts
ruff format --check packages tests scripts
python scripts/check_release.py
python scripts/test_packages.py --package all
```

For installed-package isolation checks, `bash smoke-latest.sh` builds both packages,
tests core alone, then tests the optional integration. It creates a temporary
virtual environment and downloads dependencies. Set `TMPDIR` and `PIP_CACHE_DIR`
to existing directories on a suitable disk if necessary.

Unit tests do not deploy services. Real CPU/GPU serving tests are explicit runs of
[the example runner](examples/README.md); configure `--output-root` for all results.
Report local tests and actual Triton checks separately, including skipped tests.

## Repository layout

- `packages/mlflow-triton/`: logging, signatures and generated runtime.
- `packages/mlflow-triton-control/`: optional deployment client and legacy imports.
- `examples/`: serving runner and the 13-case saved-model dataset.
- `tests/`: core, deployment and compatibility tests.
- `scripts/`: package-test selection and release checks.

Keep core independent of the Control API and its credentials. Preserve native
preprocessing and weights. Training sources are evidence: never import or execute
them to manufacture a serving artifact. Do not tune implementations to individual
reference answers or silently weaken comparisons.

## Pull requests

Explain the problem, resulting behavior, relevant tests and remaining limitations.
Update the API tables when arguments change. Include provenance and license evidence
for new third-party code or model assets; do not add large binaries without discussing
their purpose and Git LFS storage. Never commit credentials or generated run output.
Contributions to original project code are made under the existing Apache-2.0 license.

See [release instructions](RELEASING.md) for the shared version policy. This guide
describes the contribution workflow, not the current GitHub branch-protection settings.
