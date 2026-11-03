# Releases

Both distributions use one version and one Git tag. Installations select the
packages, not a runtime activation flag. The Control distribution depends on the
exact matching core version; existing Control logging imports forward to core.

## Prepare

Use Python 3.11+ for release tooling. Update `VERSION`, both package
`pyproject.toml` versions and the Control dependency `mlflow-triton==X.Y.Z`.
The README installation commands use the latest published version and do not
need a version edit for each release. Update them if package names, Python
requirements or installation steps change. Then run:

```bash
python scripts/check_release.py --tag v0.4.2
python -m build packages/mlflow-triton --outdir dist/core
python -m build packages/mlflow-triton-control --outdir dist/control
python -m twine check --strict dist/core/* dist/control/*
# Temporary files and pip cache can be placed on another disk:
TMPDIR=/mnt/container-data/tmp PIP_CACHE_DIR=/mnt/container-data/pip-cache bash smoke-latest.sh
```

Create the chosen temporary/cache directories first. `smoke-latest.sh` builds
both packages, installs core alone, runs its tests, then installs and tests the
Control integration. It accepts `latest` (default) or an MLflow version. Tests
mock deployment requests; actual Triton serving remains an explicit example run.
Optional PyTorch/Hugging Face tests need their framework dependencies and may be
skipped in the lightweight packaging environment.

## Publish

Before the first release, confirm ownership/availability of both PyPI project
names and configure separate Trusted Publishers for `mlflow-triton` and
`mlflow-triton-control`: repository `ai-lab-tech/mlflow-plugin`, workflow
`mlflow-plugin-ci.yml`, environments `pypi-core` and `pypi-control` respectively.
Create those GitHub environments and configure any desired approval rules.

After review, commit the release and push a matching `vX.Y.Z` tag. CI checks the
shared version, builds both wheels and source distributions, and tests MLflow
3.14.0 plus the latest supported 3.x release. Only tag pushes publish; core is
published before Control. Branch/PR/scheduled runs never publish. Each package
has its own README as its PyPI description. After both publications succeed, CI
creates a GitHub release with generated notes and all four distribution files
attached (a wheel and source archive for each package). The release job does not
run if either publication fails. No PyPI API token is needed: both publishers
use GitHub OIDC. This workflow does not use a `PUBLISH_PYPI` variable; matching
tag pushes trigger publishing through the two environments above.

After both packages are published, verify the root README's PyPI installation
steps in a fresh virtual environment: install core alone, then add Control and
run `python -m pip check` and `mlflow deployments help -t triton-control`. Confirm
that both installed versions match the release. Keep the source-installation
instructions available for unreleased changes.

If core publishes but Control fails, fix the external publishing problem and
rerun failed jobs. Do not replace an already published version. Code fixes require
a new shared version and tag. Creating these files does not publish a release or
configure the external Trusted Publishers.

## Public repository preparation

The repository files do not configure GitHub settings. Before the first public
release, maintainers should enable private vulnerability reporting, select branch
protection/review rules, and configure the PyPI environments described above.
Resolve the specific outstanding dataset redistribution review in
[third-party notices](THIRD_PARTY_NOTICES.md), including the existing Git/LFS history.
Review history for credentials before changing visibility; ignore patterns only
prevent future accidental additions and do not scrub previous commits.
