# mlflow-triton-control

MLflow deployment target for models that already exist in a Triton S3 model
repository. Install this Python package in the environment running the MLflow
CLI or Python API. Triton Control creates the Kubernetes deployment and resolves
the selected user-owned S3 profile.

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
python -m pip install hypothesis-jsonschema jsonschema
python -m pip check
python -m unittest discover -s tests -v
```

The smoke tests verify installed plugin discovery, deployment requests with a
mock HTTP transport, and real model logging, registration, and retrieval with
a temporary SQLite tracking database.

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

## Log training artifacts

Capture the training source and installed Python packages in an active MLflow
run, before a Triton repository exists:

```python
import mlflow
from mlflow_triton_control import triton

with mlflow.start_run():
    artifacts = triton.log_training_artifacts(training_code_path="/workspace/source")
    # Start training here.
```

The plugin runs `pip freeze` with the current Python interpreter automatically.
It logs the source directory recursively under `training-artifacts/code/` and the
captured environment under `training-artifacts/requirements.txt`. Keep training
data outside the source directory. The source directory is not modified.
These requirements describe the installed training environment; they are not
minimized for serving.

Use `artifact_path` to change the default `training-artifacts` location. The return
value contains run-relative paths under `code` and `requirements`. No Triton
repository or Model Registry entry is created by this function. Once a repository
has been generated, use `triton.log_model()` to log it.

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

`triton.log_model()` automatically sets the `triton_model_uri` version tag to
the concrete S3 repository URI when registering a model in an S3-backed artifact
store. It still returns MLflow's `ModelInfo`; use `model_info.model_uri` directly
with the deployment client. Local artifact stores remain supported for logging
and loading, but S3-backed storage is required for deployment.

The deployment client accepts `models:/<model-id>`, registered model references
such as `models:/iris-serving/1` or `models:/iris-serving@candidate`, `runs:/`
model references, and existing `s3://` repository URIs. MLflow references must
have the `triton` flavor. The plugin resolves their artifact location and the
flavor's model directory before calling Triton Control. Proxy artifact storage
is resolved using Triton Control's MLflow artifact-store endpoint and the current
MLflow tracking credentials.

The selected S3 profile must have read access to the MLflow artifact bucket and
endpoint. This uses the repository already stored by MLflow; it does not copy it
to a different bucket or profile prefix.

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

## S3 deployment of logged models (0.3.1)

The Triton flavor stores data under `model/<original-model-name>/` inside the
MLflow Model. Pass `ModelInfo.model_uri` directly to the deployment plugin;
S3-backed registered versions receive the concrete repository URI in their
`triton_model_uri` tag automatically. The selected S3 profile must be able to
read the artifact bucket. Existing models using `data: model` remain loadable.
See the Iris auto-deployment workflow for the complete example.

## Publishing a wheel to PyPI

Run from the `mlflow-plugin` repository root. Set the new release version in
`pyproject.toml` first; the commands below use `0.3.1`. Use a fresh output directory
for every release so that older distributions are not uploaded accidentally.

Create a release environment, build the wheel and source distribution, and check
their metadata and README rendering with a current Twine version:

```bash
python3 -m venv /tmp/mlflow-release-tools
. /tmp/mlflow-release-tools/bin/activate
python -m pip install --upgrade build twine
python -m build --outdir dist/mlflow-triton-control-0.3.1
python -m twine check dist/mlflow-triton-control-0.3.1/*
```

Create a PyPI API token under [PyPI account settings](https://pypi.org/manage/account/),
scoped to `mlflow-triton-control`. Upload the wheel to the official PyPI registry:

```bash
python -m twine upload \
  --repository-url https://upload.pypi.org/legacy/ \
  --username __token__ \
  dist/mlflow-triton-control-0.3.1/*.whl
```

When prompted for the password/API token, paste your PyPI API token. The terminal
hides the input. Do not put the token in source files or commit it to Git.
Twine needs distribution **files**, not a directory: keep `*.whl` at the end of
the path. To also publish the source distribution, upload it separately:

```bash
python -m twine upload \
  --repository-url https://upload.pypi.org/legacy/ \
  --username __token__ \
  dist/mlflow-triton-control-0.3.1/*.tar.gz
```

If the build files are already in another directory, use that directory's
`*.whl` or `*.tar.gz` path instead; rebuilding is not required. PyPI does not allow
replacing an uploaded release file. Publish subsequent changes with a new version.

After publishing, install the release into your training/deployment environment:

```bash
python -m pip install --upgrade "mlflow-triton-control==0.3.1"
mlflow deployments help -t triton-control
```

The help command confirms that MLflow can discover the installed plugin; it does
not create a deployment.

## Contract tests

CI checks out `triton-control/main` and exports the deployment router's OpenAPI
contract without starting the server or its lifespan. Backend dependencies are
installed in a separate environment. Plugin HTTP requests are validated against
those routes and request schemas; POST and DELETE mock responses are generated
from their current response schemas using hypothesis-jsonschema.

The GET endpoints currently declare free-form dictionaries, so OpenAPI cannot
describe their fields. For those responses the exporter calls the upstream
`describe` serializer with a synthetic model record and simulated Triton
readiness. The response fields and readiness values come from backend code,
not a duplicated response dictionary in this repository.

To run locally, install the backend dependencies in a separate virtual environment,
then export and run (replace the backend Python and checkout paths):

```bash
DATABASE_URL=sqlite:// PYTHONPATH=../triton-control/triton-backend \
  /path/to/backend-venv/bin/python tests/export_contract.py /tmp/triton-contract.json
python -m pip install hypothesis-jsonschema jsonschema
TRITON_CONTROL_CONTRACT=/tmp/triton-contract.json python -m unittest discover -s tests -v
```

The contract test is skipped when `TRITON_CONTROL_CONTRACT` is unset. CI always
sets it; missing or invalid exports fail the job. These tests exercise API shapes
and plugin behavior, not backend authentication or real deployment execution.
