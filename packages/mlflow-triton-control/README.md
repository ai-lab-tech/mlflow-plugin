# mlflow-triton-control

Optional MLflow deployment integration for Triton Control. Installing this package
automatically installs the matching `mlflow-triton` core release.

```bash
pip install mlflow-triton-control
mlflow deployments help -t triton-control
```

```bash
mlflow deployments create --target triton-control \
  --name demo --model-uri models:/m-your-logged-model-id \
  -C s3_profile_name=workflow-training -C image=your-triton-image \
  -C wait_timeout_seconds=600
```

```python
from mlflow.deployments import get_deploy_client

client = get_deploy_client("triton-control")
deployment = client.create_deployment(
    name="iris-classifier",
    model_uri=info.model_uri,
    config={
        "s3_profile_name": "workflow-training",
        "image": "your-triton-image",
        "wait_timeout_seconds": 600,
    },
)
```

## Deployment arguments

The MLflow CLI and Python client use the same deployment implementation.

| CLI argument | Python argument | Required / default | Meaning |
| --- | --- | --- | --- |
| `--target`, `-t` | `get_deploy_client(target_uri)` | Required | Plugin address for Plugin selection. |
| `--name` | `name` | Required | New deployment name: 1–63 lowercase letters, digits or hyphens. Must start and end with a letter or digit. Example: `iris-classifier`.|
| `--model-uri` | `model_uri` | Required | MLflow model reference. |
| Repeated `-C key=value` | `config` | Required configuration | Deployment settings described in the next table. |

### Configuration (`-C` / `config`)

| Key | Required / default | Accepted value | Meaning |
| --- | --- | --- | --- |
| `s3_profile_name` | Required | Non-empty string | Exact name of a profile owned by the authenticated caller. |
| `image` | Required | Non-empty container image reference | Triton serving image |
| `wait_timeout_seconds` | Optional / `300` | Finite positive number of seconds | How long to poll for readiness after the create request succeeds. The client checks approximately every three seconds and raises `MlflowException` on timeout. A timeout does not delete the deployment. |

 The requirements file for serving is read automatically from the artifact.


## Model URI and artifact storage

Supported model references:

| URI | Meaning |
| --- | --- |
| `models:/m-...` | Logged model ID returned by the framework wrappers |
| `models:/iris-serving/3` | Registered model version |
| `models:/iris-serving@champion` | Registry alias resolved once at deployment time |
| `runs:/<run-id>/<model-path>` | MLflow model stored as a run artifact |


## Other deployment operations

| Python method | Arguments | Result |
| --- | --- | --- |
| `list_deployments()` | None | List of deployment API records, each with `flavor="triton"`. |
| `get_deployment(name)` | Deployment name | Deployment API record with `flavor="triton"`. |
| `delete_deployment(name)` | Deployment name | Triton Control's deletion response. |

`update_deployment()`, `predict()` and `run_local()` are not supported and raise
`MlflowException`. Send inference requests to the deployed Triton server directly.

## Compatibility

Existing imports such as `from mlflow_triton_control import sklearn` still
work when this integration is installed. These are aliases to the core modules.

## How MLflow finds the plugin

`mlflow deployments create` is a command provided by MLflow. This package
registers the deployment target through its Python package metadata:

```toml
[project.entry-points."mlflow.deployments"]
triton-control = "mlflow_triton_control.deployment_client"
```

For `--target triton-control`, MLflow looks up the
`triton-control` entry point among installed packages, loads this module, and
calls `TritonControlDeploymentClient.create_deployment()`. That class inherits
from MLflow's `BaseDeploymentClient` and sends the deployment request to the
Triton Control API. MLflow reads entry-point metadata; it does not search
through every package's source files. The package must be installed in the
same Python environment as the `mlflow` CLI.


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
`TRITON_CONTROL_TOKEN`, `TRITON_CONTROL_S3_PROFILE_NAME`, and
`TRITON_CONTROL_S3_BUCKET` into the named `script` or `container` template.
Use the name with `-C s3_profile_name="$TRITON_CONTROL_S3_PROFILE_NAME"`.
The scoped token can resolve only that profile; it cannot list profiles or read
their credentials. The backend keeps profile IDs internal.
The token is valid for 60 minutes and limited
to the specified deployment and S3 profile. The Workflow receives a five-minute
completion TTL unless it already defines one, and the Secret receives a
Workflow owner reference for garbage collection.

The opted-in Workflow must have a fixed `metadata.name` and run in the Triton
Control namespace. Direct submissions to Argo bypass this token injection.

See the [shared release process](https://github.com/ai-lab-tech/mlflow-plugin/blob/main/RELEASING.md).
