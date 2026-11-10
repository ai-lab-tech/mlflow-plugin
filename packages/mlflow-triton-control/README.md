# mlflow-triton-control

Optional MLflow deployment integration for Triton Control. Installing this package
automatically installs the matching `mlflow-triton` core release.

Install in the same Python environment as your `mlflow` command.

```bash
pip install mlflow-triton-control
mlflow deployments help -t triton-control
```

Run these commands in your Triton Control workspace or workflow. Connection
and authentication are configured automatically. Use a model logged to the
workspace's MLflow server and the name of your S3 profile.

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
        # Optional external HTTPS endpoint:
        # "ingress_host": "iris",
    },
)
```

## Deployment arguments

The MLflow CLI and Python client use the same deployment implementation.

| CLI argument | Python argument | Required / default | Meaning |
| --- | --- | --- | --- |
| `--target`, `-t` | `get_deploy_client(target_uri)` | Required | Use `triton-control`. |
| `--name` | `name` | Required | New deployment name: 1–63 lowercase letters, digits or hyphens. Must start and end with a letter or digit. Example: `iris-classifier`.|
| `--model-uri` | `model_uri` | Required | MLflow model reference. |
| Repeated `-C key=value` | `config` | Required configuration | Deployment settings described in the next table. |

### Configuration (`-C` / `config`)

| Key | Required / default | Accepted value | Meaning |
| --- | --- | --- | --- |
| `s3_profile_name` | Required | Non-empty string | Exact name of a profile owned by the authenticated caller. |
| `image` | Required | Non-empty container image reference | Triton serving image |
| `wait_timeout_seconds` | Optional / `300` | Finite positive number of seconds | How long to poll for readiness after the create request succeeds. The client checks approximately every three seconds and raises `MlflowException` on timeout. A timeout does not delete the deployment. |
| `access_mode` | Optional / `internal`; `ingress` when a host is supplied | `internal` or `ingress` | Internal or external access. |
| `ingress_host` | Optional / unset | Short hostname, e.g. `iris` | External endpoint name. No domain, protocol, port or path; Triton Control adds the configured domain. |
| `ingress_scheme` | Optional / `https` | `https` or `http` | HTTP requires administrator permission. |
| `basic_auth_enabled` | Optional / `false` | Boolean or CLI string `true` / `false` | Enable basic authentication; requires HTTPS. |
| `basic_auth_username` | Required with basic authentication | String | Endpoint username. |
| `basic_auth_password` | Required with basic authentication | String | Endpoint password; supply from your secret store. |

The requirements file for serving is read automatically from the artifact.
Deployments are internal by default. External URLs are returned in
`deployment["external_url"]`. To change access later, use **Endpoints** in Triton Control.


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

## Argo Workflows

Link your S3 profile under **Workflows → Configure S3 Secrets**, then submit
through Triton Control. Connection and authentication are configured automatically.

If you have multiple linked profiles, select one in your workflow:

```yaml
metadata:
  name: train-iris
  annotations:
    triton-control.ai/mlflow-s3-profile-name: workflow-training
```

After logging the model, deploy it from your workflow step:

```bash
mlflow deployments create --target triton-control \
  --name iris-classifier --model-uri "$MODEL_URI" \
  -C s3_profile_name="$TRITON_CONTROL_S3_PROFILE_NAME" \
  -C image=your-triton-image
```

Set `MODEL_URI` to the logged model's `info.model_uri` and replace
`your-triton-image` with your serving image.

Serving requirements are read from `<model_version>/requirements.txt` (version 1
when legacy metadata omits the number). Older artifacts with requirements next to
`config.pbtxt` remain supported as a fallback.

### Shared Triton versions

Compatible logged/registered models use the first model's Triton repository as
a shared serving location. The deployment client follows the Triton flavor's
`repository_model_id`, so Registry versions 1 and 2 deploy from the same
repository containing `1/` and `2/`. Their native MLflow packages remain separate.
The requested snapshot's version-local requirements are read from the shared
repository. A configuration change starts a new repository at `1/`.
