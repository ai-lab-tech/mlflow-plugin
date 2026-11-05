# mlflow-triton-control

Optional MLflow deployment integration for Triton Control. Installing this package
automatically installs the matching `mlflow-triton` core release.

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

### External model endpoints

The default is cluster-only access. External access uses the same REST API and
administrator policy as Triton Control's deployment UI. The plugin reads
`GET /api/deployments/policy` before creating an Ingress deployment.
The example below uses the short hostname `iris`. When a base domain is
configured, the backend appends it; read the full URL from the returned record.

```python
deployment = client.create_deployment(
    "iris",
    model_uri="models:/your-model-id",
    config={
        "s3_profile_name": "training",
        "image": "your-triton-image",
        "access_mode": "ingress",
        "ingress_host": "iris",
        "ingress_scheme": "https",
    },
)
print(deployment["external_url"])
```

| Config key | Default | Meaning |
|---|---|---|
| `access_mode` | `internal`; inferred as `ingress` when a host is supplied | `internal` or `ingress` |
| `ingress_host` | unset | Short hostname only, e.g. `smollm`; no domain, protocol, port or path |
| `ingress_scheme` | `https` | `https` or `http`, separate from the hostname |
| `basic_auth_enabled` | `false` | Boolean or CLI string `true` / `false` |
| `basic_auth_username` | unset | Required when basic authentication is enabled |
| `basic_auth_password` | unset | Required when basic authentication is enabled; write-only |

HTTP requires administrator permission. Basic authentication requires HTTPS.
Pass passwords through your application's secret handling, not literal shell
arguments or committed configuration. Controller class and certificate issuer
are administrator-managed; `ingress_class_name` is rejected.

The backend checks policy again on creation. Existing endpoints are not removed
when an administrator disables Ingress. Update access through Triton Control's
**Endpoints** tab; MLflow `update_deployment` remains unsupported.

Deployment lookup results preserve `url` as the internal connection for
compatibility and add `internal_url`, `external_url`, `access` and
`access_error`. `external_url` is null for cluster-only access. If live
Kubernetes exposure cannot be read, `access_error` explains that the external
URL is unknown; readiness still uses the internal connection.

Always supply only the short hostname (for example, `ingress_host="smollm"`).
Domains are administrator-managed through `tritonDeployments.baseDomain`.
The REST API appends the configured domain for the Ingress and certificate, and `external_url` in
deployment lookups contains the full endpoint. The plugin never appends the
domain itself. Full domains are rejected even when the base domain is empty;
with an empty base domain, the short hostname is used unchanged.
