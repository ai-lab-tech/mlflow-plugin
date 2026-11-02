# mlflow-triton-control

Optional MLflow deployment integration for Triton Control. Installing this package
automatically installs the matching `mlflow-triton` core release.

```bash
pip install mlflow-triton-control
mlflow deployments help -t triton-control
```

For logging only, install the [core package](https://github.com/ai-lab-tech/mlflow-plugin/blob/main/packages/mlflow-triton/README.md).
The Control integration requires a reachable Triton Control API, a user-bound
`TRITON_CONTROL_TOKEN`, an owned S3 profile, a published Triton model repository,
and a suitable Triton runtime image. Keep credentials out of model artifacts.

```bash
mlflow deployments create --target triton-control://localhost:8000 \
  --name demo --model-uri s3://models/repository/demo \
  --flavor triton -C s3_profile_id=7 -C image=your-triton-image
```

The existing client supports create, list, get and delete. It accepts concrete
`s3://` model-directory URIs; it does not upload `models:/` or `runs:/` artifacts,
serve locally, update deployments or provide prediction. The current API adapter
uses HTTP; use it in the same trusted network as the Control service.

## Compatibility

Existing imports such as `from mlflow_triton_control import sklearn, triton` still
work when this integration is installed. These are aliases to the core modules;
new applications should use `from mlflow_triton import sklearn, triton`.
The deployment class and `triton-control://` target retain their existing names.
No model artifacts are rewritten by this split.

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

## S3 deployment of logged models

The Triton flavor now stores data under `model/<original-model-name>/` inside
the MLflow Model. Pass the concrete S3 URI of that subdirectory to the deployment
plugin, using an S3 profile that can read the artifact bucket. Resolve proxy
artifact locations against the managed server's S3 root; the deployment client
continues to accept only S3 URIs. Existing models using `data: model` remain
loadable. 


See the [shared release process](https://github.com/ai-lab-tech/mlflow-plugin/blob/main/RELEASING.md).
