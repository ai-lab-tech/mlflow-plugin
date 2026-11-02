# Security policy

## Report privately

Do not disclose vulnerabilities, credentials or private artifacts in public issues.
Use [GitHub private vulnerability reporting](https://github.com/ai-lab-tech/mlflow-plugin/security/advisories/new)
when enabled. If that route is unavailable, contact a repository maintainer privately
through their GitHub profile before sharing details. There is no guaranteed response SLA.

Include the package/version, affected workflow, impact and a minimal reproduction
with secrets removed. Fixes target the current development line and latest release;
there is no promise of security backports to older releases.

## Trust boundaries

Only load artifacts and custom Python code you trust. Pickle, cloudpickle, joblib,
PyTorch serialization and custom Hugging Face code can execute code while loading.
Logging runs saved-model prediction locally; it is not a sandbox for untrusted models.

Input examples are saved in MLflow artifacts. Use non-sensitive examples and keep
credentials out of code paths, model metadata and logs. Install dependencies in an
isolated serving environment. The optional Control integration uses its own API
authentication; follow its deployment documentation for network and token handling.
