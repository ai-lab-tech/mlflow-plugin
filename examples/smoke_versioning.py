"""Train and register real Iris models; retain the registry and verification report.

Run: python examples/smoke_versioning.py --output /tmp/triton-versioning-smoke
The output directory must not already exist.
"""

import argparse
import hashlib
import json
from pathlib import Path

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from mlflow_triton import sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


def tree_digest(path):
    return {
        file.relative_to(path).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def check(condition, message):
    """Keep smoke checks active even when Python runs with optimization enabled."""
    if not condition:
        raise AssertionError(message)


def run_smoke(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    previous_tracking = mlflow.get_tracking_uri()
    previous_registry = mlflow.get_registry_uri()
    tracking_uri = f"sqlite:///{output / 'mlflow.db'}"
    report = {
        "status": "running",
        "tracking_uri": tracking_uri,
        "mlflow_version": mlflow.__version__,
        "registered_model_name": "iris-versioning-smoke",
        "versions": [],
    }
    try:
        # Arrange: train models with new weights, then a changed input schema.
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        experiment = mlflow.create_experiment(
            "triton-versioning-smoke", artifact_location=(output / "artifacts").as_uri()
        )
        client = MlflowClient()
        inputs, targets = load_iris(return_X_y=True, as_frame=True)
        cases = [
            ("original", inputs, 1.0, ["1"]),
            ("compatible", inputs, 0.05, ["2"]),
            ("incompatible", inputs.iloc[:, :3], 0.05, ["1"]),
        ]
        models = [
            LogisticRegression(C=regularization, max_iter=300).fit(features, targets)
            for _, features, regularization, _ in cases
        ]
        infos = []

        original_digest = None
        # Act: log each model through the public API into the same registry entry.
        for (case, features, _, _), model in zip(cases, models):
            with mlflow.start_run(experiment_id=experiment, run_name=case):
                infos.append(
                    sklearn.log_model(
                        model,
                        name="iris",
                        registered_model_name="iris-versioning-smoke",
                        input_example=features.iloc[:5],
                        pyfunc_predict_fn="predict_proba",
                    )
                )
            if len(infos) == 1:
                first = Path(
                    mlflow.artifacts.download_artifacts(
                        artifact_uri="models:/iris-versioning-smoke/1"
                    )
                )
                original_digest = tree_digest(first)

        # Assert: read registered artifacts independently of the logging results.
        repositories, rows = [], report["versions"]
        for index, (case, features, regularization, expected_versions) in enumerate(
            cases, 1
        ):
            model, info = models[index - 1], infos[index - 1]
            uri = f"models:/iris-versioning-smoke/{index}"
            registered = client.get_model_version("iris-versioning-smoke", str(index))
            check(registered.status == "READY", f"{uri}: status is {registered.status}")
            check(
                str(info.registered_model_version) == str(index),
                f"{uri}: wrong Registry version",
            )
            metadata = mlflow.models.get_model_info(uri)
            flavor = metadata.flavors["triton"]
            package = Path(mlflow.artifacts.download_artifacts(artifact_uri=uri))
            repository = package / flavor["data"]
            repositories.append(repository)
            versions = sorted(p.name for p in repository.iterdir() if p.is_dir())
            check(
                versions == expected_versions,
                f"{case}: expected {expected_versions}, found {versions}",
            )
            check(
                flavor["model_version"] == int(expected_versions[-1]),
                f"{case}: incorrect Triton flavor version",
            )
            example = features.iloc[20:40]
            expected = model.predict_proba(example)
            np.testing.assert_allclose(
                mlflow.pyfunc.load_model(uri).predict(example), expected
            )
            for version in versions:
                check(
                    (repository / version / "model.py").is_file(),
                    f"{case}: missing backend in Triton version {version}",
                )
                native = mlflow.pyfunc.load_model(
                    str(repository / version / "artifact")
                )
                np.testing.assert_allclose(native.predict(example), expected)
            rows.append(
                {
                    "case": case,
                    "registry_version": str(registered.version),
                    "registry_status": registered.status,
                    "model_uri": uri,
                    "source": registered.source,
                    "repository": str(repository),
                    "triton_versions": versions,
                    "repository_group": flavor["repository_group"],
                    "config_sha256": hashlib.sha256(
                        (repository / "config.pbtxt").read_bytes()
                    ).hexdigest(),
                    "predictions": "passed",
                }
            )
        check(
            not np.array_equal(models[0].coef_, models[1].coef_), "Weights must change"
        )
        check(
            rows[0]["config_sha256"] == rows[1]["config_sha256"],
            "Compatible config changed",
        )
        check(
            rows[1]["config_sha256"] != rows[2]["config_sha256"],
            "Incompatible config did not change",
        )
        check(
            rows[0]["repository_group"] == rows[1]["repository_group"],
            "Compatible group changed",
        )
        check(
            rows[1]["repository_group"] != rows[2]["repository_group"],
            "Incompatible group did not change",
        )
        check(
            tree_digest(repositories[0].parents[1]) == original_digest,
            "Original artifact files changed",
        )
        check(not (repositories[0] / "2").exists(), "Original artifact was modified")
        registered_versions = client.search_model_versions(
            "name = 'iris-versioning-smoke'"
        )
        check(
            {str(v.version) for v in registered_versions} == {"1", "2", "3"},
            "Unexpected Registry versions",
        )
        report.update(
            status="passed",
            checks=[
                "changed_weights",
                "config_equality",
                "config_change",
                "only_current_version_packaged",
                "original_unchanged",
                "native_and_packaged_predictions",
                "three_ready_registry_versions",
            ],
        )
        return report
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        mlflow.set_tracking_uri(previous_tracking)
        mlflow.set_registry_uri(previous_registry)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.output), indent=2))
