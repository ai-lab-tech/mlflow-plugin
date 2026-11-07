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


def run_smoke(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    previous_tracking = mlflow.get_tracking_uri()
    previous_registry = mlflow.get_registry_uri()
    tracking_uri = f"sqlite:///{output / 'mlflow.db'}"
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        experiment = mlflow.create_experiment(
            "triton-versioning-smoke", artifact_location=(output / "artifacts").as_uri()
        )
        client = MlflowClient()
        inputs, targets = load_iris(return_X_y=True, as_frame=True)
        cases = [
            ("original", inputs, 1.0, ["1"]),
            ("compatible", inputs, 0.05, ["1", "2"]),
            ("incompatible", inputs.iloc[:, :3], 0.05, ["1"]),
        ]
        models, repositories, rows = [], [], []
        for index, (case, features, regularization, expected_versions) in enumerate(
            cases, 1
        ):
            model = LogisticRegression(C=regularization, max_iter=300).fit(
                features, targets
            )
            models.append(model)
            with mlflow.start_run(experiment_id=experiment, run_name=case):
                info = sklearn.log_model(
                    model,
                    name="iris",
                    registered_model_name="iris-versioning-smoke",
                    input_example=features.iloc[:5],
                    pyfunc_predict_fn="predict_proba",
                )
            uri = f"models:/iris-versioning-smoke/{index}"
            registered = client.get_model_version("iris-versioning-smoke", str(index))
            assert registered.status == "READY", registered.status
            assert str(info.registered_model_version) == str(index)
            metadata = mlflow.models.get_model_info(uri)
            flavor = metadata.flavors["triton"]
            package = Path(mlflow.artifacts.download_artifacts(artifact_uri=uri))
            repository = package / flavor["data"]
            repositories.append(repository)
            versions = sorted(p.name for p in repository.iterdir() if p.is_dir())
            assert versions == expected_versions, (case, versions)
            assert flavor["model_version"] == int(expected_versions[-1])
            example = features.iloc[20:40]
            expected = model.predict_proba(example)
            np.testing.assert_allclose(
                mlflow.pyfunc.load_model(uri).predict(example), expected
            )
            for version in versions:
                assert (repository / version / "model.py").is_file()
                native = mlflow.pyfunc.load_model(
                    str(repository / version / "artifact")
                )
                trained = models[int(version) - 1] if index == 2 else model
                np.testing.assert_allclose(
                    native.predict(example), trained.predict_proba(example)
                )
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
        assert not np.array_equal(models[0].coef_, models[1].coef_), (
            "Weights must change"
        )
        assert rows[0]["config_sha256"] == rows[1]["config_sha256"]
        assert rows[1]["config_sha256"] != rows[2]["config_sha256"]
        assert rows[0]["repository_group"] == rows[1]["repository_group"]
        assert rows[1]["repository_group"] != rows[2]["repository_group"]
        assert tree_digest(repositories[0] / "1") == tree_digest(repositories[1] / "1")
        assert not (repositories[0] / "2").exists(), "Original artifact was modified"
        registered_versions = client.search_model_versions(
            "name = 'iris-versioning-smoke'"
        )
        assert {str(v.version) for v in registered_versions} == {"1", "2", "3"}
        report = {
            "status": "passed",
            "tracking_uri": tracking_uri,
            "mlflow_version": mlflow.__version__,
            "registered_model_name": "iris-versioning-smoke",
            "checks": [
                "changed_weights",
                "config_equality",
                "config_change",
                "retained_files_identical",
                "original_unchanged",
                "native_and_packaged_predictions",
                "three_ready_registry_versions",
            ],
            "versions": rows,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        mlflow.set_tracking_uri(previous_tracking)
        mlflow.set_registry_uri(previous_registry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.output), indent=2))
