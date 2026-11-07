"""Exercise the core plugin against a real, temporary MLflow HTTP server.

Run with --output <new-directory> and --framework all (the default), sklearn,
pytorch, or transformers. Selected frameworks are required; none are skipped.
The server, registry, proxied artifacts, and test report are retained on disk.
"""

import argparse
import importlib.util
import json
import os
import signal
import socket
import subprocess
import sys
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

import mlflow
import numpy as np
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient


@contextmanager
def mlflow_server(output, startup_timeout=60):
    """Own the server process group, including cleanup after startup failures."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    uri = f"http://127.0.0.1:{port}"
    command = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        "1",
        "--backend-store-uri",
        f"sqlite:///{output / 'mlflow.db'}",
        "--serve-artifacts",
        "--artifacts-destination",
        str(output / "artifacts"),
    ]
    with (output / "server.log").open("w") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=output,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + startup_timeout
            opener = build_opener(ProxyHandler({}))
            while True:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"MLflow server exited; see {output / 'server.log'}"
                    )
                try:
                    with opener.open(uri + "/health", timeout=1) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"MLflow server did not start; see {output / 'server.log'}"
                    )
                time.sleep(0.2)
            yield uri
        finally:
            # MLflow's CLI launches a child server. Stop the entire owned group.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


class CoreServerSmokeTests(unittest.TestCase):
    """AAA scenarios using public logging APIs and an independent HTTP client."""

    def setUp(self):
        self.client = MlflowClient(
            tracking_uri=self.server_uri, registry_uri=self.server_uri
        )
        self.experiment_id = self.client.create_experiment(self._testMethodName)
        mlflow.set_experiment(experiment_id=self.experiment_id)
        self.directory = self.output / self._testMethodName
        self.directory.mkdir()
        run = self.client.create_run(self.experiment_id, run_name=self._testMethodName)
        self.run_id = run.info.run_id
        self.addCleanup(self.client.set_terminated, self.run_id)
        experiment = self.client.get_experiment(self.experiment_id)
        self.assertTrue(experiment.artifact_location.startswith("mlflow-artifacts:/"))

    def verify_model(self, info, example, expected, flavor, triton_name):
        """Inspect server records and download/load both native and serving copies."""
        logged = self.client.get_logged_model(info.model_id)
        self.assertEqual(logged.source_run_id, self.run_id)
        self.assertEqual(logged.experiment_id, self.experiment_id)
        self.assertTrue(logged.artifact_location.startswith("mlflow-artifacts:/"))
        metadata = mlflow.models.get_model_info(info.model_uri)
        self.assertEqual(set(metadata.flavors), {flavor, "python_function", "triton"})
        self.assertIsNotNone(metadata.signature)
        download = self.directory / info.model_id
        download.mkdir()
        package = Path(
            mlflow.artifacts.download_artifacts(
                artifact_uri=info.model_uri, dst_path=str(download)
            )
        )
        triton = metadata.flavors["triton"]
        self.assertEqual(triton["model_name"], triton_name)
        repository = package / triton["data"]
        version = str(triton.get("model_version", 1))
        for filename in (
            "config.pbtxt",
            "requirements.txt",
            "validation.json",
            "request.json",
        ):
            self.assertTrue((repository / filename).is_file(), filename)
        self.assertTrue((repository / version / "model.py").is_file())
        self.assertTrue((repository / version / "artifact/MLmodel").is_file())
        self.assertIn(
            f'name: "{triton_name}"', (repository / "config.pbtxt").read_text()
        )
        for uri in (info.model_uri, str(repository / version / "artifact")):
            prediction = mlflow.pyfunc.load_model(uri).predict(example)
            if isinstance(expected, np.ndarray):
                np.testing.assert_allclose(prediction, expected)
            else:
                self.assertEqual(prediction, expected)
        self.report_models.append(
            {
                "test": self._testMethodName,
                "flavor": flavor,
                "model_uri": info.model_uri,
                "artifact_location": logged.artifact_location,
                "registered_version": info.registered_model_version,
                "triton_version": version,
                "package": str(package),
            }
        )
        return package, repository

    @staticmethod
    def iris(columns=4, regularization=1.0):
        from sklearn.datasets import load_iris
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        features, targets = load_iris(return_X_y=True, as_frame=True)
        features = features.iloc[:, :columns]
        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=regularization, max_iter=300)
        )
        model.fit(features, targets)
        return model, features.iloc[::15]

    def test_sklearn_signature_metadata_and_saved_validation(self):
        from mlflow_triton import sklearn

        # Arrange
        model, example = self.iris()
        expected = model.predict_proba(example)
        signature = infer_signature(example, expected)

        # Act
        info = sklearn.log_model(
            model,
            input_example=example,
            signature=signature,
            name="probabilities",
            triton_model_name="iris_serving",
            run_id=self.run_id,
            metadata={"purpose": "http-smoke"},
            tags={"smoke": "core"},
            params={"prediction": "probabilities"},
            pyfunc_predict_fn="predict_proba",
            validate_saved_model=True,
        )

        # Assert
        package, repository = self.verify_model(
            info, example, expected, "sklearn", "iris_serving"
        )
        self.assertIsNone(info.registered_model_version)
        stored = self.client.get_logged_model(info.model_id)
        self.assertEqual(stored.tags["smoke"], "core")
        self.assertEqual(stored.params["prediction"], "probabilities")
        metadata = mlflow.models.get_model_info(info.model_uri)
        self.assertEqual(metadata.signature, signature)
        self.assertEqual(metadata.metadata["purpose"], "http-smoke")
        validation = json.loads((repository / "validation.json").read_text())
        self.assertEqual(validation["saved_artifact_validation"], "passed")
        restored = mlflow.sklearn.load_model(str(package))
        np.testing.assert_allclose(restored.predict_proba(example), expected)

    def test_sklearn_active_run_and_inferred_signature(self):
        from mlflow_triton import sklearn

        # Arrange
        model, example = self.iris()
        expected = model.predict(example)

        # Act
        with mlflow.start_run(run_id=self.run_id):
            info = sklearn.log_model(model, input_example=example, name="classifier")

        # Assert
        _, repository = self.verify_model(
            info, example, expected, "sklearn", "classifier"
        )
        validation = json.loads((repository / "validation.json").read_text())
        self.assertEqual(validation["signature_origin"], "mlflow_inferred")
        self.assertEqual(validation["saved_artifact_validation"], "not_run")

    def test_sklearn_registry_updates_and_alias(self):
        from mlflow_triton import sklearn

        # Arrange
        cases = [self.iris(), self.iris(regularization=0.05), self.iris(columns=3)]
        registered_name = "http-iris"

        # Act
        infos = [
            sklearn.log_model(
                model,
                input_example=example,
                name="iris",
                run_id=self.run_id,
                registered_model_name=registered_name,
            )
            for model, example in cases
        ]
        self.client.set_registered_model_alias(registered_name, "champion", "2")

        # Assert
        groups = []
        for index, (info, (model, example), versions) in enumerate(
            zip(infos, cases, (["1"], ["2"], ["1"])), 1
        ):
            _, repository = self.verify_model(
                info, example, model.predict(example), "sklearn", "iris"
            )
            registered = self.client.get_model_version(registered_name, str(index))
            self.assertEqual(registered.status, "READY")
            self.assertEqual(str(info.registered_model_version), str(index))
            self.assertEqual(
                sorted(p.name for p in repository.iterdir() if p.is_dir()), versions
            )
            groups.append(
                mlflow.models.get_model_info(info.model_uri).flavors["triton"][
                    "repository_group"
                ]
            )
        self.assertEqual(groups[0], groups[1])
        self.assertNotEqual(groups[1], groups[2])
        # Earlier versions remain loadable through their original Registry URI.
        model, example = cases[0]
        original = mlflow.pyfunc.load_model(f"models:/{registered_name}/1")
        np.testing.assert_array_equal(original.predict(example), model.predict(example))
        model, example = cases[1]
        aliased = mlflow.pyfunc.load_model(f"models:/{registered_name}@champion")
        np.testing.assert_array_equal(aliased.predict(example), model.predict(example))

    def test_sklearn_update_after_separate_registration(self):
        from mlflow_triton import sklearn

        # Separately registered models have the same flavor metadata as legacy
        # packages: a Triton repository with folder 1/ and no model_version field.
        model, example = self.iris()
        previous = sklearn.log_model(
            model, input_example=example, name="iris", run_id=self.run_id
        )
        self.assertNotIn(
            "model_version",
            mlflow.models.get_model_info(previous.model_uri).flavors["triton"],
        )
        registered_name = "http-separately-registered-iris"
        mlflow.register_model(previous.model_uri, registered_name)
        updated, example = self.iris(regularization=0.05)

        info = sklearn.log_model(
            updated,
            input_example=example,
            name="iris",
            run_id=self.run_id,
            registered_model_name=registered_name,
        )

        _, repository = self.verify_model(
            info, example, updated.predict(example), "sklearn", "iris"
        )
        self.assertEqual(str(info.registered_model_version), "2")
        self.assertEqual({p.name for p in repository.iterdir() if p.is_dir()}, {"2"})
        original = mlflow.pyfunc.load_model(f"models:/{registered_name}/1")
        np.testing.assert_array_equal(original.predict(example), model.predict(example))

    def test_sklearn_invalid_inputs_do_not_publish_models(self):
        from mlflow_triton import sklearn

        # Arrange
        model, example = self.iris()
        cases = [
            {"input_example": None},
            {"input_example": example.iloc[:0]},
            {"input_example": example, "signature": False},
            {"input_example": example, "device": "gpu"},
            {"input_example": example, "triton_model_name": "bad/name"},
        ]

        # Act / Assert
        for options in cases:
            with self.subTest(options=list(options)), self.assertRaises(ValueError):
                sklearn.log_model(model, run_id=self.run_id, **options)
        self.assertEqual(
            list(self.client.search_logged_models([self.experiment_id])), []
        )

    def test_pytorch_custom_code_and_architecture(self):
        from mlflow_triton import pytorch

        # Arrange
        source = self.directory / "smoke_network.py"
        source.write_text(
            "import torch\n"
            "class Offset(torch.nn.Module):\n"
            "    def __init__(self, offset=4.0):\n"
            "        super().__init__()\n"
            "        self.offset = torch.nn.Parameter(torch.tensor([offset], dtype=torch.float64))\n"
            "    def forward(self, values):\n"
            "        return values + self.offset\n"
        )
        spec = importlib.util.spec_from_file_location("smoke_network", source)
        module = importlib.util.module_from_spec(spec)
        sys.modules["smoke_network"] = module
        self.addCleanup(sys.modules.pop, "smoke_network", None)
        spec.loader.exec_module(module)
        model = module.Offset().eval()
        example = np.array([[1.0], [3.0]])
        expected = example + 4.0

        # Act
        info = pytorch.log_model(
            model,
            input_example=example,
            name="offset",
            run_id=self.run_id,
            architecture_init_kwargs={"offset": 4.0},
            code_paths=[str(source)],
            validate_saved_model=True,
        )
        source.unlink()
        sys.modules.pop("smoke_network", None)

        # Assert
        package, repository = self.verify_model(
            info, example, expected, "pytorch", "offset"
        )
        architecture = json.loads((package / "architecture.json").read_text())
        self.assertEqual(architecture["init_kwargs"], {"offset": 4.0})
        self.assertTrue(
            any(p.endswith("smoke_network.py") for p in architecture["code_files"])
        )
        self.assertEqual(
            (package / "architecture.json").read_bytes(),
            (repository / "1/artifact/architecture.json").read_bytes(),
        )
        validation = json.loads((repository / "validation.json").read_text())
        self.assertEqual(validation["saved_artifact_validation"], "passed")

    def test_pytorch_training_mode_is_rejected(self):
        import torch
        from mlflow_triton import pytorch

        # Arrange
        model = torch.nn.Linear(2, 1).train()

        # Act / Assert
        with self.assertRaisesRegex(ValueError, "eval"):
            pytorch.log_model(model, input_example=np.ones((1, 2)), run_id=self.run_id)
        self.assertEqual(
            list(self.client.search_logged_models([self.experiment_id])), []
        )

    def test_transformers_pipeline_and_components(self):
        import torch
        from huggingface_hub import ModelCard
        from mlflow_triton import transformers as triton_transformers
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import (
            GPT2Config,
            GPT2LMHeadModel,
            PreTrainedTokenizerFast,
            pipeline,
        )

        # Arrange: tiny local weights and vocabulary; no Hub downloads.
        torch.manual_seed(7)
        core = Tokenizer(
            WordLevel(
                {"[UNK]": 0, "[EOS]": 1, "hello": 2, "world": 3}, unk_token="[UNK]"
            )
        )
        core.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=core,
            unk_token="[UNK]",
            eos_token="[EOS]",
            pad_token="[EOS]",
        )
        network = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=4,
                n_layer=1,
                n_head=1,
                n_embd=8,
                n_positions=16,
                eos_token_id=1,
                pad_token_id=1,
            )
        ).eval()
        predictor = pipeline(
            "text-generation", model=network, tokenizer=tokenizer, device=-1
        )
        options = {"max_new_tokens": 2, "do_sample": False, "return_full_text": True}
        example = ["hello"]
        expected = [
            result[0]["generated_text"] for result in predictor(example, **options)
        ]
        card = ModelCard("---\nlanguage: en\n---\nLocal HTTP smoke fixture")
        cases = [
            ("pipeline", predictor),
            ("components", {"model": network, "tokenizer": tokenizer}),
        ]

        # Act
        infos = [
            triton_transformers.log_model(
                supplied,
                input_example=example,
                name=name,
                run_id=self.run_id,
                task="text-generation",
                model_config=options,
                model_card=card,
                pip_requirements=["transformers", "torch"],
            )
            for name, supplied in cases
        ]

        # Assert
        for (name, _), info in zip(cases, infos):
            with self.subTest(model=name):
                package, _ = self.verify_model(
                    info, example, expected, "transformers", name
                )
                architecture = json.loads((package / "architecture.json").read_text())
                self.assertIn("tokenizer", architecture["components"])
                restored = mlflow.transformers.load_model(str(package))
                self.assertEqual(restored.tokenizer.get_vocab(), tokenizer.get_vocab())


def run_smoke(output, framework="all"):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    frameworks = (
        ["sklearn", "pytorch", "transformers"] if framework == "all" else [framework]
    )
    report = {
        "status": "running",
        "mlflow_version": mlflow.__version__,
        "frameworks": frameworks,
        "models": [],
    }
    tracking, registry = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    overrides = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR": "false",
    }
    previous_env = {name: os.environ.get(name) for name in overrides}
    os.environ.update(overrides)
    try:
        for name in frameworks:
            dependency = "torch" if name == "pytorch" else name
            if importlib.util.find_spec(dependency) is None:
                raise ImportError(
                    f"Selected framework requires {dependency}; install smoke dependencies"
                )
        with mlflow_server(output) as uri:
            report["server_uri"] = uri
            mlflow.set_tracking_uri(uri)
            mlflow.set_registry_uri(uri)
            suite = unittest.TestSuite()
            for method in unittest.defaultTestLoader.getTestCaseNames(
                CoreServerSmokeTests
            ):
                if any(method.startswith(f"test_{name}_") for name in frameworks):
                    test = CoreServerSmokeTests(method)
                    test.server_uri, test.output = uri, output
                    test.report_models = report["models"]
                    suite.addTest(test)
            result = unittest.TextTestRunner(verbosity=2).run(suite)
            report.update(
                status="passed" if result.wasSuccessful() else "failed",
                tests_run=result.testsRun,
                failures=[
                    {"test": str(test), "traceback": trace}
                    for test, trace in result.failures + result.errors
                ],
            )
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        mlflow.set_tracking_uri(tracking)
        mlflow.set_registry_uri(registry)
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--framework",
        choices=["all", "sklearn", "pytorch", "transformers"],
        default="all",
    )
    args = parser.parse_args()
    report = run_smoke(args.output, args.framework)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)
