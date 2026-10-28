"""Local saved-artifact and wrapper tests; no live Triton or model downloads."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import Model, infer_signature
from mlflow.types.schema import ColSpec, DataType, Schema
from mlflow_triton import (
    _logging,
    _runtime,
    pytorch,
    sklearn,
    transformers,
    triton,
)
from mlflow_triton._architecture import validate_init_kwargs, write_architecture
from mlflow_triton._signature import config_text, contract_from_signature, model_config
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fixture_model():
    """Explicit tiny fitted-state fixture: no training is executed."""
    scale = StandardScaler()
    scale.mean_ = np.array([2.0, 4.0])
    scale.scale_ = np.array([2.0, 3.0])
    scale.var_ = scale.scale_**2
    scale.n_features_in_ = 2
    scale.n_samples_seen_ = 3
    scale.feature_names_in_ = np.array(["left", "right"], dtype=object)
    linear = LinearRegression()
    linear.coef_ = np.array([3.0, -2.0])
    linear.intercept_ = 1.0
    linear.n_features_in_ = 2
    return Pipeline([("scale", scale), ("linear", linear)])


class LoggingWrapperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.example = pd.DataFrame({"left": [2.0, 6.0], "right": [4.0, 7.0]})

    def prepare(self, **kwargs):
        destination = self.root / "saved"
        _logging.prepare_model(
            "sklearn",
            fixture_model(),
            destination,
            input_example=self.example,
            pip_requirements=["scikit-learn==" + __import__("sklearn").__version__],
            **kwargs,
        )
        return destination

    @unittest.skipUnless(
        importlib.util.find_spec("tritonclient"), "tritonclient is optional"
    )
    def test_config_parses_multiple_inputs_and_outputs_with_triton_schema(self):
        from google.protobuf import json_format, text_format
        from tritonclient.grpc import model_config_pb2

        signatures = [
            infer_signature(
                pd.DataFrame(
                    {"text": ["example"], "count": np.array([1], dtype=np.int64)}
                ),
                pd.DataFrame({"label": ["ok"], "score": [0.5]}),
            ),
            infer_signature(
                {
                    "tokens": np.ones((2, 3), dtype=np.int64),
                    "mask": np.ones((2, 3), dtype=np.int64),
                },
                {
                    "scores": np.ones((2, 5), dtype=np.float32),
                    "ids": np.ones((2,), dtype=np.int64),
                },
            ),
            infer_signature(
                np.ones((2, 3), dtype=np.float32), np.ones((2, 1), dtype=np.float32)
            ),
        ]
        for signature in signatures:
            for device in ["cpu", "gpu"]:
                with self.subTest(signature=signature, device=device):
                    contract = contract_from_signature(signature, "pytorch")
                    config = model_config_pb2.ModelConfig()
                    text_format.Parse(config_text("example", contract, device), config)
                    for field, specs in [
                        (config.input, contract["inputs"]),
                        (config.output, contract["outputs"]),
                    ]:
                        self.assertEqual(
                            [item.name for item in field], [s["name"] for s in specs]
                        )
                        for item, spec in zip(field, specs):
                            self.assertEqual(list(item.dims), spec["dims"])
                            dtype = (
                                "STRING"
                                if spec["datatype"] == "BYTES"
                                else spec["datatype"]
                            )
                            self.assertEqual(
                                item.data_type,
                                model_config_pb2.DataType.Value("TYPE_" + dtype),
                            )
                    self.assertEqual(config.backend, "python")
                    self.assertEqual(config.max_batch_size, 0)
                    restored = _runtime.contract_from_config(
                        json_format.MessageToDict(
                            config, preserving_proto_field_name=True
                        )
                    )
                    for direction in ("inputs", "outputs"):
                        for actual, expected in zip(
                            restored[direction], contract[direction]
                        ):
                            for key in (
                                "name",
                                "source_name",
                                "dims",
                                "datatype",
                                "logical_type",
                                "named",
                            ):
                                self.assertEqual(actual[key], expected[key])

    @unittest.skipUnless(
        importlib.util.find_spec("tritonclient"), "tritonclient is optional"
    )
    def test_example_reader_uses_generated_config_without_contract_file(self):
        # Arrange
        path = self.prepare()
        repository = path / "model_repository/model"
        source = Path(__file__).resolve().parents[1] / "examples/serving_validation.py"
        spec = importlib.util.spec_from_file_location("validation_reader_test", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Act
        contract = module.read_model_contract(repository)
        restored = _runtime.decode_inputs(
            _runtime.example_tensors(self.example, contract), contract
        )

        # Assert
        self.assertFalse((repository / "1/contract.json").exists())
        pd.testing.assert_frame_equal(restored, self.example)

    def test_every_public_wrapper_requires_example_before_logging(self):
        for wrapper in (sklearn, pytorch, transformers):
            with (
                self.subTest(wrapper=wrapper.__name__),
                patch.object(Model, "log") as log,
            ):
                with self.assertRaises(TypeError):
                    wrapper.log_model(object())
                with self.assertRaisesRegex(ValueError, "required"):
                    wrapper.log_model(object(), input_example=None)
                log.assert_not_called()

    def test_saved_pipeline_native_and_triton_copy_have_same_predictions(self):
        path = self.prepare()
        metadata = Model.load(path / "MLmodel")
        self.assertEqual(
            set(metadata.flavors), {"sklearn", "python_function", "triton"}
        )
        repository = path / metadata.flavors["triton"]["data"]
        native = mlflow.pyfunc.load_model(str(path))
        packaged = mlflow.pyfunc.load_model(str(repository / "1/artifact"))
        held_out = pd.DataFrame({"left": [10.0, -4.0, 0.0], "right": [1.0, 16.0, 4.0]})
        expected = fixture_model().predict(held_out)
        np.testing.assert_array_equal(native.predict(held_out), expected)
        np.testing.assert_array_equal(packaged.predict(held_out), expected)
        text = (repository / "config.pbtxt").read_text()
        self.assertIn('backend: "python"', text)
        self.assertIn("max_batch_size: 0", text)
        self.assertIn("KIND_CPU", text)
        self.assertIn('key: "mlflow_contract"', text)
        request = json.loads((repository / "request.json").read_text())
        self.assertEqual(request["outputs"], [{"name": "output_0"}])
        self.assertEqual(request["inputs"][0]["data"], self.example.left.tolist())
        self.assertFalse((repository / "request.sh").exists())
        self.assertFalse((repository / "REQUEST.md").exists())

        self.assertFalse((repository / "1/contract.json").exists())
        report = json.loads((repository / "validation.json").read_text())
        self.assertEqual(report["triton_validation"], "not_run")
        self.assertEqual(report["signature_origin"], "mlflow_inferred")

    def test_supplied_signature_is_preserved(self):
        signature = infer_signature(self.example, fixture_model().predict(self.example))
        path = self.prepare(signature=signature)
        self.assertEqual(Model.load(path / "MLmodel").signature, signature)

    def test_default_logging_skips_model_loading_and_dependency_inference(self):
        model = fixture_model()
        calls = []
        original_predict = Pipeline.predict

        def predict(instance, values, **kwargs):
            calls.append(instance)
            return original_predict(instance, values, **kwargs)

        def inspect_logged_package(**kwargs):
            path = Path(kwargs["prepared_path"])
            report = json.loads(
                (path / "model_repository/model/validation.json").read_text()
            )
            self.assertEqual(
                report["local_validation"],
                "in_memory_prediction_and_tensor_mapping_passed",
            )
            self.assertEqual(report["saved_artifact_validation"], "not_run")
            requirements = (path / "requirements.txt").read_text()
            self.assertIn("scikit-learn==", requirements)
            self.assertIn("cloudpickle==", requirements)
            return "logged"

        with (
            patch.object(Pipeline, "predict", predict),
            patch("mlflow.pyfunc.load_model", side_effect=AssertionError("reload")),
            patch(
                "mlflow.models.infer_pip_requirements",
                side_effect=AssertionError("dependency inference reload"),
            ),
            patch.object(Model, "log", side_effect=inspect_logged_package),
        ):
            result = sklearn.log_model(model, input_example=self.example)
        self.assertEqual(result, "logged")
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], model)

    def test_default_requirements_keep_extra_requirements_and_constraints(self):
        constraints = self.root / "constraints.txt"
        constraints.write_text("numpy>=1\n")
        extra_file = self.root / "extra.txt"
        extra_file.write_text(f"pandas\n-c {constraints}\n")
        for index, extra in enumerate((["pandas"], str(extra_file))):
            with (
                self.subTest(extra=extra),
                patch(
                    "mlflow.models.infer_pip_requirements",
                    side_effect=AssertionError("dependency inference reload"),
                ),
            ):
                path = self.root / f"extra-{index}"
                _logging.prepare_model(
                    "sklearn",
                    fixture_model(),
                    path,
                    input_example=self.example,
                    extra_pip_requirements=extra,
                )
                requirements = (path / "requirements.txt").read_text()
                self.assertIn("scikit-learn==", requirements)
                self.assertIn("pandas", requirements)
                if isinstance(extra, str):
                    self.assertIn("-c constraints.txt", requirements)
                    self.assertIn("numpy>=1", (path / "constraints.txt").read_text())

    def test_in_memory_validation_uses_selected_sklearn_prediction_method(self):
        model = LogisticRegression()
        model.classes_ = np.array([0, 1])
        model.coef_ = np.array([[1.0, -1.0]])
        model.intercept_ = np.array([0.0])
        model.n_features_in_ = 2
        values = self.example.to_numpy()
        expected = model.predict_proba(values)
        path = self.root / "probabilities"
        _logging.prepare_model(
            "sklearn",
            model,
            path,
            input_example=values,
            pyfunc_predict_fn="predict_proba",
        )
        signature = Model.load(path / "MLmodel").signature
        self.assertEqual(list(signature.outputs.inputs[0].shape), [-1, 2])
        np.testing.assert_array_equal(
            mlflow.pyfunc.load_model(str(path)).predict(values), expected
        )

    def test_saved_artifact_validation_is_explicit_and_reported(self):
        with patch.object(
            _runtime, "load_predictor", wraps=_runtime.load_predictor
        ) as load:
            path = self.prepare(validate_saved_model=True)
        load.assert_called_once()
        report = json.loads(
            (path / "model_repository/model/validation.json").read_text()
        )
        self.assertEqual(
            report["local_validation"],
            "saved_artifact_prediction_and_tensor_mapping_passed",
        )
        self.assertEqual(report["saved_artifact_validation"], "passed")
        self.assertEqual(report["validation_device"], "cpu")

    def test_saved_artifact_validation_failure_prevents_logging(self):
        with (
            patch.object(
                _runtime, "load_predictor", side_effect=ImportError("missing code")
            ),
            patch.object(Model, "log") as log,
        ):
            with self.assertRaisesRegex(ImportError, "missing code"):
                sklearn.log_model(
                    fixture_model(),
                    input_example=self.example,
                    validate_saved_model=True,
                    pip_requirements=["scikit-learn"],
                )
            log.assert_not_called()

    def test_invalid_signature_does_not_log(self):
        # Arrange
        signature = mlflow.models.ModelSignature(
            inputs=Schema([ColSpec(DataType.double, "absent")])
        )
        with patch.object(Model, "log") as log:
            with self.assertRaises((ValueError, mlflow.exceptions.MlflowException)):
                sklearn.log_model(
                    fixture_model(),
                    input_example=self.example,
                    signature=signature,
                    pip_requirements=["scikit-learn"],
                )
            log.assert_not_called()

    def test_invalid_signature_leaves_no_destination(self):
        # Arrange
        destination = self.root / "invalid-model"
        signature = mlflow.models.ModelSignature(
            inputs=Schema([ColSpec(DataType.double, "absent")])
        )

        # Act / Assert
        with self.assertRaises((ValueError, mlflow.exceptions.MlflowException)):
            _logging.prepare_model(
                "sklearn",
                fixture_model(),
                destination,
                input_example=self.example,
                signature=signature,
                pip_requirements=["scikit-learn"],
            )
        self.assertFalse(destination.exists())

    def test_invalid_serving_options_fail_before_saving(self):
        # Arrange
        cases = [
            {"device": "invalid"},
            {"device": "gpu"},
            {"triton_model_name": "bad/name"},
            {"triton_model_name": ""},
        ]
        for options in cases:
            with (
                self.subTest(options=options),
                patch("mlflow.sklearn.save_model") as save,
            ):
                # Act / Assert
                with self.assertRaises(ValueError):
                    sklearn.log_model(
                        fixture_model(), input_example=self.example, **options
                    )
                save.assert_not_called()

    def test_partial_serving_copy_is_cleaned_up(self):
        # Arrange
        destination = self.root / "saved"

        def fail_copy(source, target, **kwargs):
            Path(target).mkdir()
            (Path(target) / "partial.bin").write_bytes(b"partial")
            raise OSError("disk full")

        # Act / Assert
        with patch.object(_logging.shutil, "copytree", side_effect=fail_copy):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.prepare()
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_publish_failure_cleans_staging_and_preserves_existing_target(self):
        # Arrange
        destination = self.root / "saved"
        real_assemble = _logging._assemble_repository

        def create_conflict(*args, **kwargs):
            package = real_assemble(*args, **kwargs)
            destination.mkdir()
            (destination / "existing.txt").write_text("keep")
            return package

        # Act / Assert
        with patch.object(
            _logging, "_assemble_repository", side_effect=create_conflict
        ):
            with self.assertRaises(FileExistsError):
                self.prepare()
        self.assertEqual((destination / "existing.txt").read_text(), "keep")
        self.assertEqual(list(self.root.iterdir()), [destination])

    def test_real_log_register_and_load_keeps_both_flavors(self):
        tracking, registry = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
        try:
            uri = "sqlite:///" + str(self.root / "tracking.db")
            mlflow.set_tracking_uri(uri)
            mlflow.set_registry_uri(uri)
            experiment = mlflow.create_experiment(
                "wrapper", artifact_location=(self.root / "artifacts").as_uri()
            )
            with mlflow.start_run(experiment_id=experiment):
                with (
                    patch("mlflow.pyfunc.load_model") as reload,
                    patch("mlflow.models.utils._validate_serving_input") as validate,
                    patch("mlflow.models.infer_pip_requirements") as infer,
                ):
                    info = sklearn.log_model(
                        fixture_model(),
                        input_example=self.example,
                        name="wrapped",
                        registered_model_name="wrapped",
                    )
                    reload.assert_not_called()
                    validate.assert_not_called()
                    infer.assert_not_called()
            metadata = mlflow.models.get_model_info(info.model_uri)
            self.assertIsNotNone(metadata.signature)
            self.assertEqual(
                set(metadata.flavors), {"sklearn", "python_function", "triton"}
            )
            repository = triton.load_model("models:/wrapped/1")
            self.assertTrue((repository / "1/model.py").is_file())
            np.testing.assert_array_equal(
                mlflow.pyfunc.load_model(info.model_uri).predict(self.example),
                fixture_model().predict(self.example),
            )
        finally:
            mlflow.set_tracking_uri(tracking)
            mlflow.set_registry_uri(registry)

    def test_generated_python_backend_executes_and_rejects_bad_requests(self):
        path = self.prepare()
        version = path / "model_repository/model/1"
        spec = importlib.util.spec_from_file_location(
            "generated_backend", version / "model.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Tensor:
            def __init__(self, name, value):
                self.name, self.value = name, value

            def as_numpy(self):
                return self.value

        utils = SimpleNamespace(
            Tensor=Tensor,
            TritonError=str,
            InferenceResponse=lambda **kwargs: kwargs,
            get_input_tensor_by_name=lambda request, name: request.get(name),
        )
        with patch.dict(sys.modules, {"triton_python_backend_utils": utils}):
            backend = module.TritonPythonModel()
            backend.initialize(
                {
                    "model_instance_kind": "CPU",
                    "model_instance_device_id": "0",
                    "model_config": json.dumps(
                        model_config(
                            "model",
                            contract_from_signature(
                                Model.load(path / "MLmodel").signature, "sklearn"
                            ),
                            "cpu",
                        )
                    ),
                }
            )
            inputs = {
                "left": Tensor("left", self.example.left.to_numpy()),
                "right": Tensor("right", self.example.right.to_numpy()),
            }
            replies = backend.execute(
                [inputs, {}, {**inputs, "left": Tensor("left", np.ones(2, np.int32))}]
            )
        np.testing.assert_array_equal(
            replies[0]["output_tensors"][0].value, fixture_model().predict(self.example)
        )
        self.assertIn("Missing input", replies[1]["error"])
        self.assertIn("dtype", replies[2]["error"])

    @unittest.skipUnless(
        importlib.util.find_spec("torch"), "Optional torch dependency is not installed"
    )
    def test_pytorch_keeps_python_model_without_graph_export(self):
        import torch

        model = torch.nn.Linear(2, 1).double().eval()
        values = self.example.to_numpy()
        original = {key: tensor.clone() for key, tensor in model.state_dict().items()}
        path = self.root / "torch"
        _logging.prepare_model(
            "pytorch",
            model,
            path,
            input_example=values,
            pip_requirements=["torch==" + torch.__version__],
        )
        report = json.loads((path / "architecture.json").read_text())
        self.assertEqual(report["model_class"], "torch.nn.modules.linear.Linear")
        self.assertEqual(report["init_kwargs_origin"], "unknown")
        self.assertEqual(report["tensors"][0]["shape"], [1, 2])
        self.assertEqual(
            (path / "architecture.json").read_bytes(),
            (path / "model_repository/model/1/artifact/architecture.json").read_bytes(),
        )
        restored = mlflow.pytorch.load_model(str(path))
        for key, tensor in restored.state_dict().items():
            torch.testing.assert_close(tensor, original[key], rtol=0, atol=0)
        self.assertTrue((path / "model_repository/model/1/model.py").exists())

    @unittest.skipUnless(
        importlib.util.find_spec("torch"), "Optional torch dependency is not installed"
    )
    def test_pytorch_logging_reuses_original_weights_for_gpu_target(self):
        import torch

        model = torch.nn.Linear(2, 1).double().eval()
        values = np.array([[2.0, 4.0], [6.0, 7.0]])
        original = {key: tensor.clone() for key, tensor in model.state_dict().items()}
        calls = []
        original_forward = torch.nn.Linear.forward

        def forward(instance, inputs):
            calls.append(instance)
            return original_forward(instance, inputs)

        def inspect_logged_package(**kwargs):
            path = Path(kwargs["prepared_path"])
            repository = path / "model_repository/model"
            self.assertIn("KIND_GPU", (repository / "config.pbtxt").read_text())
            report = json.loads((repository / "validation.json").read_text())
            self.assertEqual(report["validation_device"], "cpu")
            self.assertEqual(report["saved_artifact_validation"], "not_run")
            self.assertIn("torch==", (path / "requirements.txt").read_text())
            return "logged"

        with (
            patch.object(torch.nn.Linear, "forward", forward),
            patch("torch.load", side_effect=AssertionError("weight reload")),
            patch("mlflow.pyfunc.load_model", side_effect=AssertionError("reload")),
            patch(
                "mlflow.models.infer_pip_requirements",
                side_effect=AssertionError("dependency inference reload"),
            ),
            patch.object(Model, "log", side_effect=inspect_logged_package),
        ):
            result = pytorch.log_model(model, input_example=values, device="gpu")
        self.assertEqual(result, "logged")
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0], model)
        self.assertFalse(model.training)
        for key, tensor in model.state_dict().items():
            self.assertEqual(str(tensor.device), "cpu")
            torch.testing.assert_close(tensor, original[key], rtol=0, atol=0)

    @unittest.skipUnless(
        importlib.util.find_spec("torch"), "Optional torch dependency is not installed"
    )
    def test_pytorch_rejects_state_dict(self):
        # Arrange
        import torch

        weights = torch.nn.Linear(2, 1).state_dict()
        values = self.example.to_numpy()

        # Act / Assert
        with self.assertRaisesRegex(ValueError, "state_dict"):
            pytorch.log_model(weights, input_example=values)

    def test_transformers_wrapper_forwards_pipeline_assets_and_options(self):
        sentinel = object()
        with patch(
            "mlflow_triton.transformers.log_native_model", return_value=sentinel
        ) as log:
            pipeline = object()
            result = transformers.log_model(
                pipeline,
                input_example=["hello"],
                architecture_init_kwargs={"hidden_size": 8},
                task="text-generation",
                model_config={"max_new_tokens": 8},
                code_paths=["architecture.py"],
            )
        self.assertIs(result, sentinel)
        self.assertEqual(
            log.call_args.kwargs["architecture_init_kwargs"], {"hidden_size": 8}
        )
        self.assertIs(log.call_args.args[1], pipeline)
        self.assertEqual(log.call_args.kwargs["model_config"], {"max_new_tokens": 8})
        self.assertEqual(log.call_args.kwargs["code_paths"], ["architecture.py"])

    @unittest.skipUnless(
        importlib.util.find_spec("torch"), "Optional torch dependency is not installed"
    )
    def test_custom_pytorch_architecture_loads_in_fresh_process(self):
        source = self.root / "custom_network.py"
        source.write_text(
            "import torch\n"
            "class OffsetNetwork(torch.nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n"
            "        self.offset = torch.nn.Parameter(torch.tensor([2.0]))\n"
            "    def forward(self, x):\n"
            "        return x + self.offset\n"
        )
        spec = importlib.util.spec_from_file_location("custom_network", source)
        module = importlib.util.module_from_spec(spec)
        # Do not snapshot/restore all sys.modules around framework imports:
        # unloading their extension modules can corrupt subsequent tests.
        sys.modules["custom_network"] = module
        self.addCleanup(sys.modules.pop, "custom_network", None)
        spec.loader.exec_module(module)
        model = module.OffsetNetwork().eval()
        path = self.root / "custom"
        _logging.prepare_model(
            "pytorch",
            model,
            path,
            input_example=np.array([[1.0]], dtype=np.float32),
            code_paths=[str(source)],
            pip_requirements=["torch", "numpy"],
        )
        report = json.loads((path / "architecture.json").read_text())
        self.assertEqual(report["model_class"], "custom_network.OffsetNetwork")
        self.assertTrue(
            any(name.endswith("custom_network.py") for name in report["code_files"])
        )
        source.unlink()
        script = (
            "import mlflow.pyfunc,numpy as np,sys\n"
            "model=mlflow.pyfunc.load_model(sys.argv[1])\n"
            "np.testing.assert_array_equal(model.predict(np.array([[5.]],dtype=np.float32)), [[7.]])\n"
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(path / "model_repository/model/1/artifact"),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )

    @unittest.skipUnless(
        importlib.util.find_spec("transformers") and importlib.util.find_spec("torch"),
        "Optional transformers/torch dependencies are not installed",
    )
    def test_local_huggingface_generation_preserves_tokenizer_and_configuration(self):
        from huggingface_hub import ModelCard
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import (
            GPT2Config,
            GPT2LMHeadModel,
            PreTrainedTokenizerFast,
            pipeline,
        )

        tokenizer_core = Tokenizer(
            WordLevel(
                {"[UNK]": 0, "[EOS]": 1, "hello": 2, "world": 3}, unk_token="[UNK]"
            )
        )
        tokenizer_core.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_core,
            unk_token="[UNK]",
            eos_token="[EOS]",
            pad_token="[EOS]",
        )
        tokenizer.chat_template = (
            "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        )
        model = GPT2LMHeadModel(
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
            "text-generation", model=model, tokenizer=tokenizer, device=-1
        )
        path = self.root / "hf"
        model_config = {
            "max_new_tokens": 2,
            "do_sample": False,
            "return_full_text": True,
        }
        expected = mlflow.transformers._TransformersWrapper(
            predictor, model_config=model_config
        ).predict(["hello"])
        with patch.dict(
            os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        ):
            with (
                patch("mlflow.pyfunc.load_model") as reload,
                patch("mlflow.transformers.infer_pip_requirements") as infer,
            ):
                _logging.prepare_model(
                    "transformers",
                    predictor,
                    path,
                    input_example=["hello"],
                    model_config=model_config,
                    model_card=ModelCard("---\nlanguage: en\n---\nLocal test fixture"),
                    pip_requirements=["transformers", "torch"],
                )
                reload.assert_not_called()
                infer.assert_not_called()
            reloaded = mlflow.transformers.load_model(str(path))
            self.assertEqual(reloaded.tokenizer.chat_template, tokenizer.chat_template)
            pyfunc = mlflow.pyfunc.load_model(str(path))
            embedded = mlflow.pyfunc.load_model(
                str(path / "model_repository/model/1/artifact")
            )
            self.assertEqual(pyfunc.predict(["hello"]), expected)
            self.assertEqual(embedded.predict(["hello"]), expected)

            # Even on a host with a GPU available, a component dictionary must
            # keep the supplied model on CPU and avoid loading another copy.
            components_path = self.root / "hf-components"
            components = {"model": model, "tokenizer": tokenizer}
            with (
                patch("torch.cuda.is_available", return_value=True),
                patch("torch.nn.Module.to", side_effect=AssertionError("model moved")),
                patch("mlflow.pyfunc.load_model") as reload,
                patch("mlflow.transformers.infer_pip_requirements") as infer,
            ):
                _logging.prepare_model(
                    "transformers",
                    components,
                    components_path,
                    task="text-generation",
                    input_example=["hello"],
                    model_config=model_config,
                    model_card=ModelCard("---\nlanguage: en\n---\nLocal test fixture"),
                    pip_requirements=["transformers", "torch"],
                )
                reload.assert_not_called()
                infer.assert_not_called()
            self.assertEqual(set(components), {"model", "tokenizer"})
            self.assertEqual(str(model.device), "cpu")
            self.assertEqual(
                mlflow.pyfunc.load_model(str(components_path)).predict(["hello"]),
                expected,
            )
        report = json.loads((path / "architecture.json").read_text())
        self.assertTrue(report["model_class"].endswith("GPT2LMHeadModel"))
        self.assertIn("tokenizer", report["components"])
        self.assertTrue(report["config_files"])
        signature = Model.load(path / "MLmodel").signature
        self.assertIsNotNone(signature.inputs)
        self.assertIsNotNone(signature.outputs)

    def test_architecture_records_explicit_constructor_settings_without_loading(self):
        # Arrange
        native = self.root / "native"
        native.mkdir()
        settings = {"hidden_size": 32, "bias": False}

        # Act
        write_architecture(native, "pytorch", object(), settings)
        report = json.loads((native / "architecture.json").read_text())

        # Assert
        self.assertEqual(report["init_kwargs"], settings)
        self.assertEqual(report["init_kwargs_origin"], "user_supplied")

    def test_architecture_rejects_non_json_constructor_settings(self):
        # Arrange
        invalid = [[], {"tensor": object()}, {"size": float("nan")}]

        # Act / Assert
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                validate_init_kwargs(settings)

    def named_tensor_fixture(self):
        inputs = {
            "tokens": np.ones((2, 3), dtype=np.int64),
            "mask": np.ones((2, 3), dtype=np.int64),
        }
        outputs = {"scores": np.ones((2, 5), dtype=np.float32)}
        contract = contract_from_signature(infer_signature(inputs, outputs), "pytorch")
        return inputs, outputs, contract

    def test_named_tensor_mapping_preserves_inputs(self):
        # Arrange
        inputs, _, contract = self.named_tensor_fixture()
        tensors = _runtime.example_tensors(inputs, contract)

        # Act
        restored = _runtime.decode_inputs(tensors, contract)

        # Assert
        for key, value in inputs.items():
            np.testing.assert_array_equal(restored[key], value)

    def test_gpu_configuration_uses_gpu_instance(self):
        # Arrange
        _, _, contract = self.named_tensor_fixture()

        # Act
        config = config_text("model", contract, "gpu")

        # Assert
        self.assertIn("KIND_GPU", config)

    def test_named_tensor_contract_preserves_signature_dimensions(self):
        # Arrange
        inputs, outputs, _ = self.named_tensor_fixture()
        signature = infer_signature(inputs, outputs)

        # Act
        contract = contract_from_signature(signature, "pytorch")

        # Assert
        self.assertEqual(contract["inputs"][0]["dims"], [-1, 3])

    def test_output_encoding_rejects_wrong_dtype(self):
        # Arrange
        _, outputs, contract = self.named_tensor_fixture()
        invalid = {"scores": outputs["scores"].astype(np.float64)}

        # Act / Assert
        with self.assertRaisesRegex(ValueError, "dtype"):
            _runtime.encode_outputs(invalid, contract)


if __name__ == "__main__":
    unittest.main()
