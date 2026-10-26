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
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlflow_triton_control import (
    _logging,
    _runtime,
    pytorch,
    sklearn,
    transformers,
    triton,
)
from mlflow_triton_control._signature import config_text, contract_from_signature


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
        report = json.loads((repository / "validation.json").read_text())
        self.assertEqual(report["triton_validation"], "not_run")
        self.assertEqual(report["signature_origin"], "mlflow_inferred")

    def test_supplied_signature_is_preserved(self):
        signature = infer_signature(self.example, fixture_model().predict(self.example))
        path = self.prepare(signature=signature)
        self.assertEqual(Model.load(path / "MLmodel").signature, signature)

    def test_invalid_signature_does_not_log_or_leave_destination(self):
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
                info = sklearn.log_model(
                    fixture_model(),
                    input_example=self.example,
                    name="wrapped",
                    registered_model_name="wrapped",
                    pip_requirements=["scikit-learn"],
                )
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
                {"model_instance_kind": "CPU", "model_instance_device_id": "0"}
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
        restored = mlflow.pytorch.load_model(str(path))
        for key, tensor in restored.state_dict().items():
            torch.testing.assert_close(tensor, original[key], rtol=0, atol=0)
        self.assertTrue((path / "model_repository/model/1/model.py").exists())
        with self.assertRaisesRegex(ValueError, "state_dict"):
            pytorch.log_model(original, input_example=values)

    def test_transformers_wrapper_forwards_pipeline_assets_and_options(self):
        sentinel = object()
        with patch(
            "mlflow_triton_control.transformers.log_native_model", return_value=sentinel
        ) as log:
            pipeline = object()
            result = transformers.log_model(
                pipeline,
                input_example=["hello"],
                task="text-generation",
                model_config={"max_new_tokens": 8},
                code_paths=["architecture.py"],
            )
        self.assertIs(result, sentinel)
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
        with patch.dict(
            os.environ, {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        ):
            _logging.prepare_model(
                "transformers",
                predictor,
                path,
                input_example=["hello"],
                model_config={
                    "max_new_tokens": 2,
                    "do_sample": False,
                    "return_full_text": True,
                },
                model_card=ModelCard("---\nlanguage: en\n---\nLocal test fixture"),
                pip_requirements=["transformers", "torch"],
            )
            reloaded = mlflow.transformers.load_model(str(path))
            self.assertEqual(reloaded.tokenizer.chat_template, tokenizer.chat_template)
            pyfunc = mlflow.pyfunc.load_model(str(path))
            embedded = mlflow.pyfunc.load_model(
                str(path / "model_repository/model/1/artifact")
            )
            self.assertEqual(pyfunc.predict(["hello"]), embedded.predict(["hello"]))
        signature = Model.load(path / "MLmodel").signature
        self.assertIsNotNone(signature.inputs)
        self.assertIsNotNone(signature.outputs)

    def test_named_tensor_mapping_and_device_configuration(self):
        inputs = {
            "tokens": np.ones((2, 3), dtype=np.int64),
            "mask": np.ones((2, 3), dtype=np.int64),
        }
        outputs = {"scores": np.ones((2, 5), dtype=np.float32)}
        contract = contract_from_signature(infer_signature(inputs, outputs), "pytorch")
        tensors = _runtime.example_tensors(inputs, contract)
        restored = _runtime.decode_inputs(tensors, contract)
        for key, value in inputs.items():
            np.testing.assert_array_equal(restored[key], value)
        self.assertIn("KIND_GPU", config_text("model", contract, "gpu"))
        self.assertEqual(contract["inputs"][0]["dims"], [-1, 3])
        with self.assertRaisesRegex(ValueError, "dtype"):
            _runtime.encode_outputs(
                {"scores": outputs["scores"].astype(np.float64)}, contract
            )


if __name__ == "__main__":
    unittest.main()
