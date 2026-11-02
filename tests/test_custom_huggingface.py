"""Custom HF code must survive packaging without its original source/cache."""

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from mlflow_triton._logging import prepare_model


@unittest.skipUnless(
    importlib.util.find_spec("transformers") and importlib.util.find_spec("torch"),
    "Optional transformers/torch dependencies are not installed",
)
class CustomHuggingFaceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = (
            Path(__file__).resolve().parents[1] / "tests/fixtures/huggingface_custom"
        )
        environment = patch.dict(
            os.environ,
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HOME": str(self.root / "cache"),
                "HF_MODULES_CACHE": str(self.root / "modules"),
            },
        )
        environment.start()
        self.addCleanup(environment.stop)

    def make_fixture(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import (
            AutoConfig,
            AutoModelForSequenceClassification,
            BertConfig,
            BertForSequenceClassification,
            PreTrainedTokenizerFast,
            pipeline,
        )

        source = self.root / "original"
        source.mkdir()
        for filename in ["configuration_custom.py", "modeling_custom.py"]:
            shutil.copy2(self.fixture / filename, source / filename)
        config = {
            "model_type": "mlflow_demo_custom_bert",
            "vocab_size": 4,
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "intermediate_size": 16,
            "max_position_embeddings": 32,
            "logit_temperature": 1.7,
            "auto_map": {
                "AutoConfig": "configuration_custom.CustomBertConfig",
                "AutoModelForSequenceClassification": "modeling_custom.CustomBertForSequenceClassification",
            },
        }
        (source / "config.json").write_text(json.dumps(config))
        loaded_config = AutoConfig.from_pretrained(
            source, trust_remote_code=True, local_files_only=True
        )
        model = AutoModelForSequenceClassification.from_config(
            loaded_config, trust_remote_code=True
        ).eval()
        tokenizer_core = Tokenizer(
            WordLevel(
                {"[UNK]": 0, "[PAD]": 1, "hello": 2, "world": 3}, unk_token="[UNK]"
            )
        )
        tokenizer_core.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer_core, unk_token="[UNK]", pad_token="[PAD]"
        )
        # A tiny synthetic fixture tests behavior; the runnable example reuses trained BERT weights.
        base = BertForSequenceClassification(
            BertConfig(
                **{
                    k: v
                    for k, v in config.items()
                    if k not in ["model_type", "auto_map", "logit_temperature"]
                }
            )
        ).eval()
        base.load_state_dict(model.state_dict(), strict=True)
        predictor = pipeline(
            "text-classification", model=model, tokenizer=tokenizer, device=-1
        )
        return SimpleNamespace(
            model=model,
            base=base,
            tokenizer=tokenizer,
            predictor=predictor,
            source=source,
            example=pd.DataFrame({"text": ["hello world"]}),
        )

    def prepare_fixture(self, fixture):
        from huggingface_hub import ModelCard

        destination = self.root / "saved"
        prepare_model(
            "transformers",
            fixture.predictor,
            destination,
            input_example=fixture.example,
            task="text-classification",
            model_card=ModelCard("---\nlanguage: en\n---\nLocal custom test"),
            pip_requirements=["transformers", "torch"],
        )
        return destination / "model_repository/model/1/artifact"

    def reload_in_fresh_process(self, artifact):
        script = """
import json, sys
import mlflow.pyfunc, mlflow.transformers
import pandas as pd
native = mlflow.transformers.load_model(sys.argv[1])
prediction = pd.DataFrame(mlflow.pyfunc.load_model(sys.argv[1]).predict(pd.DataFrame({'text':['hello world']})))
print(json.dumps({'class_name': type(native.model).__name__,
                  'temperature': native.model.config.logit_temperature,
                  'prediction': prediction.to_dict(orient='list')}))
"""
        environment = {
            **os.environ,
            "HF_HOME": str(self.root / "fresh-cache"),
            "HF_MODULES_CACHE": str(self.root / "fresh-modules"),
        }
        return subprocess.run(
            [sys.executable, "-c", script, str(artifact)],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=90,
        )

    def test_custom_architecture_applies_temperature(self):
        # Arrange
        import torch

        fixture = self.make_fixture()
        tokens = fixture.tokenizer("hello world", return_tensors="pt")
        with torch.inference_mode():
            expected = fixture.base(**tokens).logits.numpy() / 1.7

            # Act
            actual = fixture.model(**tokens).logits.numpy()

        # Assert
        np.testing.assert_allclose(actual, expected)

    def test_packaging_includes_custom_architecture_files(self):
        # Arrange
        fixture = self.make_fixture()

        # Act
        artifact = self.prepare_fixture(fixture)

        # Assert
        self.assertTrue(list(artifact.rglob("modeling_custom.py")))
        self.assertTrue(list(artifact.rglob("configuration_custom.py")))

    def test_custom_code_reloads_without_original_source_or_cache(self):
        # Arrange
        import mlflow.pyfunc

        fixture = self.make_fixture()
        artifact = self.prepare_fixture(fixture)
        expected = pd.DataFrame(
            mlflow.pyfunc.load_model(str(artifact)).predict(fixture.example)
        )
        shutil.rmtree(fixture.source)

        # Act
        result = self.reload_in_fresh_process(artifact)

        # Assert
        self.assertEqual(result.returncode, 0, result.stderr)
        actual = json.loads(result.stdout)
        self.assertEqual(actual["class_name"], "CustomBertForSequenceClassification")
        self.assertEqual(actual["temperature"], 1.7)
        pd.testing.assert_frame_equal(
            pd.DataFrame(actual["prediction"]),
            expected,
            check_exact=False,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_reload_fails_when_packaged_architecture_is_missing(self):
        # Arrange
        fixture = self.make_fixture()
        artifact = self.prepare_fixture(fixture)
        shutil.rmtree(fixture.source)
        for file in artifact.rglob("modeling_custom.py"):
            file.unlink()

        # Act
        result = self.reload_in_fresh_process(artifact)

        # Assert
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("modeling_custom", result.stderr)
