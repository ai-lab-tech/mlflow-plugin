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
from unittest.mock import patch

import numpy as np
import pandas as pd

from mlflow_triton_control._logging import prepare_model


@unittest.skipUnless(
    importlib.util.find_spec("transformers") and importlib.util.find_spec("torch"),
    "Optional transformers/torch dependencies are not installed",
)
class CustomHuggingFaceTests(unittest.TestCase):
    def test_custom_code_packaged_and_reloaded_without_source(self):
        import mlflow.pyfunc
        import torch
        from huggingface_hub import ModelCard
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

        fixture = (
            Path(__file__).resolve().parents[1]
            / "examples/data/custom/huggingface_custom/bundle"
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(
                os.environ,
                {
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                },
            ),
        ):
            root = Path(tmp)
            source = root / "original"
            source.mkdir()
            for filename in ["configuration_custom.py", "modeling_custom.py"]:
                shutil.copy2(fixture / filename, source / filename)
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
            tokens = tokenizer("hello world", return_tensors="pt")
            with torch.inference_mode():
                np.testing.assert_allclose(
                    model(**tokens).logits.numpy(), base(**tokens).logits.numpy() / 1.7
                )
            predictor = pipeline(
                "text-classification", model=model, tokenizer=tokenizer, device=-1
            )
            destination = root / "saved"
            example = pd.DataFrame({"text": ["hello world"]})
            prepare_model(
                "transformers",
                predictor,
                destination,
                input_example=example,
                task="text-classification",
                model_card=ModelCard("---\nlanguage: en\n---\nLocal custom test"),
                pip_requirements=["transformers", "torch"],
            )
            artifact = destination / "model_repository/model/1/artifact"
            custom_files = list(artifact.rglob("modeling_custom.py"))
            self.assertTrue(custom_files)
            self.assertTrue(list(artifact.rglob("configuration_custom.py")))
            expected = pd.DataFrame(
                mlflow.pyfunc.load_model(str(artifact)).predict(example)
            ).to_dict(orient="list")
            expected_file = root / "expected.json"
            expected_file.write_text(json.dumps(expected))
            shutil.rmtree(source)
            script = """
import json, sys
import mlflow.pyfunc, mlflow.transformers
import pandas as pd
native = mlflow.transformers.load_model(sys.argv[1])
assert type(native.model).__name__ == 'CustomBertForSequenceClassification'
assert native.model.config.logit_temperature == 1.7
actual = pd.DataFrame(mlflow.pyfunc.load_model(sys.argv[1]).predict(pd.DataFrame({'text':['hello world']})))
expected = pd.DataFrame(json.load(open(sys.argv[2])))
pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-5, atol=1e-6)
"""
            environment = {
                **os.environ,
                "HF_HOME": str(root / "fresh-cache"),
                "HF_MODULES_CACHE": str(root / "fresh-modules"),
            }
            subprocess.run(
                [sys.executable, "-c", script, str(artifact), str(expected_file)],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
                timeout=90,
            )
            for file in custom_files:
                file.unlink()
            environment["HF_MODULES_CACHE"] = str(root / "empty-modules")
            failed = subprocess.run(
                [sys.executable, "-c", script, str(artifact), str(expected_file)],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                timeout=90,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertIn("modeling_custom", failed.stderr)
