"""Mapping round trips and validation against Triton's configuration fields."""

import copy
import json
import unittest

import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow_triton import _runtime
from mlflow_triton._signature import contract_from_signature, model_config


class ConfigMappingTests(unittest.TestCase):
    def configuration(self):
        example = pd.DataFrame(
            {"customer age": np.array([30], dtype=np.int64), "city": ["Berlin"]}
        )
        output = pd.DataFrame({"score": np.array([0.5], dtype=np.float32)})
        contract = contract_from_signature(infer_signature(example, output), "sklearn")
        return model_config("model", contract, "cpu"), example, output

    def test_mapping_preserves_input_and_output_names_and_values(self):
        # Arrange
        config, example, output = self.configuration()

        # Act
        contract = _runtime.contract_from_config(config)
        decoded = _runtime.decode_inputs(
            _runtime.example_tensors(example, contract), contract
        )
        encoded = _runtime.encode_outputs(output, contract)

        # Assert
        pd.testing.assert_frame_equal(decoded, example)
        np.testing.assert_array_equal(encoded["score"], output.score.to_numpy())
        self.assertEqual(contract["inputs"][0]["name"], "input_0")
        self.assertEqual(contract["inputs"][0]["source_name"], "customer age")
        mapping = json.loads(config["parameters"]["mlflow_contract"]["string_value"])
        for spec in mapping["inputs"] + mapping["outputs"]:
            self.assertFalse({"dims", "datatype", "numpy_dtype"} & spec.keys())

    def test_types_and_shapes_come_from_standard_config_fields(self):
        # Arrange
        config, _, _ = self.configuration()
        config["input"][0].update(data_type="TYPE_INT32", dims=["2"])

        # Act
        contract = _runtime.contract_from_config(config)

        # Assert
        self.assertEqual(contract["inputs"][0]["numpy_dtype"], "int32")
        self.assertEqual(contract["inputs"][0]["dims"], [2])

    def test_invalid_mapping_fails_clearly(self):
        # Arrange
        original, _, _ = self.configuration()

        def bad_version(config):
            mapping = json.loads(
                config["parameters"]["mlflow_contract"]["string_value"]
            )
            mapping["version"] = 99
            config["parameters"]["mlflow_contract"]["string_value"] = json.dumps(
                mapping
            )

        cases = {
            "missing_parameter": lambda c: c.pop("parameters"),
            "invalid_json": lambda c: c["parameters"]["mlflow_contract"].update(
                string_value="not-json"
            ),
            "unsupported_version": bad_version,
            "renamed_tensor": lambda c: c["input"][0].update(name="renamed"),
            "duplicate_tensor": lambda c: c["input"].append(c["input"][0]),
            "batching": lambda c: c.update(max_batch_size=8),
            "unknown_dtype": lambda c: c["input"][0].update(data_type="TYPE_INVALID"),
            "reshape": lambda c: c["input"][0].update(reshape={"shape": [1]}),
        }
        for name, mutate in cases.items():
            with self.subTest(case=name):
                config = copy.deepcopy(original)
                mutate(config)

                # Act / Assert
                with self.assertRaisesRegex(
                    ValueError, "Invalid Triton MLflow configuration"
                ):
                    _runtime.contract_from_config(config)
