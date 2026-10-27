"""HTTP example serialization without sending requests or starting Triton."""

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow_triton._request import write_request_example
from mlflow_triton._signature import contract_from_signature


class RequestExampleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repository = self.root / "model files"
        self.repository.mkdir()

    def test_json_contains_actual_shapes_values_and_all_output_names(self):
        # Arrange
        example = pd.DataFrame(
            {
                "customer age": np.array([28, 42], dtype=np.int64),
                "city": ["München", "Berlin"],
            }
        )
        output = pd.DataFrame({"prediction": [1.0, 2.0], "confidence": [0.8, 0.9]})
        contract = contract_from_signature(infer_signature(example, output), "sklearn")

        # Act
        write_request_example(self.repository, example, contract)

        # Assert
        payload = json.loads((self.repository / "request.json").read_text())
        self.assertEqual(
            payload["inputs"],
            [
                {
                    "name": "input_0",
                    "shape": [2],
                    "datatype": "INT64",
                    "data": [28, 42],
                },
                {
                    "name": "city",
                    "shape": [2],
                    "datatype": "BYTES",
                    "data": ["München", "Berlin"],
                },
            ],
        )
        self.assertEqual(
            payload["outputs"], [{"name": "prediction"}, {"name": "confidence"}]
        )
        self.assertEqual({p.name for p in self.repository.iterdir()}, {"request.json"})

    def test_named_tensor_values_are_flattened_in_row_major_order(self):
        # Arrange
        example = {"image": np.arange(6, dtype=np.float32).reshape(1, 2, 3)}
        contract = contract_from_signature(
            infer_signature(example, np.ones((1, 1), np.float32)), "pytorch"
        )

        # Act
        write_request_example(self.repository, example, contract)

        # Assert
        tensor = json.loads((self.repository / "request.json").read_text())["inputs"][0]
        self.assertEqual(tensor["shape"], [1, 2, 3])
        self.assertEqual(tensor["data"], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(tensor["datatype"], "FP32")

    def test_binary_request_preserves_bytes_and_nonfinite_values(self):
        # Arrange
        example = pd.DataFrame(
            {"blob": [b"\xff\x00", b"abc"], "value": [np.nan, np.inf]}
        )
        # Explicit required columns: inferring from NaN marks the column optional,
        # which this serving adapter intentionally does not support.
        signature = infer_signature(
            pd.DataFrame({"blob": [b"a", b"b"], "value": [1.0, 2.0]}), np.ones(2)
        )
        contract = contract_from_signature(signature, "sklearn")

        # Act
        write_request_example(self.repository, example, contract)

        # Assert
        headers = dict(
            line.split(": ", 1)
            for line in (self.repository / "request.headers").read_text().splitlines()
        )
        self.assertEqual(headers["Content-Type"], "application/octet-stream")
        offset = int(headers["Inference-Header-Content-Length"])
        body = (self.repository / "request.bin").read_bytes()
        payload = json.loads(body[:offset])
        expected_bytes = (
            struct.pack("<I", 2) + b"\xff\x00" + struct.pack("<I", 3) + b"abc"
        )
        self.assertEqual(body[offset : offset + len(expected_bytes)], expected_bytes)
        numeric = np.frombuffer(body[offset + len(expected_bytes) :], dtype="<f8")
        np.testing.assert_array_equal(numeric, [np.nan, np.inf])
        self.assertEqual(
            payload["inputs"][0]["parameters"]["binary_data_size"], len(expected_bytes)
        )
        self.assertEqual(payload["inputs"][1]["parameters"]["binary_data_size"], 16)
        self.assertEqual(payload["outputs"][0]["parameters"], {"binary_data": True})
        self.assertFalse((self.repository / "request.json").exists())
        self.assertEqual(
            {p.name for p in self.repository.iterdir()},
            {"request.bin", "request.headers"},
        )
