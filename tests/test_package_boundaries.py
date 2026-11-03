"""Validate installed core metadata, independently of the Control integration."""

import importlib.metadata
import unittest


class CorePackageTests(unittest.TestCase):
    def test_core_has_no_control_dependency_or_deployment_entry_point(self):
        core = importlib.metadata.distribution("mlflow-triton")
        self.assertFalse(
            any("mlflow-triton-control" in item.lower() for item in core.requires or [])
        )
        self.assertFalse(
            any(item.lower().startswith("tritonclient") for item in core.requires or [])
        )
        self.assertFalse(
            any(ep.group == "mlflow.deployments" for ep in core.entry_points)
        )
        self.assertFalse(
            any("mlflow_triton_control" in str(f) for f in core.files or [])
        )

    def test_core_exposes_framework_logging_wrappers(self):
        # Act
        from mlflow_triton import pytorch, sklearn, transformers

        # Assert
        for module in [pytorch, sklearn, transformers]:
            self.assertTrue(callable(module.log_model))
