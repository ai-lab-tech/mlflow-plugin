"""Old public imports remain usable with the optional integration installed."""

import importlib
import importlib.metadata
import unittest


class CompatibilityTests(unittest.TestCase):
    def test_legacy_modules_are_core_aliases(self):
        for name in ["sklearn", "pytorch", "transformers"]:
            self.assertIs(
                importlib.import_module("mlflow_triton_control." + name),
                importlib.import_module("mlflow_triton." + name),
            )

    def test_matching_versions_and_control_entry_point(self):
        core = importlib.metadata.distribution("mlflow-triton")
        control = importlib.metadata.distribution("mlflow-triton-control")
        self.assertEqual(core.version, control.version)
        entries = [e for e in control.entry_points if e.group == "mlflow.deployments"]
        self.assertEqual(
            [(e.name, e.value) for e in entries],
            [("triton-control", "mlflow_triton_control.deployment_client")],
        )
