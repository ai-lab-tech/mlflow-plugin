"""Old public imports remain usable with the optional integration installed."""

import importlib
import importlib.metadata
import unittest


class CompatibilityTests(unittest.TestCase):
    def test_legacy_modules_are_core_aliases(self):
        # Arrange
        names = ["sklearn", "pytorch", "transformers"]
        for name in names:
            with self.subTest(module=name):
                core = importlib.import_module("mlflow_triton." + name)

                # Act
                alias = importlib.import_module("mlflow_triton_control." + name)

                # Assert
                self.assertIs(alias, core)

    def test_matching_versions_and_control_entry_point(self):
        core = importlib.metadata.distribution("mlflow-triton")
        control = importlib.metadata.distribution("mlflow-triton-control")
        entries = [e for e in control.entry_points if e.group == "mlflow.deployments"]
        # Assert
        self.assertEqual(core.version, control.version)
        self.assertEqual(
            [(e.name, e.value) for e in entries],
            [("triton-control", "mlflow_triton_control.deployment_client")],
        )
