"""Dependencies are observed without reloading a model or freezing training."""

import cProfile
import sys
import unittest
from types import ModuleType
from unittest.mock import patch

from mlflow_triton._dependencies import PredictionDependencies


class PredictionDependencyTests(unittest.TestCase):
    def test_cached_module_and_native_callable_are_captured_without_training_packages(
        self,
    ):
        used = ModuleType("model_dependency.native")
        unused = ModuleType("training_only")
        namespace = {"used": used, "unused": unused, "native": len}
        exec("def predict():\n    return used.__name__, native([1])\n", namespace)
        dependencies = PredictionDependencies()
        with dependencies.capture():
            self.assertEqual(namespace["predict"](), ("model_dependency.native", 1))
        self.assertIn("model_dependency", dependencies.modules)
        self.assertNotIn("training_only", dependencies.modules)

    def test_python_function_imported_before_capture_is_observed(self):
        namespace = {"__name__": "custom_package.predictor"}
        exec("def predict():\n    return 42\n", namespace)
        dependencies = PredictionDependencies()
        with dependencies.capture():
            self.assertEqual(namespace["predict"](), 42)
        self.assertIn("custom_package", dependencies.modules)

    def test_cached_closure_dependency_is_observed(self):
        module = ModuleType("closure_dependency")

        def predict():
            return module.__name__

        dependencies = PredictionDependencies()
        with dependencies.capture():
            predict()
        self.assertIn("closure_dependency", dependencies.modules)

    def test_installed_versions_are_pinned_without_exporting_entire_environment(self):
        dependencies = PredictionDependencies()
        dependencies.modules = {"custom", "os", "mlflow", "mlflow_triton"}
        with (
            patch(
                "mlflow_triton._dependencies.metadata.packages_distributions",
                return_value={
                    "custom": ["Custom-Package"],
                    "training": ["training-only"],
                    "mlflow": ["mlflow"],
                },
            ),
            patch(
                "mlflow_triton._dependencies.metadata.version", return_value="1.2+cpu"
            ),
        ):
            self.assertEqual(dependencies.requirements(), ["custom-package==1.2+cpu"])

    def test_existing_profiler_is_called_and_restored_after_failure(self):
        events = []

        def previous(frame, event, arg):
            events.append(event)

        original = sys.getprofile()
        sys.setprofile(previous)
        try:
            with self.assertRaisesRegex(ValueError, "prediction failed"):
                with PredictionDependencies().capture():
                    raise ValueError("prediction failed")
            self.assertIs(sys.getprofile(), previous)
            self.assertTrue(events)
        finally:
            sys.setprofile(original)

    def test_native_profiler_is_restored(self):
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            with PredictionDependencies().capture():
                len([1])
            self.assertIs(sys.getprofile(), profiler)
        finally:
            profiler.disable()
