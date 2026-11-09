"""Dependencies are observed without reloading a model or freezing training."""

import cProfile
import importlib.util
import sys
import unittest
import weakref
from importlib import metadata
from pathlib import Path
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
        self.assertIn("model_dependency.native", dependencies.modules)
        self.assertNotIn("training_only", dependencies.modules)

    def test_python_function_imported_before_capture_is_observed(self):
        namespace = {"__name__": "custom_package.predictor"}
        exec("def predict():\n    return 42\n", namespace)
        dependencies = PredictionDependencies()
        with dependencies.capture():
            self.assertEqual(namespace["predict"](), 42)
        self.assertIn("custom_package.predictor", dependencies.modules)

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

    def test_setuptools_is_not_dropped_when_prediction_uses_pkg_resources(self):
        dependencies = PredictionDependencies()
        dependencies.modules = {"pkg_resources"}
        with (
            patch(
                "mlflow_triton._dependencies.metadata.packages_distributions",
                return_value={"pkg_resources": ["setuptools"]},
            ),
            patch("mlflow_triton._dependencies.metadata.version", return_value="80.0"),
        ):
            self.assertEqual(dependencies.requirements(), ["setuptools==80.0"])

    def test_native_profiler_is_restored(self):
        def predict():
            return 42

        profiler = cProfile.Profile()
        profiler.enable()
        try:
            with PredictionDependencies().capture():
                predict()
            self.assertIs(sys.getprofile(), profiler)
        finally:
            profiler.disable()
        self.assertIn(predict.__code__, [entry.code for entry in profiler.getstats()])

    def test_unexecuted_training_branch_is_not_captured(self):
        namespace = {"training_only": ModuleType("training_only")}
        exec(
            "def predict(training=False):\n"
            "    if training:\n"
            "        return training_only.__name__\n"
            "    return 42\n",
            namespace,
        )
        dependencies = PredictionDependencies()
        with dependencies.capture():
            self.assertEqual(namespace["predict"](), 42)
        self.assertNotIn("training_only", dependencies.modules)

    def test_paired_local_loads_capture_both_dependencies(self):
        def predict(first, second):
            return first, second

        dependencies = PredictionDependencies()
        first, second = ModuleType("first_dependency"), ModuleType("second_dependency")
        with dependencies.capture():
            predict(first, second)
        self.assertTrue(
            {"first_dependency", "second_dependency"} <= dependencies.modules
        )

    def test_optional_relative_import_failure_does_not_disable_capture(self):
        namespace = {"__package__": "package", "dependency": ModuleType("dependency")}
        exec(
            "def predict():\n"
            "    try:\n"
            "        from ... import nonexistent\n"
            "    except ImportError:\n"
            "        return dependency.__name__\n",
            namespace,
        )
        dependencies = PredictionDependencies()
        with dependencies.capture():
            self.assertEqual(namespace["predict"](), "dependency")
        self.assertIn("dependency", dependencies.modules)

    def test_changed_global_is_captured_on_each_call(self):
        namespace = {"dependency": ModuleType("first_dependency")}
        exec("def predict():\n    return dependency.__name__\n", namespace)
        dependencies = PredictionDependencies()
        with dependencies.capture():
            namespace["predict"]()
            namespace["dependency"] = ModuleType("second_dependency")
            namespace["predict"]()
        self.assertTrue(
            {"first_dependency", "second_dependency"} <= dependencies.modules
        )

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow is optional")
    def test_prebound_native_method_passed_as_argument_is_captured(self):
        import pyarrow

        callback = pyarrow.scalar(3).as_py

        def predict(function):
            return function()

        dependencies = PredictionDependencies()
        with dependencies.capture():
            self.assertEqual(predict(callback), 3)
        self.assertIn(
            "pyarrow==" + metadata.version("pyarrow"), dependencies.requirements()
        )

    def test_cached_scipy_ufunc_is_not_misidentified_as_numpy_only(self):
        from scipy.special import expit

        def predict(function):
            return function(0)

        dependencies = PredictionDependencies()
        with dependencies.capture():
            self.assertEqual(predict(expit), 0.5)
        self.assertIn(
            "scipy==" + metadata.version("scipy"), dependencies.requirements()
        )

    def test_training_module_reexporting_a_ufunc_is_not_captured(self):
        from scipy.special import expit

        training = ModuleType("training_only")
        training.__file__ = "/training_only.py"
        training.expit = expit

        def predict(function):
            return function(0)

        dependencies = PredictionDependencies()
        with patch.dict(sys.modules, {"training_only": training}):
            with dependencies.capture():
                predict(expit)
        self.assertNotIn("training_only", dependencies.modules)
        self.assertIn(
            "scipy==" + metadata.version("scipy"), dependencies.requirements()
        )

    def test_capture_does_not_invoke_user_attribute_hooks(self):
        class HostileModule(ModuleType):
            def __getattribute__(self, name):
                raise AssertionError("capture must not call user code")

        class HostileObject:
            def __getattribute__(self, name):
                raise AssertionError("capture must not call user code")

        module = HostileModule("custom_dependency")
        obj = HostileObject()

        def predict():
            return module is not None and obj is not None

        dependencies = PredictionDependencies()
        with dependencies.capture():
            self.assertTrue(predict())
        self.assertIn("custom_dependency", dependencies.modules)

    def test_shared_namespace_selects_only_the_distribution_owning_the_module(self):
        namespace = ModuleType("shared_namespace")
        module = ModuleType("shared_namespace.serving")
        module.__file__ = "/site/shared_namespace/serving/__init__.py"
        namespace.serving = module

        def predict():
            return namespace.serving.__name__

        dependencies = PredictionDependencies()
        with dependencies.capture():
            predict()

        def distribution(name):
            class Distribution:
                files = [metadata.PackagePath(f"shared_namespace/{name}/__init__.py")]

                def locate_file(self, path):
                    return Path("/site") / path

            return Distribution()

        with (
            patch.dict(
                sys.modules,
                {"shared_namespace": namespace, "shared_namespace.serving": module},
            ),
            patch(
                "mlflow_triton._dependencies.metadata.packages_distributions",
                return_value={"shared_namespace": ["serving", "training"]},
            ),
            patch(
                "mlflow_triton._dependencies.metadata.distribution",
                side_effect=distribution,
            ),
            patch("mlflow_triton._dependencies.metadata.version", return_value="1.2"),
        ):
            self.assertEqual(dependencies.requirements(), ["serving==1.2"])

    def test_existing_trace_receives_events_and_is_restored_after_failure(self):
        events = []

        def previous(frame, event, arg):
            if frame.f_code.co_name == "predict":
                events.append(event)
            return previous

        def predict():
            raise ValueError("prediction failed")

        original = sys.gettrace()
        sys.settrace(previous)
        try:
            with self.assertRaisesRegex(ValueError, "prediction failed"):
                with PredictionDependencies().capture():
                    predict()
            self.assertIs(sys.gettrace(), previous)
            self.assertTrue({"call", "line", "exception", "return"} <= set(events))
            self.assertNotIn("opcode", events)
        finally:
            sys.settrace(original)

    def test_completed_prediction_does_not_retain_intermediates(self):
        references = []

        class Intermediate:
            pass

        def predict():
            intermediate = Intermediate()
            references.append(weakref.ref(intermediate))
            return 1

        with PredictionDependencies().capture():
            predict()
            self.assertIsNone(references[0]())

    def test_cached_import_is_captured_even_when_not_used_after_import(self):
        namespace = {}
        exec("def predict():\n    import cached_dependency\n    return 42\n", namespace)
        dependencies = PredictionDependencies()
        with patch.dict(
            sys.modules, {"cached_dependency": ModuleType("cached_dependency")}
        ):
            with dependencies.capture():
                namespace["predict"]()
        self.assertIn("cached_dependency", dependencies.modules)

    def test_relative_cached_import_captures_actual_submodule(self):
        package = ModuleType("shared_namespace")
        package.serving = ModuleType("shared_namespace.serving")
        namespace = {"__package__": "shared_namespace"}
        exec("def predict():\n    from . import serving\n    return 42\n", namespace)
        dependencies = PredictionDependencies()
        with patch.dict(
            sys.modules,
            {"shared_namespace": package, "shared_namespace.serving": package.serving},
        ):
            with dependencies.capture():
                namespace["predict"]()
        self.assertIn("shared_namespace.serving", dependencies.modules)

    def test_failed_optional_import_is_not_captured(self):
        def predict():
            try:
                import nonexistent_prediction_dependency  # noqa: F401
            except ImportError:
                return 42

        dependencies = PredictionDependencies()
        with dependencies.capture():
            self.assertEqual(predict(), 42)
        self.assertNotIn("nonexistent_prediction_dependency", dependencies.modules)

    def test_dynamic_cached_import_is_captured(self):
        from importlib import import_module

        def predict():
            import_module("cached_dependency")

        dependencies = PredictionDependencies()
        with patch.dict(
            sys.modules, {"cached_dependency": ModuleType("cached_dependency")}
        ):
            with dependencies.capture():
                predict()
        self.assertIn("cached_dependency", dependencies.modules)

    def test_suspended_generator_restores_trace_flags(self):
        def predict():
            yield 1
            yield 2

        generator = predict()
        with PredictionDependencies().capture():
            self.assertEqual(next(generator), 1)
        self.assertFalse(generator.gi_frame.f_trace_opcodes)
        self.assertIsNone(generator.gi_frame.f_trace)
        self.assertEqual(next(generator), 2)
