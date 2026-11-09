"""Capture dependencies of the existing prediction without reloading weights."""

import cProfile
import sys
from contextlib import contextmanager
from importlib import metadata
from types import BuiltinFunctionType, FunctionType, MethodType, ModuleType

from packaging.utils import canonicalize_name


class PredictionDependencies:
    """Observe executed modules and cached globals on the prediction thread.

    This is sample-based, like MLflow's dependency inference: unexecuted paths
    and work delegated to other threads/processes require explicit requirements.
    No model objects or tensor data are copied or retained.
    """

    def __init__(self):
        # The copied backend imports both, regardless of the model flavor.
        self.modules = {"numpy", "pandas"}
        self._seen = set()

    def _record(self, name):
        if isinstance(name, str):
            self.modules.add(name.split(".", 1)[0])

    def _record_value(self, value):
        if isinstance(value, ModuleType):
            self._record(value.__name__)
        elif isinstance(value, type):
            self._record(value.__module__)
        else:
            # Functions, extension callables and instances can be cached in
            # globals/closures before capture starts. Do not scan sys.modules.
            self._record(type(value).__module__)
            if isinstance(value, (FunctionType, BuiltinFunctionType, MethodType)):
                self._record(value.__module__)

    def _profile(self, frame, event, arg):
        if event not in {"call", "c_call"}:
            return
        module = frame.f_globals.get("__name__", "")
        if module.split(".", 1)[0] in {"mlflow", "mlflow_triton"}:
            return
        self._record(module)
        if event == "c_call":
            self._record_value(arg)
            return
        code = frame.f_code
        key = (code, id(frame.f_globals))
        if key not in self._seen:
            self._seen.add(key)
            for name in code.co_names:
                if name in frame.f_globals:
                    self._record_value(frame.f_globals[name])
        # Closures may bind different dependencies on each call.
        for name in code.co_freevars:
            if name in frame.f_locals:
                self._record_value(frame.f_locals[name])

    @contextmanager
    def capture(self):
        previous = sys.getprofile()
        native_profiler = isinstance(previous, cProfile.Profile)

        def profile(frame, event, arg):
            self._profile(frame, event, arg)
            if previous is not None and not native_profiler:
                previous(frame, event, arg)

        sys.setprofile(profile)
        try:
            yield
        finally:
            if native_profiler:
                previous.enable()
            else:
                sys.setprofile(previous)
            self._seen.clear()

    def requirements(self):
        """Pin observed installed distributions; ignore stdlib and bundled code."""
        providers = metadata.packages_distributions()
        packages = {
            canonicalize_name(package)
            for module in self.modules - {"mlflow", "mlflow_triton"}
            for package in providers.get(module, ())
        }
        packages -= {"mlflow", "mlflow-skinny", "mlflow-triton", "setuptools"}
        return [
            f"{package}=={metadata.version(package)}" for package in sorted(packages)
        ]
