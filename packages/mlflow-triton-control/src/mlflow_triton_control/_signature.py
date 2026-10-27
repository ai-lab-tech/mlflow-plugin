"""Compatibility import; new applications should use mlflow_triton._signature."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("mlflow_triton._signature")
