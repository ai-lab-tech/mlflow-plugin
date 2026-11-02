"""Compatibility import; new applications should use mlflow_triton.transformers."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("mlflow_triton.transformers")
