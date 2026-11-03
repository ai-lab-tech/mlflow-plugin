"""Test installed core and optional integration independently (no live services)."""

import argparse
import importlib.util
import unittest
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", choices=["core", "control", "all"], required=True)
    parser.add_argument("--require-core-only", action="store_true")
    parser.add_argument("--fail-on-skip", action="store_true")
    args = parser.parse_args()
    if (
        args.require_core_only
        and importlib.util.find_spec("mlflow_triton_control") is not None
    ):
        parser.error(
            "Core-only validation must run without mlflow-triton-control installed"
        )
    tests = Path(__file__).resolve().parents[1] / "tests"
    suite = unittest.TestSuite()
    for path in sorted(tests.glob("test_*.py")):
        integration = path.name.startswith("test_deployment") or path.name in {
            "test_compatibility.py",
            "test_generated_requirements.py",
            "test_model_uri.py",
            "test_profile_selection.py",
        }
        if args.package != "all" and integration != (args.package == "control"):
            continue
        suite.addTests(
            unittest.defaultTestLoader.discover(str(tests), pattern=path.name)
        )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return not result.wasSuccessful() or (args.fail_on_skip and bool(result.skipped))


if __name__ == "__main__":
    raise SystemExit(main())
