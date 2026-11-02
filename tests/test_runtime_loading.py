"""Environment isolation for concurrent model loads; no real models loaded."""

import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from mlflow_triton import _runtime


class RuntimeLoadingTests(unittest.TestCase):
    def test_failed_load_restores_environment(self):
        # Arrange
        keys = [
            "MLFLOW_DEFAULT_PREDICTION_DEVICE",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "MLFLOW_HUGGINGFACE_USE_DEVICE_MAP",
        ]
        with patch.dict(os.environ, {keys[0]: "original"}):
            os.environ.pop("HF_HUB_OFFLINE", None)
            before = {key: os.environ.get(key) for key in keys}
            with patch(
                "mlflow.pyfunc.load_model", side_effect=RuntimeError("cannot load")
            ):
                # Act / Assert
                with self.assertRaisesRegex(RuntimeError, "cannot load"):
                    _runtime.load_predictor("unused", "transformers", "cuda:1")
            self.assertEqual({key: os.environ.get(key) for key in keys}, before)

    def test_concurrent_loads_serialize_device_overrides(self):
        # Arrange
        entered = threading.Event()
        attempted_second_lock = threading.Event()
        release = threading.Event()
        observed = []
        real_lock = threading.RLock()

        class ObservedLock:
            def __enter__(self):
                if threading.current_thread().name.endswith("_1"):
                    attempted_second_lock.set()
                real_lock.acquire()

            def __exit__(self, *args):
                real_lock.release()

        def load(path):
            observed.append((path, os.environ["MLFLOW_DEFAULT_PREDICTION_DEVICE"]))
            if path == "first":
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("Test failed to release first loader")
            return path

        with patch.dict(os.environ, {"MLFLOW_DEFAULT_PREDICTION_DEVICE": "original"}):
            with (
                patch.object(_runtime, "_LOAD_LOCK", ObservedLock()),
                patch("mlflow.pyfunc.load_model", side_effect=load),
            ):
                with ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="loader"
                ) as pool:
                    # Act
                    first = pool.submit(
                        _runtime.load_predictor, "first", "transformers", "cuda:1"
                    )
                    try:
                        if not entered.wait(5):
                            raise TimeoutError("First loader did not start")
                        second = pool.submit(
                            _runtime.load_predictor, "second", "pytorch", "cpu"
                        )
                        if not attempted_second_lock.wait(5):
                            raise TimeoutError("Second loader did not reach the lock")
                        while_first_loading = list(observed)
                    finally:
                        release.set()
                    results = [first.result(timeout=5), second.result(timeout=5)]

            # Assert
            self.assertEqual(while_first_loading, [("first", "1")])
            self.assertEqual(observed, [("first", "1"), ("second", "cpu")])
            self.assertEqual(results, ["first", "second"])
            self.assertEqual(os.environ["MLFLOW_DEFAULT_PREDICTION_DEVICE"], "original")
