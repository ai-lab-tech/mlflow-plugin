"""Standalone serving requirements preserve restrictions without sidecar files."""

import tempfile
import unittest
from pathlib import Path

from mlflow_triton._requirements import serving_requirements
from packaging.requirements import Requirement


class ServingRequirementsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_constraints_are_merged_and_constraint_only_packages_are_explicit(self):
        # Arrange
        (self.root / "requirements.txt").write_text(
            "numpy>=1.20\nDemo_Package[fast]>=1\n-c constraints.txt\n"
        )
        (self.root / "constraints.txt").write_text(
            "numpy<2\ndemo-package!=1.5\nscipy==1.12.0\n"
        )

        # Act
        text = serving_requirements(self.root, "3.14.0")
        packages = {item.name: item for item in map(Requirement, text.splitlines())}

        # Assert
        self.assertEqual(len(text.splitlines()), 4)
        self.assertIn("1.26.4", packages["numpy"].specifier)
        self.assertNotIn("2.0", packages["numpy"].specifier)
        self.assertNotIn("1.19", packages["numpy"].specifier)
        self.assertEqual(packages["demo-package"].extras, {"fast"})
        self.assertNotIn("1.5", packages["demo-package"].specifier)
        self.assertEqual(str(packages["scipy"].specifier), "==1.12.0")
        self.assertEqual(str(packages["mlflow"].specifier), "==3.14.0")
        self.assertNotIn("constraints.txt", text)

    def test_nested_includes_and_markers_remain_portable(self):
        # Arrange
        (self.root / "requirements.txt").write_text("-r nested/requirements.txt\n")
        (self.root / "nested").mkdir()
        (self.root / "nested/requirements.txt").write_text(
            'numpy>=1; python_version < "3.12"\n--constraint=limits.txt\n'
        )
        (self.root / "nested/limits.txt").write_text(
            'numpy<2; python_version < "3.12"\nnumpy>=2; python_version >= "3.12"\n'
        )

        # Act
        text = serving_requirements(self.root, "3.14.0")
        numpy = [
            item for item in map(Requirement, text.splitlines()) if item.name == "numpy"
        ]

        # Assert
        self.assertEqual(len(numpy), 2)
        older = next(
            item for item in numpy if item.marker.evaluate({"python_version": "3.11"})
        )
        newer = next(
            item for item in numpy if item.marker.evaluate({"python_version": "3.12"})
        )
        self.assertIn("1.26.4", older.specifier)
        self.assertNotIn("2.0", older.specifier)
        self.assertIn("2.0", newer.specifier)
        self.assertNotIn(".txt", text)

    def test_conflicting_versions_are_not_silently_dropped(self):
        # Arrange
        (self.root / "requirements.txt").write_text("numpy==2.0\n-c constraints.txt\n")
        (self.root / "constraints.txt").write_text("numpy<2\n")

        # Act
        packages = list(
            map(Requirement, serving_requirements(self.root, "3.14.0").splitlines())
        )

        # Assert: pip must see the conflict, not an arbitrarily selected version.
        numpy = next(item for item in packages if item.name == "numpy")
        self.assertEqual(str(numpy.specifier), "<2,==2.0")
        self.assertNotIn("2.0", numpy.specifier)

    def test_missing_or_cyclic_includes_fail_packaging(self):
        for content, error in [
            ("-c missing.txt", FileNotFoundError),
            ("-r requirements.txt", ValueError),
        ]:
            with self.subTest(content=content):
                # Arrange
                (self.root / "requirements.txt").write_text(content)

                # Act / Assert
                with self.assertRaises(error):
                    serving_requirements(self.root, "3.14.0")

    def test_nonportable_options_and_file_references_are_rejected(self):
        for content in [
            "--index-url https://example.test/simple",
            "demo @ file:///tmp/demo.whl",
            "-r ../outside.txt",
        ]:
            with self.subTest(content=content):
                # Arrange
                (self.root / "requirements.txt").write_text(content)

                # Act / Assert
                with self.assertRaises(ValueError):
                    serving_requirements(self.root, "3.14.0")

    def test_existing_mlflow_dependency_is_preserved(self):
        # Arrange
        (self.root / "requirements.txt").write_text("mlflow==3.14.0\n")

        # Act
        text = serving_requirements(self.root, "3.17.2")

        # Assert
        self.assertEqual(text, "mlflow==3.14.0\n")

    def test_conditional_mlflow_always_has_a_supported_unconditional_fallback(self):
        for package in ("mlflow", "mlflow-skinny"):
            for marker in ('python_version < "3.10"', 'python_version < "3.12"'):
                with self.subTest(package=package, marker=marker):
                    # Arrange
                    original = Requirement(f"{package}==3.14.0; {marker}")
                    (self.root / "requirements.txt").write_text(f"{original}\n")

                    # Act
                    text = serving_requirements(self.root, "3.17.2")
                    requirements = list(map(Requirement, text.splitlines()))

                    # Assert
                    self.assertIn(original, requirements)
                    fallback = next(
                        (
                            item
                            for item in requirements
                            if item.name == "mlflow" and item.marker is None
                        ),
                        None,
                    )
                    self.assertIsNotNone(fallback)
                    self.assertIn("3.14.0", fallback.specifier)
                    self.assertIn("3.17.2", fallback.specifier)
                    self.assertNotIn("4.0", fallback.specifier)
                    for python_version in ("3.11", "3.12"):
                        active = [
                            item
                            for item in requirements
                            if item.marker is None
                            or item.marker.evaluate({"python_version": python_version})
                        ]
                        self.assertTrue(any(item.name == "mlflow" for item in active))

    def test_unconditional_skinny_pin_is_preserved(self):
        # Arrange
        (self.root / "requirements.txt").write_text("mlflow-skinny==3.14.0\n")

        # Act
        text = serving_requirements(self.root, "3.17.2")

        # Assert
        self.assertEqual(text, "mlflow-skinny==3.14.0\n")
