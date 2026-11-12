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

    def test_automatic_cpu_torch_pins_cannot_select_a_cuda_build(self):
        (self.root / "requirements.txt").write_text(
            "torch==2.14.1\ntorchvision==0.29.1\n"
            'torchaudio==2.9.0; python_version < "3.14"\n'
        )
        text = serving_requirements(self.root, "3.17.2", prefer_cpu_torch=True)
        requirements = {
            item.name: item
            for item in map(
                Requirement,
                (line for line in text.splitlines() if not line.startswith("-")),
            )
        }
        self.assertIn("2.14.1+cpu", requirements["torch"].specifier)
        self.assertNotIn("2.14.1", requirements["torch"].specifier)
        self.assertNotIn("2.14.1+cu130", requirements["torch"].specifier)
        self.assertEqual(str(requirements["torchvision"].specifier), "==0.29.1+cpu")
        self.assertIsNotNone(requirements["torchaudio"].marker)
        for package in ("torch", "torchvision", "torchaudio"):
            self.assertIn(
                f"--find-links https://download.pytorch.org/whl/cpu/{package}/", text
            )
        self.assertNotIn("--index-url", text)
        self.assertNotIn("--extra-index-url", text)

    def test_cpu_wheel_selection_preserves_public_version_and_other_constraints(self):
        (self.root / "requirements.txt").write_text(
            "torch==2.9.0+cu128\n-c constraints.txt\n"
        )
        (self.root / "constraints.txt").write_text("torch<3\n")
        text = serving_requirements(self.root, "3.17.2", prefer_cpu_torch=True)
        self.assertIn("torch<3,==2.9.0+cpu", text)
        self.assertIn(
            "torch==2.9.0+cu128", (self.root / "requirements.txt").read_text()
        )

    def test_explicit_or_gpu_requirements_keep_the_selected_build(self):
        (self.root / "requirements.txt").write_text("torch==2.14.1+cu130\n")
        text = serving_requirements(self.root, "3.17.2")
        self.assertIn("torch==2.14.1+cu130", text)
        self.assertNotIn("find-links", text)

    def test_automatic_gpu_requirements_do_not_reuse_an_installed_cpu_build(self):
        for version in ("2.14.1", "2.14.1+cpu", "2.14.1+cu130"):
            with self.subTest(version=version):
                (self.root / "requirements.txt").write_text(f"torch=={version}\n")
                text = serving_requirements(self.root, "3.17.2", exclude_cpu_torch=True)
                torch = next(
                    item
                    for item in map(Requirement, text.splitlines())
                    if item.name == "torch"
                )
                self.assertNotIn("2.14.1+cpu", torch.specifier)
                self.assertIn("2.14.1+cu130", torch.specifier)
                self.assertNotIn("find-links", text)

    def test_cpu_policy_does_not_override_direct_wheel_urls(self):
        original = "torch @ https://wheels.example.test/torch-custom.whl"
        (self.root / "requirements.txt").write_text(original + "\n")
        text = serving_requirements(self.root, "3.17.2", prefer_cpu_torch=True)
        self.assertEqual(Requirement(text.splitlines()[0]), Requirement(original))
        self.assertNotIn("find-links", text)

    def test_remote_wheel_sources_survive_flattening_and_are_deduplicated(self):
        source = "https://download.pytorch.org/whl/cpu/torch/"
        (self.root / "requirements.txt").write_text(
            f"--find-links={source}\n-r nested.txt\n"
        )
        (self.root / "nested.txt").write_text(f"-f {source}\ntorch==2.14.1+cpu\n")
        text = serving_requirements(self.root, "3.17.2", prefer_cpu_torch=True)
        self.assertEqual(text.count("--find-links"), 1)
        self.assertIn("torch==2.14.1+cpu", text)
        (self.root / "requirements.txt").write_text(text)
        self.assertEqual(serving_requirements(self.root, "3.17.2"), text)

    def test_local_wheel_sources_are_rejected(self):
        for source in ("/tmp/wheels", "file:///tmp/wheels"):
            with self.subTest(source=source):
                (self.root / "requirements.txt").write_text(f"--find-links {source}\n")
                with self.assertRaisesRegex(ValueError, "HTTP"):
                    serving_requirements(self.root, "3.17.2")

    def test_cpu_policy_does_not_hide_conflicting_framework_pins(self):
        for pins in (
            "torch==2.9.0\ntorch==2.10.0\n",
            "torch==2.9.0+cpu\ntorch==2.9.0+cu128\n",
        ):
            with self.subTest(pins=pins):
                (self.root / "requirements.txt").write_text(pins)
                text = serving_requirements(self.root, "3.17.2", prefer_cpu_torch=True)
                self.assertNotIn("find-links", text)
                self.assertIn(
                    ",",
                    next(
                        line for line in text.splitlines() if line.startswith("torch")
                    ),
                )
