"""Dataset preparation uses verified files and never silently replaces artifacts."""

import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "data_preparation", ROOT / "scripts/prepare_test_data.py"
)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class DataPreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.public = self.root / "data/public_case"
        self.public.mkdir(parents=True)
        (self.public / "model.bin").write_bytes(b"public")
        self.output = self.root / "output"
        self.catalog_path = self.root / "catalog.json"
        self.catalog = {
            "schema_version": 1,
            "sources": {
                "public": {"path": "data"},
                "private": {"revision": "a" * 40, "path": "data", "url": "unused"},
            },
            "cases": {
                name: {
                    "source": source,
                    "license_status": "documented_synthetic"
                    if source == "public"
                    else "needs_review",
                    "dataset_directory": name,
                    "files": {
                        "model.bin": {
                            "size": len(value),
                            "sha256": hashlib.sha256(value).hexdigest(),
                        }
                    },
                }
                for name, source, value in [
                    ("public_case", "public", b"public"),
                    ("private_case", "private", b"private"),
                ]
            },
        }
        self.save_catalog()

    def save_catalog(self):
        self.catalog_path.write_text(json.dumps(self.catalog))

    def test_public_only_never_accesses_private_remote_and_is_repeatable(self):
        # Arrange
        with patch.object(prepare, "private_checkout") as private:
            # Act
            first = prepare.prepare(self.catalog_path, self.output)
            second = prepare.prepare(self.catalog_path, self.output)

        # Assert
        private.assert_not_called()
        self.assertEqual(first, ["public_case"])
        self.assertEqual(second, first)
        self.assertEqual(
            (self.output / "public_case/model.bin").read_bytes(), b"public"
        )
        self.assertFalse((self.output / "private_case").exists())

    def test_corrupt_source_fails_before_materialization(self):
        # Arrange
        (self.public / "model.bin").write_bytes(b"broken")

        # Act / Assert
        with self.assertRaisesRegex(ValueError, "checksum"):
            prepare.prepare(self.catalog_path, self.output)
        self.assertFalse(self.output.exists())

    def test_existing_different_destination_is_not_overwritten(self):
        # Arrange
        folder = self.output / "public_case"
        folder.mkdir(parents=True)
        (folder / "model.bin").write_bytes(b"custom")

        # Act / Assert
        with self.assertRaisesRegex(ValueError, "checksum"):
            prepare.prepare(self.catalog_path, self.output)
        self.assertEqual((folder / "model.bin").read_bytes(), b"custom")

    def test_private_case_requires_explicit_opt_in(self):
        # Arrange / Act / Assert
        with self.assertRaisesRegex(ValueError, "requires --include-private"):
            prepare.prepare(self.catalog_path, self.output, selected=["private_case"])

    def test_pinned_local_checkout_combines_both_sources(self):
        # Arrange
        checkout = self.root / "private-repo"
        folder = checkout / "data/private_case"
        folder.mkdir(parents=True)
        (folder / "model.bin").write_bytes(b"private")

        def git(*args):
            return subprocess.check_output(
                ["git", "-C", str(checkout), *args], text=True
            ).strip()

        git("init", "-b", "main")
        git("add", ".")
        git(
            "-c",
            "user.name=Test Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-m",
            "fixture",
        )
        revision = git("rev-parse", "HEAD")
        self.catalog["sources"]["private"]["revision"] = revision
        self.save_catalog()

        # Act
        available = prepare.prepare(self.catalog_path, self.output, True, checkout)

        # Assert
        self.assertEqual(set(available), {"public_case", "private_case"})
        state = json.loads((self.output / "dataset-state.json").read_text())
        self.assertEqual(state["private_revision"], revision)
        self.assertEqual(
            (self.output / "private_case/model.bin").read_bytes(), b"private"
        )

    def test_wrong_private_revision_is_rejected(self):
        # Arrange
        with patch.object(prepare, "git", return_value="b" * 40):
            # Act / Assert
            with self.assertRaisesRegex(ValueError, "pinned revision"):
                prepare.private_checkout(
                    self.catalog["sources"]["private"],
                    self.root,
                    self.output,
                    ["private_case"],
                )

    def test_path_traversal_is_rejected(self):
        # Arrange / Act / Assert
        with self.assertRaisesRegex(ValueError, "Unsafe catalog path"):
            prepare.safe_child(self.root, "../outside")

    def test_public_tree_has_no_private_case_directories(self):
        # Arrange
        catalog = json.loads((ROOT / "examples/test_catalog.json").read_text())
        private = [
            case for case in catalog["cases"].values() if case["source"] == "private"
        ]

        # Act
        present = [
            case["dataset_directory"]
            for case in private
            if (ROOT / "examples/data" / case["dataset_directory"]).exists()
        ]

        # Assert
        self.assertEqual(present, [])
