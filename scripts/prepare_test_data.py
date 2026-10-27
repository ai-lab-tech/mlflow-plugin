"""Prepare pinned public/private fixtures outside the repository; never run training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "examples/test_catalog.json"


def safe_child(root, relative):
    path = Path(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe catalog path: {relative}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes dataset root: {relative}")
    return root / path


def verify_case(directory, case):
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError(f"Missing case directory: {directory}")
    for relative, expected in case["files"].items():
        path = safe_child(directory, relative)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing regular artifact: {path}")
        if path.stat().st_size != expected["size"]:
            raise ValueError(
                f"Artifact size mismatch (Git LFS files may be missing): {path}"
            )
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {path}")


def git(directory, *arguments):
    return subprocess.check_output(
        ["git", "-C", str(directory), *arguments], text=True
    ).strip()


def private_checkout(source, supplied, cache, selected):
    revision = source["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Private data must be pinned to a complete Git commit SHA")
    if supplied:
        checkout = supplied.resolve(strict=True)
        if git(checkout, "rev-parse", "HEAD") != revision:
            raise ValueError(f"Private checkout must be at pinned revision {revision}")
        # Checksums below validate working-tree data; never switch or modify a user's checkout.
        return checkout
    checkout = cache / revision
    if checkout.exists():
        if git(checkout, "rev-parse", "HEAD") != revision:
            raise ValueError("Cached private checkout has an unexpected revision")
    else:
        if not source.get("url"):
            raise ValueError(
                "Configure the private repository URL or pass --private-checkout"
            )
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="checkout-", dir=cache) as temporary:
            stage = Path(temporary) / "repo"
            environment = {
                **os.environ,
                "GIT_LFS_SKIP_SMUDGE": "1",
                "GIT_TERMINAL_PROMPT": "0",
            }
            subprocess.run(
                ["git", "clone", "--no-checkout", source["url"], str(stage)],
                env=environment,
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(stage), "checkout", "--detach", revision],
                env=environment,
                check=True,
            )
            stage.rename(checkout)
    # Configure only our managed cache, never global Git settings or user checkouts.
    subprocess.run(
        ["git", "-C", str(checkout), "lfs", "install", "--local"], check=True
    )
    include = ",".join(f"{source['path']}/{name}/**" for name in selected)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "lfs",
            "pull",
            "--include",
            include,
            "--exclude",
            "",
        ],
        check=True,
    )
    return checkout


def prepare(
    catalog_path, data_root, include_private=False, private_path=None, selected=None
):
    catalog_path = catalog_path.resolve()
    catalog = json.loads(catalog_path.read_text())
    if catalog.get("schema_version") != 1:
        raise ValueError("Unsupported test catalog version")
    cases = catalog["cases"]
    names = list(
        dict.fromkeys(
            selected
            or [
                n
                for n, c in cases.items()
                if include_private or c["source"] == "public"
            ]
        )
    )
    for name in names:
        if name not in cases:
            raise ValueError(f"Unknown case: {name}")
        if cases[name]["source"] == "private" and not include_private:
            raise ValueError(f"{name} requires --include-private")
    data_root = data_root.expanduser().resolve()
    if data_root == ROOT or data_root.is_relative_to(ROOT):
        raise ValueError("--data-root must be outside the plugin repository")
    # Avoid creating a destination inside either source.
    public_root = safe_child(catalog_path.parent, catalog["sources"]["public"]["path"])
    if data_root == public_root.resolve() or data_root.is_relative_to(
        public_root.resolve()
    ):
        raise ValueError("Destination must be outside source data")
    needed_private = [n for n in names if cases[n]["source"] == "private"]
    private_root = None
    if needed_private:
        checkout = private_checkout(
            catalog["sources"]["private"],
            private_path,
            data_root / ".sources/private",
            needed_private,
        )
        private_root = safe_child(checkout, catalog["sources"]["private"]["path"])
        if data_root == checkout or data_root.is_relative_to(checkout):
            raise ValueError(
                "Destination must be outside the supplied private checkout"
            )
    plans = []
    # Check all selected sources before copying anything; never overwrite differing data.
    for name in names:
        case = cases[name]
        source_root = public_root if case["source"] == "public" else private_root
        source = safe_child(source_root, case["dataset_directory"])
        destination = safe_child(data_root, case["dataset_directory"])
        verify_case(source, case)
        if destination.exists():
            verify_case(destination, case)
        plans.append((name, source, destination))
    data_root.mkdir(parents=True, exist_ok=True)
    for name, source, destination in plans:
        if not destination.exists():
            with tempfile.TemporaryDirectory(
                prefix=".prepare-", dir=data_root
            ) as temporary:
                stage = Path(temporary) / "case"
                stage.mkdir()
                # Copy only manifest entries, not arbitrary untracked files.
                for relative in cases[name]["files"]:
                    target = safe_child(stage, relative)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(safe_child(source, relative), target)
                verify_case(stage, cases[name])
                stage.rename(destination)
        print(
            f"ready: {name} ({cases[name]['source']}, {cases[name]['license_status']})"
        )
    # Index includes existing valid cases too; no model result or pass rate is implied.
    available = []
    for name, case in cases.items():
        if (data_root / case["dataset_directory"]).exists():
            verify_case(data_root / case["dataset_directory"], case)
            available.append(name)
    state = {
        "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
        "private_revision": catalog["sources"]["private"]["revision"]
        if any(cases[n]["source"] == "private" for n in available)
        else None,
        "available_cases": available,
    }
    with tempfile.NamedTemporaryFile(
        mode="w", dir=data_root, prefix=".state-", delete=False
    ) as stream:
        json.dump(state, stream, indent=2)
        stream.write("\n")
        temporary_state = Path(stream.name)
    temporary_state.replace(data_root / "dataset-state.json")
    return available


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--include-private", action="store_true")
    parser.add_argument("--private-checkout", type=Path)
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    if args.private_checkout and not args.include_private:
        parser.error("--private-checkout requires --include-private")
    try:
        available = prepare(
            args.catalog,
            args.data_root,
            args.include_private,
            args.private_checkout,
            args.case,
        )
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Data preparation failed: {error}\n")
    print(f"Verified {len(available)} available cases in {args.data_root}")


if __name__ == "__main__":
    main()
