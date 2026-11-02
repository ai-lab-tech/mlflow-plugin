"""Validate shared package versions and an optional release tag (Python 3.11+)."""

import argparse
import re
from pathlib import Path

import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    version = (root / "VERSION").read_text().strip()
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        parser.error("VERSION must contain a stable X.Y.Z release")
    for name in ["mlflow-triton", "mlflow-triton-control"]:
        project = tomllib.loads(
            (root / "packages" / name / "pyproject.toml").read_text()
        )["project"]
        if project["version"] != version or project["name"] != name:
            parser.error(f"Package/version mismatch: {name}")
        if (
            name == "mlflow-triton-control"
            and f"mlflow-triton=={version}" not in project["dependencies"]
        ):
            parser.error("Control must depend on the matching core release")
    if args.tag and args.tag != f"v{version}":
        parser.error(f"Expected tag v{version}, got {args.tag}")
    print(f"Both packages: {version}")


if __name__ == "__main__":
    main()
