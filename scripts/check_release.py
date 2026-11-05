"""Validate package versions or prepare them from a release tag (Python 3.11+)."""

import argparse
import re
from pathlib import Path

import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag")
    parser.add_argument(
        "--write-version",
        action="store_true",
        help="Set VERSION and package metadata from --tag before building",
    )
    args = parser.parse_args()
    if args.write_version and not args.tag:
        parser.error("--write-version requires --tag")
    if args.tag and not re.fullmatch(
        r"v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", args.tag
    ):
        parser.error("Tag must be a stable vX.Y.Z release")
    root = Path(__file__).resolve().parents[1]
    version = (root / "VERSION").read_text().strip()
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        parser.error("VERSION must contain a stable X.Y.Z release")
    updates = {}
    for name in ["mlflow-triton", "mlflow-triton-control"]:
        path = root / "packages" / name / "pyproject.toml"
        source = path.read_text()
        project = tomllib.loads(source)["project"]
        if project["version"] != version or project["name"] != name:
            parser.error(f"Package/version mismatch: {name}")
        if (
            name == "mlflow-triton-control"
            and f"mlflow-triton=={version}" not in project["dependencies"]
        ):
            parser.error("Control must depend on the matching core release")
        if args.write_version:
            next_version = args.tag[1:]
            source, count = re.subn(
                rf'(?m)^version = "{re.escape(version)}"$',
                f'version = "{next_version}"',
                source,
            )
            if count != 1:
                parser.error(f"Expected one version declaration in {path}")
            if name == "mlflow-triton-control":
                source = source.replace(
                    f'"mlflow-triton=={version}"',
                    f'"mlflow-triton=={next_version}"',
                )
            updates[path] = source
    if args.write_version:
        version = args.tag[1:]
        updates[root / "VERSION"] = f"{version}\n"
        for path, source in updates.items():
            path.write_text(source)
    if args.tag and args.tag != f"v{version}":
        parser.error(f"Expected tag v{version}, got {args.tag}")
    print(f"Both packages: {version}")


if __name__ == "__main__":
    main()
