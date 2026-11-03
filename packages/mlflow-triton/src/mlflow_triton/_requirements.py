"""Build standalone serving requirements from MLflow's saved environment files."""

import shlex
from pathlib import Path
from urllib.parse import urlsplit

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name


def _read_requirements(path: Path, root: Path, active: frozenset[Path]):
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError(
            "Serving requirements must reference files inside the saved model"
        )
    if path in active:
        raise ValueError("Cyclic serving requirements file reference")
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            options = shlex.split(line)
            # MLflow normally writes '-c constraints.txt'. Also accept saved
            # requirement includes without leaving references in the output.
            if len(options) == 1:
                for flag in ("--constraint=", "--requirement=", "-c", "-r"):
                    if options[0].startswith(flag):
                        options = [flag.rstrip("="), options[0][len(flag) :]]
                        break
            if len(options) != 2 or options[0] not in {
                "-c",
                "--constraint",
                "-r",
                "--requirement",
            }:
                raise ValueError(f"Unsupported serving requirements option: {line}")
            yield from _read_requirements(
                path.parent / options[1], root, active | {path}
            )
            continue
        try:
            requirement = Requirement(line)
        except InvalidRequirement as exc:
            raise ValueError(f"Invalid serving requirement: {line}") from exc
        if requirement.url and urlsplit(requirement.url).scheme not in {
            "http",
            "https",
        }:
            raise ValueError(
                "Serving requirements cannot depend on local files or VCS checkouts"
            )
        yield requirement


def serving_requirements(package: Path, mlflow_version: str) -> str:
    """Promote constraints to dependencies so the API needs only one text file.

    Constraints for packages not otherwise required deliberately become explicit
    serving dependencies. Markers are retained for evaluation on the serving host.
    No dependency resolution, installation or network access happens here.
    """
    root = package.resolve()
    source = root / "requirements.txt"
    requirements = (
        list(_read_requirements(source, root, frozenset())) if source.exists() else []
    )
    mlflow_requirements = [
        item
        for item in requirements
        if canonicalize_name(item.name) in {"mlflow", "mlflow-skinny"}
    ]
    if not any(item.marker is None for item in mlflow_requirements):
        # A marker may exclude MLflow on the serving host. Require it there too,
        # without adding a version pin that conflicts with an active user pin.
        fallback = (
            "mlflow>=3.14,<4" if mlflow_requirements else f"mlflow=={mlflow_version}"
        )
        requirements.append(Requirement(fallback))
    merged = {}
    for requirement in requirements:
        requirement.name = canonicalize_name(requirement.name)
        key = (requirement.name, str(requirement.marker))
        previous = merged.get(key)
        if previous is None:
            merged[key] = requirement
            continue
        if previous.url or requirement.url:
            urls = {url for url in (previous.url, requirement.url) if url}
            if len(urls) > 1 or previous.specifier or requirement.specifier:
                raise ValueError(
                    f"Cannot combine URL and version constraints for {requirement.name}"
                )
            previous.url = next(iter(urls))
        previous.specifier &= requirement.specifier
        previous.extras |= requirement.extras
    return "".join(f"{requirement}\n" for requirement in merged.values())
