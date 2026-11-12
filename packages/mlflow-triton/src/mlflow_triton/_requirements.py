"""Build standalone serving requirements from MLflow's saved environment files."""

import shlex
from pathlib import Path
from urllib.parse import urlsplit

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

_TORCH_PACKAGES = {"torch", "torchvision", "torchaudio"}


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
                for flag in (
                    "--constraint=",
                    "--requirement=",
                    "--find-links=",
                    "-c",
                    "-r",
                    "-f",
                ):
                    if options[0].startswith(flag):
                        options = [flag.rstrip("="), options[0][len(flag) :]]
                        break
            if len(options) == 2 and options[0] in {"-f", "--find-links"}:
                location = urlsplit(options[1])
                if (
                    location.scheme not in {"http", "https"}
                    or not location.netloc
                    or any(character.isspace() for character in options[1])
                ):
                    raise ValueError("Serving wheel sources must be HTTP(S) URLs")
                yield "--find-links " + options[1]
                continue
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


def _torch_exact_pins(requirement):
    if requirement.name not in _TORCH_PACKAGES or requirement.url:
        return []
    exact = [
        spec
        for spec in requirement.specifier
        if spec.operator == "==" and "*" not in spec.version
    ]
    versions = {Version(spec.version).public for spec in exact}
    local_versions = {
        Version(spec.version).local
        for spec in exact
        if Version(spec.version).local is not None
    }
    if len(versions) != 1 or len(local_versions) > 1:
        # Preserve unpinned user additions and conflicting version restrictions.
        return []
    return exact


def _cpu_torch_requirement(requirement):
    """Select CPU wheels without changing the public framework version.

    A bare torch==X pin also matches CUDA wheels. The +cpu label makes GPU
    fallback impossible; package-specific find-links makes those wheels visible
    to pip without changing the package index for unrelated dependencies.
    """
    exact = _torch_exact_pins(requirement)
    if not exact:
        return None
    cpu_pin = "==" + Version(exact[0].version).public + "+cpu"
    requirement.specifier = SpecifierSet(
        ",".join(
            cpu_pin if spec in exact else str(spec) for spec in requirement.specifier
        )
    )
    return f"--find-links https://download.pytorch.org/whl/cpu/{requirement.name}/"


def _exclude_cpu_torch_requirement(requirement):
    exact = _torch_exact_pins(requirement)
    if not exact:
        return
    public = Version(exact[0].version).public
    specs = [
        "==" + public
        if spec in exact and Version(spec.version).local == "cpu"
        else str(spec)
        for spec in requirement.specifier
    ]
    # An installed +cpu build satisfies a plain ==public pin. Exclude it so
    # image-package reuse cannot silently retain that build for a GPU model.
    requirement.specifier = SpecifierSet(",".join([*specs, "!=" + public + "+cpu"]))


def serving_requirements(
    package: Path,
    mlflow_version: str,
    *,
    prefer_cpu_torch: bool = False,
    exclude_cpu_torch: bool = False,
) -> str:
    """Promote constraints to dependencies so the API needs only one text file.

    Constraints for packages not otherwise required deliberately become explicit
    serving dependencies. Markers are retained for evaluation on the serving host.
    No dependency resolution, installation or network access happens here.
    """
    if prefer_cpu_torch and exclude_cpu_torch:
        raise ValueError("CPU and GPU Torch wheel policies are mutually exclusive")
    root = package.resolve()
    source = root / "requirements.txt"
    entries = (
        list(_read_requirements(source, root, frozenset())) if source.exists() else []
    )
    wheel_sources = dict.fromkeys(item for item in entries if isinstance(item, str))
    requirements = [item for item in entries if isinstance(item, Requirement)]
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
    if prefer_cpu_torch:
        for requirement in merged.values():
            if source := _cpu_torch_requirement(requirement):
                wheel_sources[source] = None
    elif exclude_cpu_torch:
        for requirement in merged.values():
            _exclude_cpu_torch_requirement(requirement)
    return "".join(f"{line}\n" for line in [*wheel_sources, *merged.values()])
