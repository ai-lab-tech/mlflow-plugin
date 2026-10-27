"""Shared case catalog and explicit selection of locally available artifacts."""

import json
import os
from pathlib import Path

CATALOG = json.loads((Path(__file__).with_name("test_catalog.json")).read_text())
CASES = {
    name: (case["family"], case["artifact_path"])
    for name, case in CATALOG["cases"].items()
}
DEFAULT_DATA_ROOT = Path(
    os.environ.get("MLFLOW_TRITON_DATA_ROOT", Path(__file__).with_name("data"))
)


def select_cases(data_root, requested=None, families=None, device="cpu"):
    names = list(dict.fromkeys(requested or CASES))
    if families:
        names = [name for name in names if CASES[name][0] in families]
    if device == "gpu":
        if (requested or families) and any(
            CASES[name][0] == "sklearn" for name in names
        ):
            raise ValueError(
                "sklearn supports CPU only; select PyTorch/Hugging Face for --device gpu"
            )
        names = [name for name in names if CASES[name][0] != "sklearn"]
    missing = [
        name for name in names if not (Path(data_root) / CASES[name][1]).is_dir()
    ]
    if missing and (requested or families):
        descriptions = ", ".join(
            f"{name} ({CATALOG['cases'][name]['source']})" for name in missing
        )
        raise ValueError(
            "Missing selected artifacts: "
            + descriptions
            + ". Run scripts/prepare_test_data.py --data-root <directory>; add --include-private for private cases, then pass that directory to --data-root."
        )
    if missing:
        print("Unavailable cases (not run): " + ", ".join(missing))
    names = [name for name in names if name not in missing]
    if not names:
        raise ValueError(
            "No available cases match the selection; prepare the dataset first"
        )
    return names
