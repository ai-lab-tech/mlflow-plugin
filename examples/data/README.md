# Saved test artifacts

This directory contains the data for all 12 cases in `../real_models.py`:
four sklearn, four PyTorch and four Hugging Face examples. It is a byte-for-byte
copy of the selected case directories from triton-model-generator/benchmarks/data.
No generator checkout, training run or model download is needed to load them.

Binary weights, serialized architectures and NumPy fixtures are tracked by Git LFS.
After cloning, run `git lfs install --local` and `git lfs pull` from the repository
root. The checkout contains about 600 MiB of test data.

`manifest.json` lists the case paths, file sizes and SHA-256 checksums. Existing
provenance, dependency files, tokenizer assets and original training sources are
preserved. Historical absolute paths in provenance describe the original runs;
the example runner uses repository-relative paths, not those historical paths.
Original source notices and provenance remain with their respective files.

Training sources are evidence only: the examples never import or execute them.
Legacy `independent-inputs.npz` files are retained as source data but are not used
by our examples. Example requests are defined in `../real_models.py`, and expected
serving outputs are computed from the saved native model during each run.
Generated MLflow packages, caches and Triton logs stay outside Git on the data disk.
