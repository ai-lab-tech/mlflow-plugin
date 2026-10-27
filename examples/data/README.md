# Test dataset

This directory contains the data for all 13 cases in `../real_models.py`:
four sklearn, four PyTorch and five Hugging Face examples. The original twelve
are byte-for-byte copies of selected case directories from
triton-model-generator/benchmarks/data. The additional custom HF case reuses the
unchanged hf_glue weights/tokenizer with local custom configuration/model code.
No generator checkout, training run or model download is needed to load them.

Binary weights, serialized architectures and NumPy fixtures are tracked by Git LFS.
After cloning, run `git lfs install --local` and `git lfs pull` from the repository
root. The checkout contains about 600 MiB of test data.

All cases live directly under `examples/data/<case-name>/`, using the same names
as `--case` in the runner. There are no historical batch directories or separate
custom-model dataset. sklearn/Hugging Face assets are in each case's `bundle/`;
PyTorch cases retain their `artifact/` and saved architecture files.

`manifest.json` lists current `dataset_directory` and `artifact_path` values,
file sizes and SHA-256 checksums. `source_directory` records the original source
location for provenance; it is not a path used by the runner. Existing
provenance, dependency files, tokenizer assets and original training sources are
preserved. Historical absolute paths in provenance describe the original runs;
the example runner uses repository-relative paths, not those historical paths.
Original source notices and provenance remain with their respective files.

Training sources are evidence only: the examples never import or execute them.
Legacy `independent-inputs.npz` files are retained as source data but are not used
by our examples. Example requests are defined in `../real_models.py`, and expected
serving outputs are computed from the saved native model during each run.
Generated MLflow packages, caches and Triton logs stay outside Git on the data disk.
