# Public test fixtures

The shared [catalog](../test_catalog.json) lists **all 13 stable case IDs** with
family, artifact path, data source, license-review status and SHA-256 file hashes.
This directory contains only the five cases whose saved-model provenance identifies
synthetic training inputs and whose source-code licenses are documented:

| Case | Saved training inputs |
| --- | --- |
| `sklearn_gradient_boosting` | `make_hastie_10_2` |
| `sklearn_scaled_mlp` | Final `make_moons` fit and matching scaler |
| `pytorch_polynomial` | Synthetic polynomial samples |
| `pytorch_custom_network` | Random synthetic tensors |
| `pytorch_dcgan` | Recorded `--dataset fake` run |

The other eight cases are stored in the private `ai-lab-tech/mlflow-plugin-testdata`
repository while redistribution questions remain open. Private storage is not a
license grant. Classification as `documented_synthetic` records technical evidence;
it is not a blanket legal opinion about every upstream component.

## Prepare local data (Python 3.11+)

From the plugin repository root, choose a data directory outside this checkout:

```bash
# Public fixture binaries use Git LFS:
git lfs pull
python scripts/prepare_test_data.py --data-root /path/to/mlflow-test-data

# Add private cases using normal Git/SSH access to the private repository:
python scripts/prepare_test_data.py --include-private --data-root /path/to/mlflow-test-data

# Or reuse an existing checkout at the catalog's pinned commit, without downloads:
python scripts/prepare_test_data.py --include-private \
  --private-checkout /path/to/mlflow-plugin-testdata \
  --data-root /path/to/mlflow-test-data
```

Use `--case <case-id>` repeatedly to prepare only selected cases. Private selection
requires `--include-private`. The script pins the private Git commit, fetches LFS
objects only for selected private case directories, checks sizes and SHA-256 hashes,
and copies only catalog-listed files. An existing identical destination is reused;
differing files cause an error rather than being overwritten. A managed private
checkout cache resides inside `<data-root>/.sources/` on the same disk.

`dataset-state.json` records the catalog hash, private revision when applicable,
and available cases. It is an inventory, not a test-success report. No training,
model execution or server startup happens during preparation. Credentials remain
in the user's SSH/Git configuration and are never written into the catalog.

Run the existing tests against the combined directory:

```bash
python examples/run_real_models.py \
  --data-root /path/to/mlflow-test-data --output-root /path/to/mlflow-test-results
```

Alternatively set `MLFLOW_TRITON_DATA_ROOT`. Without a configured data directory,
runners use the public fixtures in this checkout. Default selection runs available
cases and explicitly lists unavailable cases; explicitly selecting a missing case
or family fails with preparation instructions. The public CI checks public fixtures
and uses synthetic unit fixtures, never private credentials or private model weights.

## Provenance and updates

Original invocation/preparation files and upstream notices remain with each case.
Historical paths record the original runs, not current loading locations. Training
sources are evidence and are never executed by the serving runner.

When updating a private fixture: commit and push its weights through Git LFS, then
update the public catalog's private commit and file hashes in the same reviewed PR
as any test-definition changes. Do not relabel a case as public without documenting
the redistribution evidence. Keep IDs stable; preserve benchmark results separately.
See [third-party notices](../../THIRD_PARTY_NOTICES.md).

Moving files out of the working tree does not remove old Git/LFS history. See
[publication preparation](../../docs/DATA_PUBLICATION.md) before making this repo public.
