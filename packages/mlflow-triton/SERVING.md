# Prepared serving models

`mlflow_triton.serving.log_model` publishes an engineer-prepared Triton repository
as a **new logged model**, with its exact source, conversion run and optional
serving predecessor. It does not convert weights, execute model code, deploy a
workload or approve a release.

## Setup and example

```bash
pip install 'mlflow-triton[serving]'
```

Configure the destination with MLflow's normal tracking and Registry settings.
Provision a publication database and set
`MLFLOW_TRITON_PUBLICATION_DATABASE_URL` through your environment or secret
manager. Use a `postgresql+psycopg2://...` SQLAlchemy URL for publishers on multiple
hosts. All publishers for a Registry must share this database and the same
Registry URI. The database account needs permission to create and read/write the
plugin's receipt table; it does not need access to MLflow's internal tables.

For one POSIX development host, an explicit persistent
`sqlite:////absolute/path/publications.db` also works. Its sibling `.locks`
directory provides process locking. Do not put this database on a shared network
filesystem or use separate SQLite files for cooperating publishers.

```python
from mlflow_triton import serving

info = serving.log_model(
    source_model_uri="models:/scientist-model/12",
    triton_repository_path="./prepared-repository",
    conversion_run_id=conversion_run.info.run_id,
    source_tracking_installation_id="science-lab",
    tracking_installation_id="engineering-lab",
    registered_model_name="serving-onnx",
    publication_operation_id="conversion-job-2026-11-16-attempt-1",
)

print(info.model_uri)
print(info.registered_model_version)
print(info.metadata["triton_lineage"])
```

The example assumes that the source model and conversion run already exist and
the caller can read them. Replace their names and IDs with real records.

Expected input layout:

```text
prepared-repository/
  iris/
    config.pbtxt
    1/
      model.onnx
```

V1 accepts one model and one positive numeric version, preserving its number and
all file bytes. Configurations are parsed with Triton's protobuf schema. Model
names, backend/platform declarations, selected versions and required weight/code
files must agree. Custom backends require `default_model_filename`. Ensembles,
repository agents and multiple models/versions need a future contract. Files
must be frozen while captured, with NFC relative POSIX names. Symlinks, hard
links, special files and changes detected during copying are rejected. Capture
is bounded to 10 GiB, 10,000 files and a 1 MiB config. This is not a sandbox for a
directory concurrently modified by an adversary.

## Arguments

All arguments are keyword-only.

| Argument | Meaning |
| --- | --- |
| `source_model_uri` | Required: `models:/m-...`, numeric Registry version, `models:/name@alias`, or `runs:/run-id/path`. Registry references must resolve to an MLflow 3 logged model. |
| `triton_repository_path` | Required: frozen local prepared repository root. |
| `conversion_run_id` | Required: actual engineering run on the destination tracking server. |
| `previous_serving_model_uri` | Optional engineering predecessor, independent of source and release replacement. It must have a Triton flavor. |
| `registered_model_name` | Optional serving Registry name, distinct from the selected source's Registry name. |
| `name` | Logged-model name; defaults to `serving-model`. |
| `backend` | Optional expected backend; must match the config. |
| `precision` | Optional caller-declared precision label; no inference or validation approval is implied. |
| `tracking_installation_id` | Destination origin ID. Defaults to a hash of its tracking URI. |
| `source_tracking_installation_id` | Source origin ID. Defaults to a hash of the source tracking URI. |
| `previous_tracking_installation_id` | Predecessor location ID. Existing candidate origins remain preserved when available. |
| `source_tracking_uri`, `source_registry_uri` | Optional source server configuration, separate from destination. |
| `previous_tracking_uri`, `previous_registry_uri` | Optional predecessor server configuration. |
| `publication_operation_id` | Stable retry key. Defaults to a hash of selected references, measured output and options. Reuse it for retries; use a new one for a new publication. |
| `publication_database_url` | Overrides `MLFLOW_TRITON_PUBLICATION_DATABASE_URL`. Never included in lineage or tags. |

Use explicit installation IDs for stable origins across server-address changes.
Server URIs are hashed in receipt intent; connection credentials are not copied
to lineage or searchable tags. Use MLflow's authentication configuration for
provider access, and do not put secrets in model references, IDs or labels.

## Artifacts and identity

The logged artifact contains `MLmodel`, `lineage.json` and the unchanged files
under `model_repository/`. `ModelInfo.metadata["triton_lineage"]` supplies the
logged-model URI, relative manifest path, manifest/candidate digests and operation
ID. Download `info.model_uri` with MLflow, then read its `lineage.json`; appending
`/lineage.json` directly to a logged-model URI is not portable across MLflow
versions.

The manifest includes the `ServingCandidateV1` envelope and complete input
and output inventories. Canonical JSON uses UTF-8, sorted keys, compact separators
and explicit defaults/nulls; file entries sort by UTF-8 path bytes. The output
inventory covers only the prepared repository. `MLmodel`, lineage and receipts
are outside it, avoiding circular hashes. Source artifacts are measured from a
private capture; they are never rewritten or loaded as executable models.

Registry aliases are resolved once before the receipt is stored. Retries reuse
that captured identity even if an alias moves. A changed output, source selection
or option under an explicit operation ID fails; it cannot overwrite the receipt.
Changed weights under a new operation create a new candidate, even with compatible
configuration. Backend forks retain the same source and an independently chosen
predecessor. Cross-server identities include installation IDs; equal model names
or Registry numbers do not establish ancestry.

The caller identifies the actual conversion run. Inventories establish the bytes
captured, not proof that the caller consumed those inputs during conversion.
The conversion tool is left unspecified rather than fabricated. This manifest
is provenance evidence, not a runtime-validation report or release approval.
Legacy Triton predecessors without candidate lineage retain their exact MLflow
identity; downstream consumers must capture/register that predecessor before
publishing a candidate that references it.

Searchable `triton.source.*`, `triton.conversion.run_id`, `triton.output.inventory`
and `triton.previous.*` tags are written to the logged model, engineering run and
Registry version. Run tags summarize its latest publication; use logged-model
search for multiple outputs. No growing descendant list is written to the source.

## Retries and concurrency

The additive `triton_serving_publications_v1` table stores captured intent and
publication progress. PostgreSQL session advisory locks serialize operations and
Registry names across cooperating publishers. Existing MLflow schemas and model
artifacts are untouched. Back up the receipt database with model artifacts; losing
it loses retry identity. There is no migration of older wrapper publications.

Before each non-idempotent model/version creation, the plugin commits its intent.
After a timeout or process interruption, repeat the same call. It searches all
pages for the recorded operation and content and verifies the resulting model/run
identity before proceeding. Uploads retry identical bytes only to the operation's
own model. Registry requests that remain pending can be retried for reconciliation.

If no unique matching result is visible, `AmbiguousPublication` stops the call
without issuing another creation request. This can also occur if a process died
after committing intent but before sending the request. Investigate the provider
and retry reconciliation later; do not delete the receipt or change the operation
ID merely to bypass uncertainty. Tags support lookup; the durable intent and
serialization provide the duplicate-write protection. There is no claim of a
distributed transaction with MLflow, or serialization of unrelated clients that
bypass this API. Native Registry writers need their own coordination.

The existing framework wrappers, `register_model`, and `_versions.py` retain their
current compatibility and shared-repository numbering meanings. This API preserves
prepared versions and records explicit lineage; it does not use a predecessor to
append new weights to an older model. A runtime-only release can reuse the same
candidate and select a new image through the release system.

## Verification

```bash
python -m unittest discover -s tests -p test_derived_serving.py -v
# To exercise the same suite with real PostgreSQL receipt storage:
SERVING_TEST_DATABASE_URL="$TEST_POSTGRES_URL" \
  python -m unittest discover -s tests -p test_derived_serving.py -v
```

Tests use real local MLflow tracking/Registry databases and artifacts, inject
lost responses after successful writes, and cover aliases, predecessor links,
forks, changed weights, source immutability, shared schemas, cross-server origins,
authorization failures and concurrent publications. CI runs PostgreSQL tests for
minimum and latest supported MLflow. No Kubernetes workload is created by this
API; actual inference and production qualification remain separate checks.

Local verification on 2026-11-16: all 18 publication tests passed with PostgreSQL
16 on MLflow 3.14.0 and 3.17.2. The full 3.17.2 package suite passed 237 tests
(nine optional framework skips); the isolated 3.14.0 core suite passed 182 tests
(with the same skips). Ruff lint/format, release consistency, core wheel/sdist
builds and strict distribution metadata checks passed.
