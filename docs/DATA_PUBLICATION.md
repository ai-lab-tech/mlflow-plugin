# Dataset publication preparation

The current working tree separates five public synthetic-data cases from eight
private cases. The private export preserves original files and hashes in
`ai-lab-tech/mlflow-plugin-testdata`; the shared catalog pins its commit.

## Existing history still contains restricted artifacts

This repository previously committed every case under `examples/data/`, including
historical `legacy59`, `trained21` and `custom` layouts. Deleting the current files
or adding ignore rules does not remove those commits or LFS objects.

Before changing visibility, publish a reviewed clean snapshot to a new public
repository, or rewrite the existing Git history and clean up remote refs/LFS storage.
Both options need explicit maintainer approval: replacing existing history affects
all collaborators, branch references and tags. Do not force-push automatically.

A clean public snapshot must contain the current plugin sources, five approved
fixture directories, the shared catalog, public tests and upstream notices. It
must exclude the original `.git` directory, cached datasets, private checkout,
credentials and all old binary objects. Verify it independently before publication.

If preserving historical commits is required, remove historical restricted paths
with an appropriate history-rewrite tool, audit all branches/tags and coordinate
remote LFS cleanup with the hosting provider. Avoid publishing private object IDs
as downloadable release assets. Do not delete the verified private export.

GitHub repository visibility and security/branch settings are external settings;
these documents and scripts do not change them.
