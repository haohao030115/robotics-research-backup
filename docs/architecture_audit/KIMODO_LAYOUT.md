# Kimodo path ownership

## `/home/group2/kimodo-t6`

This is the external Kimodo integration and artifact boundary. It contains:

- `kimodo-main/`: upstream Kimodo generation library and CLI;
- `g1-integration/`: G1 conversion, checkpoints, qualification runs, logs, and
  canonical reference snapshots;
- `models/`: downloaded Kimodo model assets;
- `kimodo-viser-main/` and related checkouts: visualization dependencies;
- historical staging bundles and experiment outputs.

The maintained local orchestration and adapters are in
`/home/group2/research/kimodo` and `/home/group2/experiments`. The new workflow
still resolves the validated Kimodo-G1 checkpoint and historical reference
paths inside `kimodo-t6`, so deleting or flattening this directory would break
reproducibility and provenance.

## `kimodo_musa_env`

The former top-level directory was only a 28 KB Python 3.10 virtual-environment
shell. It pointed to `/usr/bin/python3.10` and had no `torch` or `torch_musa`.
It was not the validated MUSA runtime and had no active source references. It is
now retained at `archive/environments/kimodo_musa_env_legacy` with a snapshot.
Use `va-train` for Kimodo/VA execution.
