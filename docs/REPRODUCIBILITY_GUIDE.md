# Reproducibility guide

Run the following from the workspace root:

```bash
bash scripts/check_environment.sh
python3 -m experiments --help
python3 -m experiments catalog --output docs/artifact_catalog.json
```

Use a fresh output directory for every executable run. The primary workflows
are:

```bash
python3 -m experiments run --family kimodo --run-id ID --output outputs/ID
python3 -m experiments run --family ardy --run-id ID --output outputs/ID --config research/ardy/config/default.json
python3 -m experiments canonical --run-id ID --reference REFERENCE --output outputs/ID --mode production
python3 -m experiments dataset --run-id ID --canonical-root CANONICAL --output outputs/ID --all-train
python3 -m experiments va train --family v4 --run-id ID --manifest DATASET_MANIFEST --output outputs/ID --device musa:0
python3 -m experiments va offline --family v4 --run-id ID --checkpoint CHECKPOINT --output outputs/ID
python3 -m experiments va closed --family v4 --run-id ID --checkpoint CHECKPOINT --output outputs/ID
```

Add `--dry-run` to inspect command construction without starting a workload.
Training, native replay, rendering, and closed-loop execution require the
validated `va-train` container. Native runs require the existing isolated DDS
wrapper and production rendering requires the MTT hardware renderer.

The catalog distinguishes canonical episodes, compiled datasets, checkpoints,
evaluation reports, intermediate manifests, and incomplete artifacts. It also
records duplicate episode identities, embedded container paths, missing
references, and SHA-256 hashes. Begin with
[REPRODUCIBILITY_INDEX.md](REPRODUCIBILITY_INDEX.md) for representative
source-to-result chains.
