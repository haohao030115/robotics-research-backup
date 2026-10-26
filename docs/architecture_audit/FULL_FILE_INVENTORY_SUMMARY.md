# Full workspace file inventory (2026-10-15)

The metadata inventory covers every reachable entry under `/home/group2` (the container path is `/workspace/group2`). It records paths, type, size, mode, uid/gid, symlink targets, and classification. It does not hash 150GB of data.

- Records: **565,212**
- Walk errors: **0** (inventory ran as the validated container root to inspect protected artifact directories)
- Inventory: `FULL_FILE_INVENTORY_20261015.jsonl.gz`

## Classification totals

| Category | Entries | Recorded bytes | Ownership |
|---|---:|---:|---|
| `compatibility_repository` | 42,033 | 4,410,076,853 | legacy/compatibility repositories |
| `external_repository` | 84,983 | 49,562,255,328 | external Kimodo/SONIC checkouts |
| `generated_artifacts` | 105,277 | 15,900,633,142 | outputs, datasets, experiment workspaces |
| `historical_archive` | 105,867 | 71,962,224,643 | archive and historical workspace |
| `maintained_source` | 437 | 3,192,523 | authoritative modules and tests |
| `runtime_environment` | 146,096 | 9,018,074,892 | Python/MUSA/runtime installations |
| `tooling_runtime` | 80,516 | 10,868,623,885 | editor, Codex, cache and session data |
| `workspace_control` | 3 | 4,352 | root control files |

## Top-level ownership map

- `.anaconda/`: `tooling_runtime` (2)
- `.bash_history/`: `tooling_runtime` (1)
- `.bash_logout/`: `tooling_runtime` (1)
- `.bashrc/`: `tooling_runtime` (1)
- `.cache/`: `tooling_runtime` (52)
- `.codex/`: `tooling_runtime` (8,616)
- `.conda/`: `tooling_runtime` (9)
- `.config/`: `tooling_runtime` (9)
- `.copilot/`: `tooling_runtime` (119)
- `.dotnet/`: `tooling_runtime` (9)
- `.gnupg/`: `tooling_runtime` (2)
- `.lesshst/`: `tooling_runtime` (1)
- `.local/`: `tooling_runtime` (3)
- `.profile/`: `tooling_runtime` (1)
- `.python_history/`: `tooling_runtime` (1)
- `.ssh/`: `tooling_runtime` (3)
- `.viminfo/`: `tooling_runtime` (1)
- `.vscode/`: `tooling_runtime` (2)
- `.vscode-remote-containers/`: `tooling_runtime` (16)
- `.vscode-server/`: `tooling_runtime` (71,666)
- `.wget-hsts/`: `tooling_runtime` (1)
- `AGENTS.md/`: `workspace_control` (1)
- `GR00T-WholeBodyControl/`: `external_repository` (5,580)
- `README.md/`: `workspace_control` (1)
- `archive/`: `historical_archive` (1,903)
- `ardy_musa_env/`: `runtime_environment` (3)
- `canonical_va/`: `compatibility_repository` (282)
- `config.yaml/`: `workspace_control` (1)
- `contracts/`: `maintained_source` (14)
- `data_factory/`: `maintained_source` (42)
- `dl-group2/`: `compatibility_repository` (41,035)
- `docs/`: `maintained_source` (43)
- `execution/`: `maintained_source` (19)
- `experiments/`: `maintained_source` (19)
- `expert_datasets/`: `generated_artifacts` (44,319)
- `grasp_pipeline/`: `compatibility_repository` (45)
- `grasp_workspace/`: `generated_artifacts` (66)
- `handoffs/`: `historical_archive` (1)
- `kimodo-t6/`: `external_repository` (79,403)
- `miniconda3/`: `runtime_environment` (146,090)
- `outputs/`: `generated_artifacts` (60,892)
- `research/`: `maintained_source` (174)
- `scripts/`: `maintained_source` (3)
- `tests/`: `maintained_source` (9)
- `torch_musa_post1/`: `runtime_environment` (3)
- `va_research/`: `maintained_source` (114)
- `va_training/`: `compatibility_repository` (671)
- `workspace/`: `historical_archive` (103,963)

No top-level entry is unclassified. Moving a path requires updating provenance manifests and compatibility links; the inventory is the authoritative checklist for any future deletion or migration.
