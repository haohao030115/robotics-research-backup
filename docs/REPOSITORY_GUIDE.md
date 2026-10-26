# Repository guide

The workspace is organized around seven maintained areas:

| Area | Maintained path | Purpose |
|---|---|---|
| Kimodo | `research/kimodo/` | Generation, export, adaptation, qualification |
| ARDY | `research/ardy/` | Segmented generation, overlap, adaptation, qualification, Qwen research |
| Contracts | `contracts/` | Canonical references, run manifests, policy checks |
| SONIC/MuJoCo | `execution/`, protected `GR00T-WholeBodyControl/`, `dl-group2/` | Native control, DDS, rendering, scene, Hand PD |
| Canonical Data Factory | `data_factory/` | Trace retention, forensic validation, V4 dataset compilation |
| VA research | `va_research/` | V2/V3 history and maintained V4/BCZ/V4.1/oracle code |
| Experiments | `experiments/` | One CLI, stage logs, immutable manifests, artifact catalog |

Use [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for architecture, timing,
interfaces, and results. Use [REPRODUCIBILITY_GUIDE.md](REPRODUCIBILITY_GUIDE.md)
for command-level operation. Use [REPRODUCIBILITY_INDEX.md](REPRODUCIBILITY_INDEX.md)
and [artifact_catalog.json](artifact_catalog.json) to locate evidence.

`workspace/` contains external checkouts and historical snapshots; `archive/`
contains retired reports and one-off orchestration. They are not alternate
maintained implementations. The preserved Git repositories retain their own
history and uncommitted work.
