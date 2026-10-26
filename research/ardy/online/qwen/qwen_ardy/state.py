from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .models import EpisodeSummary


def summarize_episode(attempt: int, output_dir: Path) -> EpisodeSummary:
    result = json.loads((output_dir / "result.json").read_text())
    with np.load(output_dir / "simulation.npz") as data:
        states = np.asarray(data["states"], dtype=float)
    # test_ardy_grasp rows: time, block xyz, grasp-center xyz, root xyz, contact count.
    distance = np.linalg.norm(states[:, 1:4] - states[:, 4:7], axis=1)
    index = int(np.argmin(distance))
    block_initial = states[0, 1:4]
    return EpisodeSummary(
        attempt=attempt,
        success=bool(result.get("success", False)),
        block_world_initial=block_initial.round(6).tolist(),
        block_world_final=states[-1, 1:4].round(6).tolist(),
        closest_grasp_center_world=states[index, 4:7].round(6).tolist(),
        closest_delta_world=(states[index, 1:4] - states[index, 4:7]).round(6).tolist(),
        minimum_hand_block_distance_m=float(distance[index]),
        block_lift_max_m=float(result["block_lift_max_m"]),
        block_lift_final_m=float(result["block_lift_final_m"]),
        final_contacts=int(result["final_contacts"]),
        minimum_root_height_m=float(result["minimum_root_height_m"]),
    )


def write_summary(summary: EpisodeSummary, path: Path) -> None:
    path.write_text(json.dumps(summary.to_dict(), indent=2) + "\n")

