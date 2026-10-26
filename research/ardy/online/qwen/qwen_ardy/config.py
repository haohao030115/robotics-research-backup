from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    project: Path
    workspace: Path
    fuyuhan: Path
    sonic: Path
    ardy: Path
    scene: Path
    baseline_motion: Path

    @classmethod
    def discover(cls) -> "Paths":
        project = Path(__file__).resolve().parents[1]
        workspace = Path(os.getenv("RESEARCH_ROOT", "/workspace/group2"))
        fuyuhan = Path(os.getenv("FUYUHAN_ROOT", workspace / "fuyuhan"))
        sonic = Path(os.getenv("SONIC_ROOT", fuyuhan / "GR00T-WholeBodyControl"))
        ardy = Path(os.getenv("ARDY_ROOT", fuyuhan / "ardy"))
        scene = Path(os.getenv(
            "GRASP_SCENE",
            sonic / "gear_sonic/data/robot_model/model_data/g1/scene_43dof_blocks_ardy_sides.xml",
        ))
        baseline = sonic / "gear_sonic_deploy/reference/ardy_grasp_sides"
        return cls(project, workspace, fuyuhan, sonic, ardy, scene, baseline)

    def validate(self) -> None:
        required = {
            "SONIC checkout": self.sonic,
            "ARDY checkout": self.ardy,
            "scene": self.scene,
            "baseline motion": self.baseline_motion,
            "SONIC binary": self.sonic / "gear_sonic_deploy/target/release/g1_deploy_onnx_ref",
        }
        missing = [f"{label}: {path}" for label, path in required.items() if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing runtime assets:\n" + "\n".join(missing))


QWEN_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
QWEN_MODEL = os.getenv("QWEN_VL_MODEL", "qwen3-vl-flash")
