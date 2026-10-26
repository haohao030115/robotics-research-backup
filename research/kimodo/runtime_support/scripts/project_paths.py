"""Project inputs/outputs, independent of the shell's current directory."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
SCENES = ROOT / "scenes"
OUTPUTS = ROOT / "outputs"
DEFAULT_EPISODE = OUTPUTS / "expert_episodes/freebase_wbc_002"
SONIC_REPO = Path(os.environ.get("SONIC_REPO", ROOT.parent / "GR00T-WholeBodyControl")).expanduser().resolve()
PINNED_REPO = ROOT / "third_party/GR00T-WholeBodyControl"
ROBOT_RELATIVE = Path("decoupled_wbc/control/robot_model/model_data/g1/g1_29dof_with_hand_rev_1_0_activatedfinger.xml")


def default_robot_xml():
    """Prefer the pinned model; only reuse external checkout at the same commit.

    Read-only fallback supports the HKU layout with an unpopulated submodule.
    Neither repository is initialized or modified by this helper.
    """
    pinned = PINNED_REPO / ROBOT_RELATIVE
    if pinned.is_file():
        return pinned
    external = SONIC_REPO / ROBOT_RELATIVE
    if external.is_file():
        try:
            entry = subprocess.check_output(
                ["git", "-C", str(ROOT), "ls-tree", "HEAD", "third_party/GR00T-WholeBodyControl"], text=True
            ).split()
            commit = subprocess.check_output(["git", "-C", str(SONIC_REPO), "rev-parse", "HEAD"], text=True).strip()
            if len(entry) >= 3 and entry[2] == commit:
                return external
        except (OSError, subprocess.CalledProcessError):
            pass
    return pinned  # A missing/mismatched model fails clearly at loading, not silently.
