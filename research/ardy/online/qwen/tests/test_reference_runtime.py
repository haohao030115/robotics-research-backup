import csv
from pathlib import Path

from qwen_ardy.reference_runtime import publish_segment


def test_segment_is_atomic_and_body_only(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ["joint_pos.csv", "right_hand_pos.csv"]:
        with (source / name).open("w", newline="") as stream:
            writer = csv.writer(stream); writer.writerow(["a", "b"])
            for i in range(10): writer.writerow([i, i + 1])
    root = tmp_path / "references"; root.mkdir()
    result = publish_segment(source, root, "raise_hand", 2, 6, "raise", {})
    assert result.is_dir()
    assert not (result / "right_hand_pos.csv").exists()
    assert len((result / "joint_pos.csv").read_text().splitlines()) == 5
    assert not list(root.glob("*.publishing-*"))
