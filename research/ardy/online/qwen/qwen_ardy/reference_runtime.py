from __future__ import annotations

import csv
import json
import os
import shutil
import time
from pathlib import Path


FORBIDDEN_HAND_FILES = {"left_hand_pos.csv", "right_hand_pos.csv"}


def publish_segment(source: Path, reference_root: Path, name: str,
                    begin: int, end: int, prompt: str,
                    constraint: dict, reverse: bool = False,
                    frame_repeat: int = 1) -> Path:
    """Publish a body-only CSV segment with one atomic directory rename."""
    final = reference_root / name
    temporary = reference_root / f".{name}.publishing-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for item in source.iterdir():
        if item.name in FORBIDDEN_HAND_FILES or not item.is_file():
            continue
        target = temporary / item.name
        if item.suffix != ".csv":
            shutil.copy2(item, target)
            continue
        with item.open(newline="") as src, target.open("w", newline="") as dst:
            rows = csv.reader(src)
            writer = csv.writer(dst)
            writer.writerow(next(rows))
            selected = [row for index, row in enumerate(rows) if begin <= index < end]
            if reverse:
                selected.reverse()
            for row in selected:
                for _ in range(frame_repeat):
                    writer.writerow(row)
    (temporary / "stage.json").write_text(json.dumps({
        "stage": name, "source_frames": [begin, end], "prompt": prompt,
        "constraint": constraint, "reverse": reverse,
        "frame_repeat": frame_repeat,
        "hand_control": "contact_driven",
    }, indent=2) + "\n")
    if final.exists():
        shutil.rmtree(final)
    temporary.rename(final)
    return final


def request_switch(reference_root: Path, segment: Path, timeout_s: float = 10.) -> str:
    """Atomically ask SONIC to load a published segment and await acknowledgement."""
    command = reference_root / ".sonic_reference_command"
    status = reference_root / ".sonic_reference_status"
    token = segment.name
    tmp = command.with_suffix(".tmp")
    tmp.write_text(str(segment.resolve()) + "\n")
    tmp.replace(command)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if status.exists():
            value = status.read_text().strip()
            if value.startswith("error "):
                raise RuntimeError(value)
            if value.startswith((f"ready {token} ", f"continued {token} ")):
                return value
        time.sleep(.02)
    raise TimeoutError(f"SONIC did not acknowledge {token}")


def submit_switch(reference_root: Path, segment: Path) -> None:
    command = reference_root / ".sonic_reference_command"
    temporary = reference_root / ".sonic_reference_command.tmp"
    temporary.write_text(str(segment.resolve()) + "\n")
    temporary.replace(command)


def switch_status(reference_root: Path, segment: Path) -> str | None:
    status = reference_root / ".sonic_reference_status"
    if not status.exists():
        return None
    value = status.read_text().strip()
    if value.startswith("error "):
        raise RuntimeError(value)
    return value if value.startswith((f"ready {segment.name} ",
                                      f"continued {segment.name} ")) else None


def hold_status(reference_root: Path, segment_name: str) -> str | None:
    """Return SONIC's acknowledgement once a segment is held at its last frame."""
    status = reference_root / ".sonic_reference_status"
    if not status.exists():
        return None
    value = status.read_text().strip()
    if value.startswith("error "):
        raise RuntimeError(value)
    return value if value.startswith(f"holding {segment_name} ") else None
