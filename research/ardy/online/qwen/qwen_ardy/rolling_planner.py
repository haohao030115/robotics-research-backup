"""Non-blocking latest-frame scheduler for local Qwen Visual Trace planning."""
from __future__ import annotations

import json
import select
import time
from dataclasses import dataclass
from pathlib import Path
from subprocess import Popen


@dataclass
class TracePlan:
    request_id: int
    captured_at: float
    decision: dict
    request: dict


class RollingVisualPlanner:
    """Keep at most one VLM inference in flight and discard stale results."""

    def __init__(self, worker: Popen, interval_s: float = 2.2, stale_after_s: float = 8.):
        self.worker = worker
        self.interval_s = interval_s
        self.stale_after_s = stale_after_s
        self.next_id = 0
        self.inflight: dict | None = None
        self.last_submit_at = float("-inf")

    def maybe_submit(self, now: float, image: Path, qpos: list[float],
                     qvel: list[float], state: dict, *,
                     full_image: Path | None = None, crop: dict | None = None) -> bool:
        if self.inflight is not None or now - self.last_submit_at < self.interval_s:
            return False
        request = {"id": self.next_id, "mode": "visual_trace", "image": str(image),
                   "qpos": qpos, "qvel": qvel, "state": state, "captured_at": now}
        if full_image is not None:
            request["full_image"] = str(full_image)
        if crop is not None:
            request["crop"] = crop
        self.worker.stdin.write(json.dumps(request) + "\n")
        self.worker.stdin.flush()
        self.inflight = request
        self.next_id += 1
        self.last_submit_at = now
        return True

    def poll(self, now: float | None = None) -> TracePlan | None:
        now = time.monotonic() if now is None else now
        if self.inflight is None or not select.select([self.worker.stdout], [], [], 0)[0]:
            return None
        response = json.loads(self.worker.stdout.readline())
        request = self.inflight
        self.inflight = None
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "local Qwen worker failed"))
        if now - request["captured_at"] > self.stale_after_s:
            return None
        return TracePlan(response["id"], request["captured_at"], response["decision"], request)
