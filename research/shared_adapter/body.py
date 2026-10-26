"""Bounded preservation of producer generated body motion.

The producer adapters solve the active wrist/arm task in task space.  Their
historical neutral-body path discarded every generated non-active-arm joint,
which made cross-seed body diversity disappear.  This module provides one
deterministic policy for retaining a bounded residual while keeping the
exported trajectory inside joint and rate limits.

The function is deliberately independent of the scene position and seed.  It
operates only on the generated source, the canonical reset posture, the
timeline, and optional MuJoCo joint ranges supplied by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class BodyPreservationConfig:
    """Fixed safety bounds for the producer-neutral body residual."""

    version: str = "bounded-body-v1"
    smooth_window: int = 9
    max_residual_rad: float = 0.20
    max_rate_rad_s: float = 3.0


def _smooth_rows(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return x.copy()
    width = int(width) | 1
    pad = width // 2
    kernel = np.ones(width, dtype=float) / width
    padded = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    return np.stack(
        [np.convolve(padded[:, j], kernel, mode="valid") for j in range(x.shape[1])],
        axis=1,
    )


def bounded_body_trajectory(
    source_body,
    initial_body,
    timestamps,
    *,
    weight: float = 1.0,
    config: BodyPreservationConfig | None = None,
    joint_lower=None,
    joint_upper=None,
):
    """Return a deterministic, bounded generated-body trajectory.

    ``source_body`` and ``initial_body`` are ``[T,J]`` and ``[J]`` arrays in
    the same joint order.  The returned trajectory starts exactly at the
    canonical reset posture.  Source residuals are smoothed, multiplied by
    the requested global preservation weight, clipped by a symmetric joint
    residual bound, clipped to MuJoCo joint limits when provided, and finally
    rate limited at the supplied timestamps.  A one-frame source spike cannot
    therefore create a discontinuity in the exported reference.

    The returned metadata is suitable for provenance and reports the actual
    residual/velocity clipping, rather than only the requested weight.
    """
    c = config or BodyPreservationConfig()
    src = np.asarray(source_body, dtype=float)
    init = np.asarray(initial_body, dtype=float)
    ts = np.asarray(timestamps, dtype=float)
    if src.ndim != 2 or init.ndim != 1 or src.shape[1] != len(init):
        raise ValueError("source_body must be [T,J] and initial_body [J]")
    if len(ts) != len(src) or len(src) < 1 or not np.isfinite(np.r_[src.ravel(), init, ts]).all():
        raise ValueError("body trajectory inputs must be finite and aligned")
    if not np.isfinite(weight) or not 0 <= float(weight) <= 1:
        raise ValueError("weight must be in [0,1]")
    if c.smooth_window < 1 or c.max_residual_rad <= 0 or c.max_rate_rad_s <= 0:
        raise ValueError("invalid bounded body policy")
    if np.any(np.diff(ts) <= 0):
        raise ValueError("timestamps must increase strictly")

    residual = _smooth_rows(src - init[None, :], c.smooth_window)
    residual *= float(weight)
    requested = residual.copy()
    residual = np.clip(residual, -float(c.max_residual_rad), float(c.max_residual_rad))
    residual[0] = 0.0
    residual_before_rate = residual.copy()

    # Keep every sample within the declared model range.  The ranges are
    # optional because the shared adapter is also used without MuJoCo.
    if joint_lower is not None or joint_upper is not None:
        if joint_lower is None or joint_upper is None:
            raise ValueError("joint_lower and joint_upper must be supplied together")
        lo = np.asarray(joint_lower, dtype=float) - init
        hi = np.asarray(joint_upper, dtype=float) - init
        if lo.shape != init.shape or hi.shape != init.shape:
            raise ValueError("joint limits must match body dimension")
        residual = np.minimum(np.maximum(residual, lo[None, :]), hi[None, :])

    # Rate-limit the already smoothed residual.  This forward pass is causal,
    # deterministic, and keeps frame zero canonical.
    for k in range(1, len(residual)):
        max_step = float(c.max_rate_rad_s) * float(ts[k] - ts[k - 1])
        residual[k] = residual[k - 1] + np.clip(
            residual[k] - residual[k - 1], -max_step, max_step
        )
        if joint_lower is not None:
            residual[k] = np.minimum(np.maximum(residual[k], lo), hi)

    output = init[None, :] + residual
    requested_norm = np.linalg.norm(requested, axis=1)
    clipped_norm = np.linalg.norm(residual - residual_before_rate, axis=1)
    metadata = {
        "policy": asdict(c),
        "requested_weight": float(weight),
        "source_residual_rms_rad": float(np.sqrt(np.mean((src - init[None, :]) ** 2))),
        "bounded_residual_rms_rad": float(np.sqrt(np.mean(residual**2))),
        "bounded_residual_max_rad": float(np.max(np.abs(residual))),
        "requested_residual_max_rad": float(np.max(np.abs(requested))),
        "rate_limit_max_step_rad": float(np.max(np.abs(np.diff(residual, axis=0)))) if len(residual) > 1 else 0.0,
        "clipped_frame_count": int(np.count_nonzero(clipped_norm > 1e-12)),
        "source_residual_rms_requested_rad": float(np.sqrt(np.mean(requested**2))),
        "joint_limit_clipping": bool(joint_lower is not None),
        "trajectory_starts_at_initial": bool(np.array_equal(output[0], init)),
        "requested_norm_max_rad": float(np.max(requested_norm)),
    }
    return output, metadata

