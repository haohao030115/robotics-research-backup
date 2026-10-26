"""Task-conditioned motion adaptation shared by native producers."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib, json
import numpy as np


def task_conditioned_clearance(model, block_xy, *, data=None, target_edge_margin_m=0.10,
                               gain=0.5, max_extra_m=0.05):
    """Derive an approach height margin from the actual tabletop geometry.

    A cube near a tabletop edge leaves less horizontal clearance for the wrist
    and forearm while crossing the table rim.  The extra vertical margin is
    therefore computed from the scene's table and cube geoms, rather than from
    a position name or a seed.  ``target_edge_margin_m`` is a conservative
    policy bound; ``gain`` and ``max_extra_m`` keep the correction bounded.
    This function is deliberately read-only and has no producer-specific
    constants beyond the policy bound.
    """
    if model is None or not hasattr(model, "geom"):
        raise ValueError("model with task_table_top/task_cube_geom is required")
    xy = np.asarray(block_xy, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("block_xy must be two finite values")
    table = model.geom("task_table_top")
    cube = model.geom("task_cube_geom")
    if data is not None and hasattr(data, "geom_xpos"):
        center = np.asarray(data.geom_xpos[table.id][:2], dtype=float)
    else:
        center = np.asarray(model.geom_pos[table.id][:2], dtype=float)
    half = np.asarray(model.geom_size[table.id][:2], dtype=float)
    # Include the cube footprint: the usable edge margin is measured from the
    # cube's outer face, not its center.
    footprint = np.asarray(model.geom_size[cube.id][:2], dtype=float)
    edge_margin = np.min(half - np.abs(xy - center) - footprint)
    deficit = max(0.0, float(target_edge_margin_m) - float(edge_margin))
    extra = np.clip(float(gain) * deficit, 0.0, float(max_extra_m))
    return float(extra), {"block_xy": xy.tolist(), "table_center_xy": center.tolist(),
                          "table_half_extent_xy": half.tolist(),
                          "cube_half_extent_xy": footprint.tolist(),
                          "edge_margin_m": float(edge_margin),
                          "target_edge_margin_m": float(target_edge_margin_m),
                          "gain": float(gain), "extra_clearance_m": float(extra)}

@dataclass(frozen=True)
class AdapterConfig:
    version: str = "unified-adapter-v1"
    hz: float = 50.0
    preserve_motion_weight: float = .75
    smooth_window: int = 9
    max_translation_m: float = .10
    max_joint_delta_rad: float = .35
    reach_margin_m: float = .02
    wrist_joint_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5)

@dataclass
class AdaptedMotion:
    native: np.ndarray
    canonical: np.ndarray
    planned: np.ndarray
    final: np.ndarray
    corrections: dict
    provenance: dict

def _smooth_rows(x, width):
    if width <= 1: return x.copy()
    width = int(width) | 1; kernel = np.ones(width)/width
    pad = width // 2
    padded = np.pad(x, ((pad, pad), (0, 0)), mode='edge')
    return np.stack([np.convolve(padded[:, j], kernel, mode='valid') for j in range(x.shape[1])], 1)

class UnifiedMotionAdapter:
    """Adapt body joint trajectories while retaining native motion residuals.

    ``task`` may contain ``cube_xy``, ``reference_cube_xy`` and ``approach``.
    The geometric translation is bounded and rejected when outside the declared
    reach corridor; no per-position calibration is inferred.
    """
    VERSION = "unified-adapter-v1"
    def __init__(self, config: AdapterConfig | None = None): self.config = config or AdapterConfig()

    def adapt(self, native_q, *, task=None, seed=0, producer="unknown"):
        q = np.asarray(native_q, float)
        if q.ndim != 2 or q.shape[1] < 6 or not np.isfinite(q).all(): raise ValueError("native_q must be finite [T,J]")
        task = task or {}; c = self.config
        canonical = q.copy()
        delta = np.asarray(task.get("cube_xy", (0., 0.)), float) - np.asarray(task.get("reference_cube_xy", (0., 0.)), float)
        if delta.shape != (2,): raise ValueError("cube_xy/reference_cube_xy must be length two")
        distance = float(np.linalg.norm(delta))
        if distance > c.max_translation_m + c.reach_margin_m: raise ValueError("task outside bounded reach corridor")
        planned = canonical.copy()
        # The first six canonical columns are the producer-neutral wrist pose
        # residual convention (xyz + axis-angle). Keep all other body motion.
        translation = np.array([delta[0], delta[1], 0., 0., 0., 0.])
        planned[:, :6] += translation[None, :]
        alpha = float(np.clip(1.-c.preserve_motion_weight, 0., 1.))
        final = canonical + alpha*(planned-canonical)
        # Smooth only the bounded correction.  Native residual motion and
        # producer-specific joints therefore remain byte-for-byte preserved.
        if c.smooth_window > 1: final = canonical + _smooth_rows(final-canonical, c.smooth_window)
        final[:, :6] = np.clip(final[:, :6], canonical[:, :6]-c.max_joint_delta_rad, canonical[:, :6]+c.max_joint_delta_rad)
        corrections = {"cube_delta_xy_m": delta.tolist(), "max_translation_m": distance,
                       "native_to_canonical_max": 0.,
                       "canonical_to_planned_max": float(np.max(np.linalg.norm(planned-canonical, axis=1))),
                       "planned_to_final_max": float(np.max(np.linalg.norm(final-planned, axis=1))),
                       "motion_preservation_weight": c.preserve_motion_weight,
                       "coordinate_convention": "canonical body columns; producer conversion happens before adapter"}
        provenance = {"adapter_version": self.VERSION, "config": asdict(c), "producer": producer,
                      "seed": int(seed), "task": task,
                      "config_sha256": hashlib.sha256(json.dumps(asdict(c), sort_keys=True, default=list).encode()).hexdigest()}
        return AdaptedMotion(q, canonical, planned, final, corrections, provenance)

    @staticmethod
    def save(result: AdaptedMotion, path):
        path = __import__('pathlib').Path(path); path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path/'motion_stages.npz', native=result.native, canonical=result.canonical,
                            planned=result.planned, final=result.final)
        (path/'adapter_provenance.json').write_text(json.dumps({"corrections": result.corrections,
            **result.provenance}, indent=2, sort_keys=True)+'\n')
