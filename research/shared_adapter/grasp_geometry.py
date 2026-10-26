"""Geometry-derived approach policy shared by producer adapters.

The hand reaches the cube through a short corridor before terminal closure.
The corridor retracts the wrist along the cube-to-wrist horizontal direction
by a bounded amount derived from the immutable cube and distal hand geometry.
It is deliberately independent of the scene position and generator seed.
"""

from __future__ import annotations

import numpy as np


# The canonical SONIC hand corridor uses a wrist frame whose x axis points
# along the forearm and whose y/z axes present the distal fingers to the cube.
# A small, fixed x tilt is a producer-neutral geometry policy: it is derived
# from the common wrist frame and does not inspect the case name or generator
# seed.  Native contact, collision, and lift gates remain authoritative.
CANONICAL_WRIST_X_TILT_RAD = float(np.deg2rad(-20.0))


def geometry_wrist_orientation(*, x_tilt_rad: float = CANONICAL_WRIST_X_TILT_RAD) -> np.ndarray:
    """Return the shared geometry-derived terminal wrist orientation.

    ``x_tilt_rad`` is exposed for offline screening only; production callers
    use the fixed module default so a position-specific orientation lookup
    cannot be hidden in an adapter run.
    """
    if not np.isfinite(x_tilt_rad):
        raise ValueError("x_tilt_rad must be finite")
    c, s = np.cos(float(x_tilt_rad)), np.sin(float(x_tilt_rad))
    x_rotation = np.array([[1.0, 0.0, 0.0],
                           [0.0, c, -s],
                           [0.0, s, c]], dtype=float)
    canonical = np.array([[1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [0.0, -1.0, 0.0]], dtype=float)
    return x_rotation @ canonical


def approach_retraction_m(model, *, cube_geom_name: str = "task_cube_geom") -> tuple[float, dict]:
    """Return a bounded pre-close retraction from model geometry.

    Half the cube side plus the largest distal-finger half-width gives a
    conservative contact corridor.  Capping at 4 cm prevents the policy from
    turning into a large task-specific IK correction on non-table scenes.
    """
    cube = model.geom(cube_geom_name).id
    cube_half = float(np.max(np.asarray(model.geom_size[cube][:2], dtype=float)))
    distal_names = (
        "right_hand_thumb_2_link",
        "right_hand_middle_1_link",
        "right_hand_index_1_link",
    )
    widths = []
    for body_name in distal_names:
        body = model.body(body_name).id
        for geom_id in np.flatnonzero(model.geom_bodyid == body):
            if model.geom_contype[geom_id]:
                widths.append(float(np.max(np.asarray(model.geom_size[geom_id][:2], dtype=float))))
    distal_half_width = max(widths, default=0.01)
    value = float(np.clip(0.5 * cube_half + distal_half_width, 0.0, 0.04))
    height_offset = -float(0.5 * cube_half + 0.5 * distal_half_width)
    return value, {
        "cube_half_extent_m": cube_half,
        "distal_finger_half_width_m": distal_half_width,
        "formula": "clip(0.5*cube_half_extent + max(distal_finger_xy_half_width), 0, 0.04)",
        "retraction_m": value,
        "grasp_height_offset_m": height_offset,
    }


def approach_retraction_weight(time_s: float) -> float:
    """Smoothly hold the corridor through closure, then release during lift."""
    t = float(time_s)
    def smooth(a: float, b: float) -> float:
        u = float(np.clip((t - a) / (b - a), 0.0, 1.0))
        return u * u * (3.0 - 2.0 * u)
    # Release as the fingers reach terminal closure.  This lets the fully
    # closed pads move onto the cube before the lift segment starts, while
    # keeping the partially closed fingers from pushing it sideways.
    return smooth(6.5, 7.5) * (1.0 - smooth(9.0, 10.0))


def opposition_centering_weight(time_s: float) -> float:
    """Return the shared preclose-to-settle weight for opposition centering.

    The correction is anchored at the start of hand preclose (7 s), rises
    continuously through 8 s, and releases during the 10--11 s settle window.
    Keeping this timing in the shared geometry module prevents Kimodo and ARDY
    adapters from applying different transient hand policies.
    """
    t = float(time_s)
    def smooth(a: float, b: float) -> float:
        u = float(np.clip((t - a) / (b - a), 0.0, 1.0))
        return u * u * (3.0 - 2.0 * u)
    return smooth(7.0, 8.0) * (1.0 - smooth(10.0, 11.0))


def distal_alignment_weight(time_s: float) -> float:
    """Gate 3-D distal alignment to preclose and release before lift."""

    t = float(time_s)
    def smooth(a: float, b: float) -> float:
        u = float(np.clip((t - a) / (b - a), 0.0, 1.0))
        return u * u * (3.0 - 2.0 * u)
    return smooth(7.0, 8.0) * (1.0 - smooth(9.0, 10.0))


def opposition_centering_correction(
    model,
    data,
    *,
    cube_body_name: str = "task_red_cube",
    thumb_body_name: str = "right_hand_thumb_2_link",
    finger_body_names: tuple[str, ...] = (
        "right_hand_index_1_link",
        "right_hand_middle_1_link",
    ),
    max_correction_m: float = 0.03,
    dominant_axis: bool = False,
) -> tuple[np.ndarray, dict]:
    """Compute a bounded horizontal correction for an opposing hand pair.

    A side-dependent wrist trajectory can reach the cube while leaving the
    thumb and either opposing finger on the same side of its centre.  That
    geometry produces a lateral shove at first contact, which is visible in
    the native right-side traces.  This helper measures the actual MuJoCo
    distal-link centres and moves the wrist target toward the centre of the
    best thumb/index-or-middle pair.  It contains no position or seed lookup:
    the cube and hand body names are canonical model geometry only.

    The returned correction is horizontal and bounded.  Adapters anchor it at
    the start of the preclose window (7.0 s), apply it only through a smooth
    preclose/closure weight, and record the selected pair and raw errors in
    provenance.  Keeping the z component zero prevents
    this policy from weakening the table-clearance gate.
    """
    if model is None or data is None or not hasattr(model, "body"):
        raise ValueError("model and data with MuJoCo body access are required")
    if not np.isfinite(max_correction_m) or float(max_correction_m) <= 0:
        raise ValueError("max_correction_m must be positive and finite")

    def body_center(name: str) -> np.ndarray:
        ident = model.body(name).id
        value = np.asarray(data.xpos[ident], dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"invalid world position for {name}")
        return value

    cube = body_center(cube_body_name)
    thumb = body_center(thumb_body_name)
    candidates = []
    for finger_name in finger_body_names:
        finger = body_center(finger_name)
        midpoint = 0.5 * (thumb + finger)
        error = cube - midpoint
        horizontal = error.copy()
        horizontal[2] = 0.0
        norm = float(np.linalg.norm(horizontal))
        candidates.append((norm, finger_name, midpoint, horizontal))
    if not candidates:
        raise ValueError("at least one opposing finger body is required")
    # Pick the geometrically closest valid opposition pair.  The tie-break is
    # lexical and therefore deterministic across producers and positions.
    norm, finger_name, midpoint, horizontal = min(candidates, key=lambda x: (x[0], x[1]))
    scale = min(1.0, float(max_correction_m) / max(norm, 1e-12))
    if dominant_axis:
        # A side approach can have a small cross-table error whose correction
        # would steer the wrist toward the table edge.  Retain only the
        # largest horizontal component while keeping the same geometry-derived
        # cap; this remains position/seed independent.
        dominant = int(np.argmax(np.abs(horizontal[:2])))
        axis = np.zeros(3, dtype=float)
        axis[dominant] = horizontal[dominant]
        axis_norm = float(np.linalg.norm(axis))
        correction = axis * min(1.0, float(max_correction_m) / max(axis_norm, 1e-12))
    else:
        correction = horizontal * scale
    return correction, {
        "cube_body": cube_body_name,
        "thumb_body": thumb_body_name,
        "selected_finger_body": finger_name,
        "pair_midpoint_world": midpoint.tolist(),
        "cube_center_world": cube.tolist(),
        "raw_horizontal_error_m": float(norm),
        "correction_world_m": correction.tolist(),
        "correction_norm_m": float(np.linalg.norm(correction)),
        "max_correction_m": float(max_correction_m),
        "dominant_axis_only": bool(dominant_axis),
        "policy": "best thumb/index-or-middle horizontal opposition midpoint",
    }


def distal_alignment_correction(
    model,
    data,
    *,
    cube_body_name: str = "task_red_cube",
    thumb_body_name: str = "right_hand_thumb_2_link",
    finger_body_names: tuple[str, ...] = (
        "right_hand_index_1_link",
        "right_hand_middle_1_link",
    ),
    max_correction_m: float = 0.04,
) -> tuple[np.ndarray, dict]:
    """Return a bounded 3-D correction toward the best distal-pair midpoint.

    This is an opt-in pre-close alignment policy.  It uses only the current
    MuJoCo geometry and therefore does not encode a scene-position or seed
    lookup.  Unlike :func:`opposition_centering_correction`, the correction
    includes the vertical component and is intended to be released before
    lift.  The caller remains responsible for applying the bounded vector
    through a smooth pre-close/closure window and for keeping native contact
    and collision gates authoritative.
    """
    if model is None or data is None or not hasattr(model, "body"):
        raise ValueError("model and data with MuJoCo body access are required")
    if not np.isfinite(max_correction_m) or float(max_correction_m) <= 0:
        raise ValueError("max_correction_m must be positive and finite")

    def body_center(name: str) -> np.ndarray:
        ident = model.body(name).id
        value = np.asarray(data.xpos[ident], dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"invalid world position for {name}")
        return value

    cube = body_center(cube_body_name)
    thumb = body_center(thumb_body_name)
    candidates = []
    for finger_name in finger_body_names:
        finger = body_center(finger_name)
        midpoint = 0.5 * (thumb + finger)
        error = cube - midpoint
        norm = float(np.linalg.norm(error))
        candidates.append((norm, finger_name, midpoint, error))
    if not candidates:
        raise ValueError("at least one opposing finger body is required")
    norm, finger_name, midpoint, error = min(candidates, key=lambda x: (x[0], x[1]))
    scale = min(1.0, float(max_correction_m) / max(norm, 1e-12))
    correction = error * scale
    return correction, {
        "cube_body": cube_body_name,
        "thumb_body": thumb_body_name,
        "selected_finger_body": finger_name,
        "pair_midpoint_world": midpoint.tolist(),
        "cube_center_world": cube.tolist(),
        "raw_error_m": float(norm),
        "correction_world_m": correction.tolist(),
        "correction_norm_m": float(np.linalg.norm(correction)),
        "max_correction_m": float(max_correction_m),
        "policy": "best thumb/index-or-middle 3-D distal midpoint alignment",
    }
