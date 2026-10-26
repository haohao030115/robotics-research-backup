"""Opt-in, geometry-aware hand clearance projection.

The producer adapters export the hand trajectory separately from the body
trajectory.  A bounded wrist/arm IK projection therefore cannot remove a
thumb--wrist overlap by itself.  This module provides a small shared hand
space policy for that case.  It only changes the two distal right-thumb
coordinates, starts at the shared pre-close phase, and leaves the opposing
finger targets and all body/arm targets untouched.

The policy is disabled unless an adapter explicitly calls
``project_hand_frame``.  MuJoCo is imported lazily so host-side inspection and
the normal unit tests do not acquire a rendering or simulator dependency.
"""

from __future__ import annotations

import numpy as np

VERSION = "shared-hand-collision-v1"


def _smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def collision_clearance_phase_weight(
    time_s: float,
    *,
    onset_s: float = 7.0,
    ramp_end_s: float = 8.0,
) -> float:
    """Return a shared phase gate for collision relief.

    Relief is zero before pre-close, rises smoothly over one second, and is
    held through lift/hold.  Keeping it active after closure avoids returning
    a thumb to a known self-contact configuration during the accepted hold.
    """

    if not np.isfinite(time_s) or not np.isfinite(onset_s) or not np.isfinite(ramp_end_s):
        raise ValueError("phase times must be finite")
    if ramp_end_s <= onset_s:
        raise ValueError("ramp_end_s must be greater than onset_s")
    return _smoothstep((float(time_s) - float(onset_s)) / (float(ramp_end_s) - float(onset_s)))


def _body_geom_ids(model, body_name: str) -> tuple[int, ...]:
    """Return collision geometry ids attached directly to ``body_name``."""

    body = model.body(body_name).id
    ids = np.flatnonzero(np.asarray(model.geom_bodyid) == int(body))
    # Visual-only geoms cannot create the contact under consideration.  Keep
    # geoms with a collision type and a nonzero contact dimension.
    ids = [int(i) for i in ids if int(model.geom_contype[i]) != 0 and int(model.geom_condim[i]) != 0]
    if not ids:
        raise ValueError(f"no collision geoms found for {body_name}")
    return tuple(ids)


def thumb_wrist_clearance(
    model,
    data,
    *,
    thumb_body_name: str | None = None,
    thumb_body_names: tuple[str, ...] = (
        "right_hand_thumb_1_link",
        "right_hand_thumb_2_link",
    ),
    wrist_body_name: str = "right_wrist_yaw_link",
    distmax_m: float = 1.0,
) -> tuple[float, dict]:
    """Measure minimum signed distance between thumb links and wrist geoms.

    MuJoCo's ``mj_geomDistance`` is signed: positive is separated and
    non-positive means geometric overlap/contact.  The returned diagnostics
    identify the closest pair and are suitable for provenance.
    The default covers both thumb-1 and thumb-2 because the strict native
    evaluator reports either link as robot self-contact.  The selected pair
    is included in the returned diagnostics.
    """

    if not np.isfinite(distmax_m) or float(distmax_m) <= 0:
        raise ValueError("distmax_m must be positive and finite")
    import mujoco

    if thumb_body_name is not None:
        thumb_body_names = (str(thumb_body_name),)
    if not thumb_body_names:
        raise ValueError("at least one thumb body is required")
    thumb_pairs = []
    for name in thumb_body_names:
        thumb_pairs.extend((name, gid) for gid in _body_geom_ids(model, name))
    wrist_ids = _body_geom_ids(model, wrist_body_name)
    best = None
    segment = np.zeros(6, dtype=float)
    for thumb_name, thumb_id in thumb_pairs:
        for wrist_id in wrist_ids:
            fromto = np.zeros(6, dtype=float)
            value = float(mujoco.mj_geomDistance(model, data, thumb_id, wrist_id, float(distmax_m), fromto))
            item = (value, thumb_name, thumb_id, wrist_id, fromto.copy())
            if best is None or item[0] < best[0]:
                best = item
    assert best is not None
    value, thumb_name, thumb_id, wrist_id, segment = best
    return value, {
        "thumb_body": thumb_name,
        "thumb_bodies": list(thumb_body_names),
        "wrist_body": wrist_body_name,
        "thumb_geom": int(thumb_id),
        "wrist_geom": int(wrist_id),
        "signed_distance_m": float(value),
        "segment_fromto_m": segment.tolist(),
        "distance_convention": "MuJoCo mj_geomDistance signed; <=0 is overlap/contact",
    }


def project_hand_frame(
    model,
    body_data,
    hand_q,
    hand_joint_names,
    time_s: float,
    *,
    min_clearance_m: float = 0.001,
    max_relief_rad: float = 0.30,
    finite_difference_rad: float = 0.01,
    max_iterations: int = 4,
    onset_s: float = 7.0,
    ramp_end_s: float = 8.0,
) -> tuple[np.ndarray, dict]:
    """Project one hand frame away from right-thumb--wrist overlap.

    ``body_data.qpos`` must contain the already solved body/arm posture.  A
    private MuJoCo data copy evaluates the candidate hand posture, so the
    caller's body data is never mutated.  A finite-difference probe chooses
    the direction of each thumb-1/thumb-2 joint that increases clearance;
    the best direction is applied up to a bounded total change.  The result
    is blended by the phase gate and only thumb joints are eligible.

    This is a collision *projection* and does not declare a grasp.  Native
    opposition, lift, table-contact, and strict expert gates remain the sole
    acceptance authority.
    """

    hand = np.asarray(hand_q, dtype=float)
    names = tuple(hand_joint_names)
    if hand.ndim != 1 or not np.isfinite(hand).all():
        raise ValueError("hand_q must be a finite vector")
    if len(names) != len(hand):
        raise ValueError("hand_joint_names and hand_q must have matching length")
    if not np.isfinite(min_clearance_m) or float(min_clearance_m) < 0:
        raise ValueError("min_clearance_m must be finite and nonnegative")
    if not np.isfinite(max_relief_rad) or not 0 < float(max_relief_rad) <= 0.5:
        raise ValueError("max_relief_rad must be in (0, 0.5]")
    if not np.isfinite(finite_difference_rad) or not 0 < float(finite_difference_rad) <= 0.1:
        raise ValueError("finite_difference_rad must be in (0, 0.1]")
    if int(max_iterations) < 1 or int(max_iterations) > 16:
        raise ValueError("max_iterations must be in [1, 16]")

    out = hand.copy()
    weight = collision_clearance_phase_weight(time_s, onset_s=onset_s, ramp_end_s=ramp_end_s)
    info = {
        "enabled": True,
        "phase_weight": float(weight),
        "onset_s": float(onset_s),
        "ramp_end_s": float(ramp_end_s),
        "min_clearance_m": float(min_clearance_m),
        "max_relief_rad": float(max_relief_rad),
        "finite_difference_rad": float(finite_difference_rad),
        "iterations": 0,
        "selected_joints": [],
        "requested_delta_rad": [0.0] * len(hand),
        "applied_delta_rad": [0.0] * len(hand),
        "before_signed_distance_m": None,
        "after_signed_distance_m": None,
        "projected": False,
    }
    if weight <= 0.0:
        info["enabled"] = False
        info["reason"] = "before_preclose_phase"
        info["clearance_target_met"] = False
        return out, info

    import mujoco

    # Find the exact distal joints by suffix so both producer joint-order
    # lists can use the same policy.  A missing joint is a hard opt-in error;
    # silently applying a different finger would invalidate provenance.
    eligible = [i for i, name in enumerate(names) if name.endswith("thumb_1_joint") or name.endswith("thumb_2_joint")]
    if not eligible:
        raise ValueError("hand joint order has no thumb_1_joint/thumb_2_joint")
    qadr = np.array([model.joint(name).qposadr[0] for name in names], dtype=int)
    probe = mujoco.MjData(model)
    probe.qpos[:] = body_data.qpos

    def evaluate(q):
        probe.qpos[qadr] = q
        mujoco.mj_forward(model, probe)
        return thumb_wrist_clearance(model, probe)

    before, before_meta = evaluate(out)
    info["before_signed_distance_m"] = float(before)
    info["before_geometry"] = before_meta
    if before >= float(min_clearance_m):
        info["after_signed_distance_m"] = float(before)
        info["reason"] = "already_clear"
        info["clearance_target_met"] = True
        return out, info

    # Search the bounded local interval instead of extrapolating a single
    # finite-difference slope.  Thumb geometry is piecewise and the closest
    # link can switch from thumb-1 to thumb-2; a purely local gradient can
    # therefore move one joint into a deeper overlap after a few steps.
    delta = np.zeros_like(out)
    remaining = float(max_relief_rad)
    for iteration in range(int(max_iterations)):
        current, _ = evaluate(out)
        if current >= float(min_clearance_m) or remaining <= 1e-12:
            break
        crossing = None
        best = None
        for idx in eligible:
            # Probe both directions; ARDY's negative distal thumb target is
            # relieved by a positive change, while another producer/pose may
            # require the opposite sign.
            for sign in (-1.0, 1.0):
                max_step = min(float(max_relief_rad), remaining)
                magnitudes = np.arange(
                    float(finite_difference_rad), max_step + 0.5 * float(finite_difference_rad),
                    float(finite_difference_rad),
                )
                if len(magnitudes) == 0 or magnitudes[-1] < max_step - 1e-10:
                    magnitudes = np.r_[magnitudes, max_step]
                for magnitude in magnitudes:
                    candidate = out.copy()
                    candidate[idx] += sign * float(magnitude)
                    value, _ = evaluate(candidate)
                    gain = float(value - current)
                    item = (gain, idx, sign, float(magnitude), value, candidate)
                    if value >= float(min_clearance_m) and (crossing is None or magnitude < crossing[3]):
                        crossing = item
                    if best is None or gain > best[0]:
                        best = item
        assert best is not None
        chosen = crossing if crossing is not None else best
        gain, idx, sign, step, probed, candidate = chosen
        if gain <= 1e-10:
            info["reason"] = "no_positive_clearance_direction"
            break
        out[:] = candidate
        delta[idx] += sign * float(step)
        remaining -= float(step)
        info["selected_joints"].append(names[idx])
        info["iterations"] = iteration + 1

    final, final_meta = evaluate(out)
    # Blend only after determining the collision-free candidate.  This keeps
    # the frame synchronised and makes pre-close samples exactly unchanged.
    blended = hand + float(weight) * (out - hand)
    info["requested_delta_rad"] = delta.tolist()
    info["applied_delta_rad"] = (blended - hand).tolist()
    info["after_signed_distance_m"] = float(final)
    blended_distance = float(evaluate(blended)[0])
    info["after_blended_signed_distance_m"] = blended_distance
    info["after_geometry"] = final_meta
    info["projected"] = bool(np.max(np.abs(blended - hand)) > 1e-12)
    info["projection_target_met"] = bool(final >= float(min_clearance_m))
    info["clearance_target_met"] = bool(blended_distance >= float(min_clearance_m))
    info["blending_preserves_nonthumb"] = bool(
        np.allclose((blended - hand)[[i for i in range(len(hand)) if i not in eligible]], 0.0)
    )
    return blended, info

