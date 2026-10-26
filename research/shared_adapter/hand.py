"""Reproducible, phase-conditioned 14-DoF G1 hand references.

The schedule is generated at 50 Hz and uses smoothstep phase curves.  Timing
variation is sampled as one correlated profile, so closure cannot begin before
the approach phase simply because independent joint noise was sampled.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class HandMotionConfig:
    hz: float = 50.0
    # The validated tabletop schedule enters preclose at seven seconds and
    # reaches full closure at ten seconds. Variation stays around that
    # physically feasible corridor rather than introducing a late grasp.
    preclose_on_s: float = 7.0
    preclose_duration_s: float = 1.0
    fullclose_s: float = 10.0
    settle_s: float = 1.0
    hold_s: float = 4.0
    preclose_fraction: float = 0.4
    # The production baseline is the validated timing corridor. Experiments
    # can opt into bounded timing variation through HandMotionConfig.
    timing_jitter_s: float = 0.0
    closure_speed: float = 1.0
    variation: float = 0.0
    # A small shared thumb lead lets the thumb establish opposition before
    # the index/middle pads finish closing.  Zero preserves the validated
    # synchronized baseline; opt-in profiles remain body-frame aligned.
    thumb_lead_s: float = 0.0
    # The thumb opposition joints need a deeper final curl to keep the thumb
    # pad in the cube grasp corridor.  The prior -1.0/-1.25 profile was
    # physically reproducible but grazed the tabletop in native replay; the
    # bounded -1.1/-1.375 profile preserves synchronized timing and is the
    # validated Kimodo center profile. ARDY selects its own bounded profile
    # through the same generator below because its wrist trajectory has a
    # different native contact corridor.
    right_final: tuple[float, ...] = (1.2, 1.2, 1.2, 1.2, 0., -1.1, -1.375)
    left_final: tuple[float, ...] = (0.,) * 7
    joint_limits: tuple[tuple[float, float], ...] = ((-2.5, 2.5),) * 14


# Named profiles are deliberately small, task-conditioned changes around the
# validated timing corridor.  Keeping them here makes Kimodo and ARDY use the
# same generator and records a stable policy name in provenance instead of
# embedding producer-specific schedules in their adapters.
def profile_config(name: str) -> HandMotionConfig:
    """Return a validated shared hand policy by name.

    ``default`` is the historical accepted Kimodo policy.  ``ardy`` changes
    only the thumb opposition target to match ARDY's measured wrist corridor.
    ``slow_close`` and ``early_preshape`` are bounded timing/preshape
    experiments. ``timing_jitter`` and ``coordination_mild`` expose small
    correlated timing and finger-shape perturbations for native replay. The
    lower-amplitude ``coordination_tiny`` profile is useful for ARDY's tighter
    contact corridor. ``ardy_thumb_relief`` is a bounded ARDY-only thumb
    clearance variant: it changes only the two distal thumb targets by
    +0.05 rad, preserving the shared phase and finger targets. The stronger
    ``ardy_thumb_relief_strong`` probe uses the same geometry-derived
    direction at +0.30 rad for a native right-side collision ablation. All
    experiments remain opt-in so existing accepted references are bit-for-bit
    reproducible.
    """
    if name == "default":
        return HandMotionConfig()
    if name == "ardy":
        return HandMotionConfig(right_final=(1.2, 1.2, 1.2, 1.2, 0., -1.3, -1.625))
    if name == "ardy_thumb_relief":
        # Fresh ARDY center replays showed a deterministic thumb--wrist
        # collision during closure. This small, opt-in reduction in thumb
        # curl keeps the accepted ``ardy`` profile bit-for-bit unchanged and
        # stays within the declared joint limits.
        return HandMotionConfig(right_final=(1.2, 1.2, 1.2, 1.2, 0., -1.25, -1.575))
    if name == "ardy_thumb_relief_balanced":
        # Midpoint of the measured contact/no-contact sweep.  This remains a
        # shared hand geometry profile and does not encode the right scene.
        return HandMotionConfig(right_final=(1.2, 1.2, 1.2, 1.2, 0., -1.05, -1.375))
    if name == "ardy_thumb_relief_medium":
        # Intermediate geometry probe between the calibrated ARDY curl and
        # the strong zero-overlap relief.  The offset is shared across
        # positions; it is chosen from the MuJoCo hand-link clearance sweep,
        # not from a scene-specific wrist or cube coordinate.
        return HandMotionConfig(right_final=(1.2, 1.2, 1.2, 1.2, 0., -1.1, -1.425))
    if name == "ardy_thumb_relief_strong":
        # Static MuJoCo contact probing found the first zero-overlap margin at
        # +0.30 rad relative to the ARDY thumb target. Keep the same phase and
        # all non-thumb joints; this is a diagnostic shared profile, not a
        # position-specific offset.
        return HandMotionConfig(right_final=(1.2, 1.2, 1.2, 1.2, 0., -1.0, -1.325))
    if name == "ardy_thumb_lead_strong":
        return HandMotionConfig(right_final=(1.2, 1.2, 1.2, 1.2, 0., -1.0, -1.325), thumb_lead_s=0.5)
    if name == "slow_close":
        return HandMotionConfig(preclose_on_s=7.25, preclose_duration_s=1.25,
                                fullclose_s=10.5, settle_s=1.0)
    if name == "early_preshape":
        return HandMotionConfig(preclose_on_s=6.75, preclose_duration_s=1.25,
                                fullclose_s=10.0, settle_s=1.0,
                                preclose_fraction=0.30)
    if name == "timing_jitter":
        # A small correlated timing perturbation is sampled once per episode;
        # the four phase boundaries move together within +/-120 ms.  This is
        # deliberately bounded so the hand remains synchronized to the 50 Hz
        # body reference while exposing timing diversity in native replay.
        return HandMotionConfig(timing_jitter_s=0.12)
    if name == "coordination_mild":
        # Keep the shared phase and terminal targets fixed, but vary the
        # finger-to-thumb preshape by at most eight percent.  The generator
        # applies one correlated factor to all fingers, avoiding independent
        # joint noise and preserving a physically coordinated grasp.
        return HandMotionConfig(variation=0.08)
    if name == "coordination_tiny":
        # A low-amplitude coordinated profile for the ARDY contact corridor.
        # The sampled factor remains bounded and keeps the phase schedule
        # identical to the validated profile.
        return HandMotionConfig(variation=0.02)
    raise ValueError(f"unknown shared hand profile: {name}")

def _smooth(x):
    x = np.clip(x, 0., 1.)
    return x*x*(3.-2.*x)

class HandMotionGenerator:
    VERSION = "shared-hand-v1"
    def __init__(self, config: HandMotionConfig | None = None):
        self.config = config or HandMotionConfig()

    def generate(self, time_s, *, seed: int = 0, phase=None,
                 right_final=None, left_final=None):
        t = np.asarray(time_s, dtype=float)
        if t.ndim != 1 or len(t) < 2 or not np.all(np.diff(t) > 0):
            raise ValueError("time_s must be a strictly increasing 1-D array")
        c = self.config; rng = np.random.default_rng(seed)
        # One profile controls all fingers; relative coordination is fixed.
        jitter = rng.uniform(-c.timing_jitter_s, c.timing_jitter_s, 4)
        onset = max(0., c.preclose_on_s + jitter[0])
        pre_end = max(onset + .05, onset + c.preclose_duration_s + jitter[1])
        full = max(pre_end + .05, c.fullclose_s + jitter[2])
        settle_end = max(full + .05, full + c.settle_s + jitter[3])
        if phase is not None:
            phase = np.asarray(phase, dtype=float)
            if phase.shape != t.shape: raise ValueError("phase shape mismatch")
            # A supplied phase is a generation-time signal, never future contact.
            phase_u = np.clip(phase, 0., 1.)
        else:
            pre = float(np.clip(c.preclose_fraction, 0., 1.))
            phase_u = np.where(t < pre_end,
                pre * _smooth((t-onset)/(pre_end-onset)),
                pre + (1.-pre) * _smooth((t-pre_end)/(full-pre_end)))
            phase_u = np.clip(phase_u, 0., 1.)
        speed = float(np.clip(c.closure_speed, .25, 2.))
        # The piecewise schedule is already smooth at each boundary. Applying
        # another smoothstep here would delay closure and alter the validated
        # preclose fraction; speed changes the exponent directly.
        phase_u = np.clip(phase_u ** (1./speed), 0., 1.)
        right = np.asarray(right_final if right_final is not None else c.right_final, float)
        left = np.asarray(left_final if left_final is not None else c.left_final, float)
        if right.shape != (7,) or left.shape != (7,): raise ValueError("hand targets must have seven joints per side")
        # Small coordinated shape variation: the index/middle finger spread
        # varies together, while the thumb opposition target remains fixed in
        # the producer-calibrated grasp corridor.
        spread = rng.uniform(-c.variation, c.variation)
        profile = np.array([1.+spread, 1.-spread*.5, 1.-spread*.5, 1.-spread*.5,
                            1., 1., 1.])
        thumb_phase = np.clip(np.interp(t + float(c.thumb_lead_s), t, phase_u, left=0., right=1.), 0., 1.)
        out = np.zeros((len(t), 14), float)
        out[:, :7] = phase_u[:, None] * left[None, :]
        out[:, 7:] = phase_u[:, None] * (right * profile)[None, :]
        # The shared order is index_0/index_1, middle_0/middle_1,
        # thumb_0/thumb_1/thumb_2 on each side.  Keep index/middle timing
        # unchanged while leading only the actual opposition pads.
        out[:, 11:14] = thumb_phase[:, None] * (right[4:7] * profile[4:7])[None, :]
        limits = np.asarray(c.joint_limits, float)
        out = np.clip(out, limits[:, 0], limits[:, 1])
        # Force a canonical initial state and compute the 50 Hz-compatible slope.
        out[0] = 0.
        return out, {"generator_version": self.VERSION, "seed": int(seed),
            "onset_s": onset, "preclose_end_s": pre_end, "fullclose_s": full,
            "settle_end_s": settle_end, "closure_speed": speed,
            "variation": spread, "preclose_fraction": c.preclose_fraction,
            "thumb_lead_s": float(c.thumb_lead_s),
            "right_final": right.tolist(), "left_final": left.tolist(),
            "reference_hz": c.hz}

    def velocities(self, time_s, hand_q):
        return np.gradient(np.asarray(hand_q, float), np.asarray(time_s, float), axis=0)
