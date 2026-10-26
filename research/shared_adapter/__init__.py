"""Shared task-conditioned adaptation primitives for Kimodo and ARDY.

The module is deliberately independent of either native model.  Producers
convert their output to the canonical arrays, then call these primitives.
"""
from .adapter import AdapterConfig, AdaptedMotion, UnifiedMotionAdapter, task_conditioned_clearance
from .hand import HandMotionConfig, HandMotionGenerator
from .validation import UnifiedValidationPolicy, ValidationResult
from .body import BodyPreservationConfig, bounded_body_trajectory
from .grasp_geometry import (
    approach_retraction_m,
    approach_retraction_weight,
    opposition_centering_correction,
    opposition_centering_weight,
)
from .hand_collision import (
    collision_clearance_phase_weight,
    thumb_wrist_clearance,
    project_hand_frame,
)

__all__ = [
    "AdapterConfig", "AdaptedMotion", "UnifiedMotionAdapter", "task_conditioned_clearance",
    "HandMotionConfig", "HandMotionGenerator",
    "UnifiedValidationPolicy", "ValidationResult",
    "BodyPreservationConfig", "bounded_body_trajectory",
    "approach_retraction_m", "approach_retraction_weight",
    "opposition_centering_correction", "opposition_centering_weight",
    "collision_clearance_phase_weight", "thumb_wrist_clearance", "project_hand_frame",
]
