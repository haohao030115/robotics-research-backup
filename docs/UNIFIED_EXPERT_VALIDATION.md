# Unified expert validation

`UnifiedValidationPolicy` is the common final gate. It reports `physical_success`, `contact_safe`, `dataset_valid`, and `expert_valid` separately. Meaningful contacts are limited to `hand_cube`, `cube_table`, and `foot_support`; contacts at or below the fixed numerical-noise force are ignored. Any other meaningful contact rejects the episode. Dataset checks are fail-closed, and expert acceptance requires all three preceding dimensions.

Producer reports remain diagnostics and cannot replace this cross-producer policy. The policy version and result are serializable with `UnifiedValidationPolicy.write()`.
