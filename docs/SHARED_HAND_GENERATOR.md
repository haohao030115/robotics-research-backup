# Shared hand generator

`HandMotionGenerator` creates a smooth 14-DoF reference at 50 Hz. One seeded timing profile controls preclose onset, preclose end, full closure, and settling. Finger coordination varies as a bounded shape profile, while joint limits and the zero initial hand state are enforced. An optional phase array is a generation-time signal and cannot depend on future contact observations.

Returned metadata records the seed, timing, closure speed, and generator version. The V4 future-hand representation remains `hand_ref_q[H,14]`.
