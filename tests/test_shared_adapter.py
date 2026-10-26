import unittest
import numpy as np
from research.shared_adapter import (HandMotionGenerator, UnifiedMotionAdapter,
    UnifiedValidationPolicy, task_conditioned_clearance)
from research.shared_adapter.hand import HandMotionConfig, profile_config
from research.shared_adapter.body import bounded_body_trajectory
from research.shared_adapter.grasp_geometry import (
    approach_retraction_weight,
    opposition_centering_correction,
    opposition_centering_weight,
)
from research.shared_adapter.hand_collision import collision_clearance_phase_weight
from research.shared_adapter.spatial_prompt import condition_prompts, descriptor

class SharedAdapterTest(unittest.TestCase):
    def test_zero_task_preserves_native_motion(self):
        q=np.random.default_rng(1).normal(size=(32,29))
        out=UnifiedMotionAdapter().adapt(q, task={'cube_xy':[0,0]})
        np.testing.assert_array_equal(out.final, q)

    def test_reach_corridor_rejects(self):
        with self.assertRaises(ValueError):
            UnifiedMotionAdapter().adapt(np.zeros((8,29)), task={'cube_xy':[.2,0]})

    def test_hand_is_smooth_and_deterministic(self):
        t=np.arange(100)/50
        a,m=HandMotionGenerator().generate(t,seed=4)
        b,_=HandMotionGenerator().generate(t,seed=4)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a.shape,(100,14)); np.testing.assert_array_equal(a[0],np.zeros(14))

    def test_contact_policy_is_shared_and_strict(self):
        p=UnifiedValidationPolicy()
        result=p.evaluate(physical_success=True,contacts=[{'kind':'hand_cube','force_n':1}],dataset_checks={'sync':True})
        self.assertTrue(result.expert_valid)
        bad=p.evaluate(physical_success=True,contacts=[{'kind':'arm_table','force_n':1}],dataset_checks={'sync':True})
        self.assertFalse(bad.expert_valid); self.assertFalse(bad.contact_safe)

    def test_native_gate_rejects_robot_self_and_environment_contacts(self):
        p = UnifiedValidationPolicy()
        base = {
            'physical_success': True,
            'completed': True,
            'frame_coverage_complete': True,
            'support_contact_steps': 0,
            'robot_table_contact_steps': 0,
            'robot_self_contact_steps': 0,
            'left_arm_environment_contact_steps': 0,
            'inactive_arm_environment_contact_steps': 0,
        }
        self.assertTrue(p.evaluate_native_result(base).expert_valid)
        for field in ('robot_self_contact_steps',
                      'left_arm_environment_contact_steps',
                      'inactive_arm_environment_contact_steps'):
            bad = dict(base, **{field: 1})
            verdict = p.evaluate_native_result(bad)
            self.assertFalse(verdict.expert_valid, field)
            self.assertFalse(verdict.contact_safe, field)

    def test_native_gate_keeps_table_alias_when_support_is_none(self):
        p = UnifiedValidationPolicy()
        result = p.evaluate_native_result({
            'physical_success': True,
            'completed': True,
            'support_contact_steps': None,
            'robot_table_contact_steps': 1,
        })
        self.assertFalse(result.expert_valid)
        self.assertFalse(result.contact_safe)

    def test_native_gate_checks_nested_shared_contact_diagnostics(self):
        p = UnifiedValidationPolicy()
        result = p.evaluate_native_result({
            'physical_success': True,
            'completed': True,
            'support_contact_steps': 0,
            'original_shared_evaluator_report': {
                'robot_table_contact_steps': 0,
                'robot_self_contact_steps': 1,
                'left_arm_environment_contact_steps': 0,
                'inactive_arm_environment_contact_steps': 0,
            },
        })
        self.assertFalse(result.expert_valid)
        self.assertFalse(result.contact_safe)

    def test_native_gate_rejects_explicit_integrity_and_policy_failures(self):
        p = UnifiedValidationPolicy()
        base = {
            'physical_success': True,
            'completed': True,
            'integrity_pass': True,
            'encoder_mode_0_verified': True,
            'foreign_processes': [],
            'policy_reference_matches': True,
        }
        self.assertTrue(p.evaluate_native_result(base).expert_valid)
        failures = (
            ('integrity_pass', False, 'dataset:integrity_pass'),
            ('encoder_mode_0_verified', False, 'dataset:encoder_mode_0_verified'),
            ('foreign_processes', ['other_native_process'], 'dataset:foreign_processes_empty'),
            ('policy_reference_matches', False, 'dataset:policy_reference_matches'),
        )
        for field, value, reason in failures:
            verdict = p.evaluate_native_result(dict(base, **{field: value}))
            self.assertFalse(verdict.expert_valid, field)
            self.assertIn(reason, verdict.reasons)

    def test_named_hand_profiles_are_bounded_and_distinct(self):
        t=np.arange(900)/50
        default,_=HandMotionGenerator(profile_config('default')).generate(t,seed=2)
        ardy,_=HandMotionGenerator(profile_config('ardy')).generate(t,seed=2)
        self.assertTrue(np.isfinite(ardy).all())
        self.assertGreater(float(np.max(np.abs(default-ardy))), 0.)
        self.assertLessEqual(float(np.max(np.abs(ardy))), 2.5)

    def test_bounded_timing_preshape_and_coordination_profiles(self):
        t=np.arange(900)/50
        profiles={}
        for name in ('default','ardy','ardy_thumb_relief','ardy_thumb_relief_balanced','ardy_thumb_relief_medium','ardy_thumb_relief_strong','ardy_thumb_lead_strong','slow_close','early_preshape','timing_jitter','coordination_mild','coordination_tiny'):
            q,info=HandMotionGenerator(profile_config(name)).generate(t,seed=7)
            profiles[name]=(q,info)
            self.assertTrue(np.isfinite(q).all())
            self.assertLessEqual(float(np.max(np.abs(q))),2.5)
            self.assertEqual(q.shape,(900,14))
            # One correlated schedule must never move before the profile onset.
            self.assertGreaterEqual(info['preclose_end_s'],info['onset_s']+.05)
            self.assertGreaterEqual(info['fullclose_s'],info['preclose_end_s']+.05)
        self.assertGreater(profiles['slow_close'][1]['fullclose_s'], profiles['default'][1]['fullclose_s'])
        self.assertLess(profiles['early_preshape'][1]['preclose_fraction'], profiles['default'][1]['preclose_fraction'])
        self.assertLessEqual(abs(profiles['timing_jitter'][1]['onset_s']-7.0),.12)
        self.assertGreater(float(np.max(np.abs(profiles['coordination_mild'][0]-profiles['default'][0]))),0.)
        self.assertLessEqual(abs(profiles['coordination_tiny'][1]['variation']),.02)
        self.assertAlmostEqual(profiles['ardy_thumb_lead_strong'][1]['thumb_lead_s'], .5)
        self.assertTrue(np.allclose(profiles['ardy_thumb_relief'][0][-1, 12:14],
                                    profiles['ardy'][0][-1, 12:14] + .05))
        self.assertTrue(np.allclose(profiles['ardy_thumb_relief'][0][-1, :12],
                                    profiles['ardy'][0][-1, :12]))

    def test_thumb_lead_targets_thumb_joints_only(self):
        t=np.arange(600)/50
        cfg=HandMotionConfig(thumb_lead_s=0.5)
        q,_=HandMotionGenerator(cfg).generate(t,seed=3)
        # With a positive lead, the four index/middle joints must remain on
        # the ordinary phase while only thumb_0..thumb_2 use the shifted one.
        phase,_=HandMotionGenerator(HandMotionConfig()).generate(t,seed=3)
        self.assertTrue(np.allclose(q[:,7:11],phase[:,7:11]))
        self.assertGreater(float(np.max(np.abs(q[:,11:14]-phase[:,11:14]))),0.)

    def test_geometry_approach_corridor_weight_is_bounded_and_released(self):
        self.assertEqual(approach_retraction_weight(6.0), 0.0)
        self.assertGreater(approach_retraction_weight(8.0), 0.99)
        self.assertGreater(approach_retraction_weight(9.5), 0.0)
        self.assertEqual(approach_retraction_weight(10.5), 0.0)

    def test_opposition_centering_weight_is_continuous_and_released(self):
        self.assertEqual(opposition_centering_weight(6.9), 0.0)
        self.assertEqual(opposition_centering_weight(7.0), 0.0)
        self.assertGreater(opposition_centering_weight(7.5), 0.0)
        self.assertAlmostEqual(opposition_centering_weight(8.0), 1.0)
        self.assertGreater(opposition_centering_weight(10.5), 0.0)
        self.assertEqual(opposition_centering_weight(11.0), 0.0)

    def test_opposition_centering_is_bounded_and_geometry_selected(self):
        class Body:
            def __init__(self, ident): self.id = ident
        class Model:
            def body(self, name):
                return Body({
                    'task_red_cube': 0,
                    'right_hand_thumb_2_link': 1,
                    'right_hand_index_1_link': 2,
                    'right_hand_middle_1_link': 3,
                }[name])
        class Data:
            xpos = np.asarray([
                [0.4, -0.3, 0.785],
                [0.37, -0.29, 0.82],
                [0.46, -0.20, 0.82],
                [0.40, -0.30, 0.82],
            ])
        correction, info = opposition_centering_correction(Model(), Data())
        # Thumb + middle is the closest pair, and the policy points toward
        # the cube while remaining within the declared horizontal bound.
        self.assertEqual(info['selected_finger_body'], 'right_hand_middle_1_link')
        self.assertGreater(correction[0], 0.)
        self.assertLessEqual(float(np.linalg.norm(correction)), .03 + 1e-12)
        self.assertEqual(float(correction[2]), 0.)

    def test_task_conditioned_clearance_uses_table_edge_margin(self):
        class Geom:
            def __init__(self, ident): self.id = ident
        class Model:
            geom_pos = np.asarray([[0., 0., .72], [0., 0., .78]])
            geom_size = np.asarray([[.4, .5, .03], [.035, .035, .035]])
            def geom(self, name): return Geom(0 if name == 'task_table_top' else 1)
        class Data:
            geom_xpos = np.asarray([[0., 0., .72], [0., 0., .78]])
        model, data = Model(), Data()
        center, _ = task_conditioned_clearance(model, [.0, -.15], data=data)
        near, policy = task_conditioned_clearance(model, [-.35, -.15], data=data)
        self.assertEqual(center, 0.)
        self.assertGreater(near, center)
        self.assertLessEqual(near, .05)
        self.assertAlmostEqual(policy['edge_margin_m'], .015)

    def test_bounded_body_policy_is_deterministic_and_rate_limited(self):
        t = np.arange(21, dtype=float) / 50
        src = np.zeros((len(t), 2)); src[1:, 0] = 1.0; src[5:, 1] = -1.0
        out, info = bounded_body_trajectory(
            src, np.zeros(2), t, weight=1.0,
            joint_lower=np.array([-.5, -.5]), joint_upper=np.array([.5, .5]))
        self.assertTrue(np.array_equal(out[0], np.zeros(2)))
        self.assertLessEqual(float(np.max(np.abs(out))), .5)
        self.assertLessEqual(float(np.max(np.abs(np.diff(out, axis=0)))), 3.0/50 + 1e-12)
        self.assertGreater(info['clipped_frame_count'], 0)
        again, _ = bounded_body_trajectory(
            src, np.zeros(2), t, weight=1.0,
            joint_lower=np.array([-.5, -.5]), joint_upper=np.array([.5, .5]))
        np.testing.assert_array_equal(out, again)

    def test_collision_clearance_phase_gate_is_synchronized_and_held(self):
        self.assertEqual(collision_clearance_phase_weight(6.99), 0.0)
        self.assertEqual(collision_clearance_phase_weight(7.0), 0.0)
        self.assertGreater(collision_clearance_phase_weight(7.5), 0.0)
        self.assertAlmostEqual(collision_clearance_phase_weight(8.0), 1.0)
        # The relief remains active through lift/hold so a solved grasp is not
        # returned to a known thumb--wrist overlap during the expert window.
        self.assertEqual(collision_clearance_phase_weight(16.0), 1.0)
        with self.assertRaises(ValueError):
            collision_clearance_phase_weight(7.0, onset_s=8.0, ramp_end_s=8.0)

    def test_spatial_prompt_condition_is_case_derived_and_bounded(self):
        base = [f"stage {i}" for i in range(5)]
        right = condition_prompts(base, 'right')
        left = condition_prompts(base, 'left')
        self.assertEqual(len(right), 5)
        self.assertNotEqual(right, left)
        self.assertTrue(all('right side' in value for value in right))
        self.assertEqual(descriptor('center'), 'The red cube is at the center of the tabletop.')
        with self.assertRaises(ValueError):
            condition_prompts(base[:4], 'right')

if __name__ == '__main__': unittest.main()
