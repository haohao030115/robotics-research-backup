import numpy as np

from qwen_ardy.geometry import robot_to_world, safe_target, world_to_robot
from qwen_ardy.models import EpisodeSummary
from qwen_ardy.qwen_vlm import heuristic_decision
from qwen_ardy.hand_controller import ContactDrivenHand
from qwen_ardy.stages import LiveState, Stage, gate


def test_coordinate_roundtrip():
    root = np.array([.2, -.1, .8])
    quat = np.array([1., 0, 0, 0])
    point = np.array([.5, -.2, .75])
    assert np.allclose(robot_to_world(world_to_robot(point, root, quat), root, quat), point)


def test_target_clipping():
    target = safe_target(np.array([.475, -.177, .735]), np.array([.2, -.2, .2]))
    assert np.allclose(target, [.555, -.257, .815])


def test_fallback_follows_error():
    summary = EpisodeSummary(0, False, [.475,-.177,.735], [.475,-.177,.735],
                             [.44,-.18,.74], [.035,.003,-.005], .04, 0, 0, 0, .75)
    decision = heuristic_decision(summary)
    assert decision.constraint_delta_world_m == [.035, .003, -.005]


def test_grasp_gate_is_contact_driven():
    state = LiveState(.01, .005, .78, 2, True, True, 0, .3)
    assert gate(Stage.GRASP, state).ready
    assert not gate(Stage.GRASP, LiveState(.01, .005, .78, 2, True, False, 0, 2)).ready


def test_hand_stops_squeezing_on_opposing_contact():
    hand = ContactDrivenHand()
    before = hand.update(.5, True, False, False).copy()
    amount = hand.amount
    after = hand.update(.5, True, True, True)
    assert hand.amount == amount
    assert np.allclose(before, after)
