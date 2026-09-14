import unittest

from control.joint_controller import apply_joint_controller, classify_state


class JointControllerSmokeTest(unittest.TestCase):
    """Smoke tests for the four-state early-stopping policy only."""

    def test_four_state_assignment(self):
        self.assertEqual(classify_state(0.8, 0.2, 0.5, 0.0), 0)
        self.assertEqual(classify_state(0.8, -0.2, 0.5, 0.0), 1)
        self.assertEqual(classify_state(0.2, 0.2, 0.5, 0.0), 2)
        self.assertEqual(classify_state(0.2, -0.2, 0.5, 0.0), 3)

    def test_threshold_boundary_is_high(self):
        self.assertEqual(classify_state(0.5, 0.0, 0.5, 0.0), 0)

    def test_state_zero_stops_immediately(self):
        decision = apply_joint_controller(
            predicted_quality=[0.2, 0.8, 0.9],
            predicted_utility=[0.2, 0.2, -0.2],
            quality_threshold=0.5,
            utility_threshold=0.0,
        )
        self.assertEqual(decision.states, (2, 0, 1))
        self.assertEqual(decision.output_index, 1)
        self.assertEqual(decision.decision_index, 1)
        self.assertEqual(decision.reason, "state_0_immediate_stop")

    def test_state_one_is_kept_as_rollback_candidate(self):
        decision = apply_joint_controller(
            predicted_quality=[0.2, 0.8, 0.9, 0.2],
            predicted_utility=[0.2, -0.1, -0.2, -0.1],
            quality_threshold=0.5,
            utility_threshold=0.0,
        )
        self.assertEqual(decision.states, (2, 1, 1, 3))
        self.assertEqual(decision.output_index, 1)
        self.assertEqual(decision.decision_index, 3)
        self.assertEqual(decision.reason, "state_3_rollback")

    def test_state_two_continues(self):
        decision = apply_joint_controller(
            predicted_quality=[0.1, 0.2, 0.3],
            predicted_utility=[0.1, 0.2, 0.3],
            quality_threshold=0.5,
            utility_threshold=0.0,
        )
        self.assertEqual(decision.states, (2, 2, 2))
        self.assertEqual(decision.output_index, 2)
        self.assertEqual(decision.decision_index, 2)
        self.assertEqual(decision.reason, "natural_stop")

    def test_state_three_without_prior_high_quality_does_not_rollback(self):
        decision = apply_joint_controller(
            predicted_quality=[0.2, 0.1, 0.3],
            predicted_utility=[0.2, -0.1, 0.2],
            quality_threshold=0.5,
            utility_threshold=0.0,
        )
        self.assertEqual(decision.states, (2, 3, 2))
        self.assertEqual(decision.output_index, 2)
        self.assertEqual(decision.decision_index, 2)
        self.assertEqual(decision.reason, "natural_stop")

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            apply_joint_controller([0.1], [0.1, 0.2], 0.5, 0.0)

    def test_rejects_empty_trajectory(self):
        with self.assertRaises(ValueError):
            apply_joint_controller([], [], 0.5, 0.0)


if __name__ == "__main__":
    unittest.main()
