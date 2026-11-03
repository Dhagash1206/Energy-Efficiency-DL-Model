import unittest

from Layerskip_energy.Observer_feedback import LearningState, validate_feedback


class ObserverFeedbackTests(unittest.TestCase):
    def test_rejected_branch_not_failure(self):
        rows = validate_feedback([10], [1, 2, 3], [1, 9, 3, 4], {2: [1, 9, 8, 5]}, 1, 2)
        self.assertEqual([r["valid"] for r in rows], [True, True, False, False])
        self.assertEqual([r["match"] for r in rows], [True, True, None, None])
        state = LearningState()
        state.update(rows)
        self.assertEqual(state.export()["2"]["observations"], 2)

    def test_eos_excludes_later_observations(self):
        rows = validate_feedback([10], [1, 2, 3], [1, 2, 3, 4], {2: [1, 2, 3, 4]}, 3, 1)
        self.assertEqual([r["valid"] for r in rows], [True, False, False, False])


if __name__ == "__main__":
    unittest.main()
