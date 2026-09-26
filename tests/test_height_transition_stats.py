import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "height_transition_stats", Path(__file__).parents[1] / "scripts/rsl_rl/height_transition_stats.py"
)
stats = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stats)
analysis_spec = importlib.util.spec_from_file_location(
    "analyze_height_transitions", Path(__file__).parents[1] / "scripts/rsl_rl/analyze_height_transitions.py"
)
analysis = importlib.util.module_from_spec(analysis_spec)
analysis_spec.loader.exec_module(analysis)


class TransitionStatsTest(unittest.TestCase):
    def summarize(self, heights):
        record = {"target_height": 0.6, "samples": [((i + 1) * 0.02, h) for i, h in enumerate(heights)]}
        return stats.summarize_segment(record, 0.02, 2.0, 0.08)

    def test_short_command_is_unobserved_not_failure(self):
        result = self.summarize([0.1] * 100)
        self.assertEqual(result["settled_s"], 0)
        self.assertIsNone(result["mean_abs_error"])

    def test_transient_excluded_from_settled_error(self):
        result = self.summarize([0.1] * 100 + [0.6] * 50)
        self.assertAlmostEqual(result["mean_abs_error"], 0)
        self.assertTrue(result["sustained_success"])

    def test_success_must_be_contiguous(self):
        result = self.summarize([0.6] * 100 + [0.6, 0.3] * 50)
        self.assertFalse(result["sustained_success"])
        self.assertAlmostEqual(result["within_threshold_fraction"], 0.5)

    def test_contact_excludes_other_bodies(self):
        self.assertEqual(stats.contact_label([[0, 0, 20], [0, 0, 30], [0, 0, 0]], [0, 1]), "two_feet")
        self.assertEqual(stats.contact_label([[0, 0, 20], [0, 0, 30], [0, 0, 6]], [0, 1]), "nonfoot_contact")
        self.assertEqual(stats.contact_label([[0, 0, 0], [0, 0, 30]], [0, 1]), "one_foot")

    def test_transient_success_is_not_whole_segment_success(self):
        result = self.summarize([0.6] * 150 + [0.2] * 50)
        result["cluster"] = "seed:1"
        aggregate = analysis.summary([result])
        self.assertEqual(aggregate["sustained_success"], 1)
        self.assertEqual(aggregate["stable_segment_success"], 0)

    def test_short_segments_not_in_success_denominator(self):
        short = self.summarize([0.1] * 100)
        good = self.summarize([0.6] * 150)
        short["cluster"] = good["cluster"] = "seed:1"
        aggregate = analysis.summary([short, good])
        self.assertEqual(aggregate["segments"], 2)
        self.assertEqual(aggregate["success_eligible_segments"], 1)
        self.assertEqual(aggregate["stable_segment_success"], 1)


if __name__ == "__main__":
    unittest.main()
