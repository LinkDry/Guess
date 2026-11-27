"""Exact checks for the permutation-invariant triangle relaxation."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bargmann_4_9 import build_witness


class BargmannTests(unittest.TestCase):
    def test_invariant_moments_and_projection_ranks(self):
        witness = build_witness()
        self.assertEqual(witness["centered_covariance_ranks"], {"real": 48, "imaginary": 56})
        self.assertEqual(witness["four_cycle_mean"], "-11/70")
        self.assertEqual(witness["trace_signature_fourth_power"], "3024/5")
        self.assertTrue(all(witness["exact_checks"].values()))
