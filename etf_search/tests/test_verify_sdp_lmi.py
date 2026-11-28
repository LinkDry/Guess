"""Proof-boundary and numeric diagnostics regressions for the reduced LMI."""

from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.sparse import csc_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from verify_sdp_lmi import (  # noqa: E402
    AffineChart, assemble_affine_objective, bounded_upper, certify_matrix,
    conic_diagnostics, exact_dual_report, lift_numeric_free, moment_bounds, propose_box_weights,
    read_primal_moments, unpack_svec,
)


def chart_document():
    # m_0 = 1, m_1 = beta*t, m_2 = 1/2-t.
    return {"parity": [0, 1, 0], "pivot_columns": [0, 2], "free_columns": [1],
            "selected_rows": [0, 1],
            "rows": [[1, [[0, 1]]], [1, [[1, 1]]], [2, [[0, 1], [1, -2]]]]}


def chart_equalities():
    return [{0: (1, 0)}, {0: (-1, 0), 1: (0, 5), 2: (2, 0)}]


class ReducedLmiTests(unittest.TestCase):
    def test_upper_and_lower_column_packing(self):
        s = np.sqrt(2)
        expected = [[1, 2, 4], [2, 3, 5], [4, 5, 6]]
        np.testing.assert_allclose(unpack_svec([1, 2*s, 4*s, 3, 5*s, 6], 3, "lower"), expected)
        np.testing.assert_allclose(unpack_svec([1, 2*s, 3, 4*s, 5*s, 6], 3, "upper"), expected)

    def test_ray_diagnostics_and_numerical_candidate_are_distinct(self):
        result = conic_diagnostics(csc_matrix([[0.0], [1.0]]), np.array([-1., 0.]),
                                   np.zeros(1), [1, 1], {"x": [1.], "z": [1., 0.]}, "lower")
        self.assertEqual(result["primal"]["min_eigenvalues"], [-1., -1.])
        self.assertEqual(result["dual"]["ray_b_dot_z"], -1.)
        self.assertEqual(result["dual"]["ray_stationarity_inf"], 0.)
        self.assertFalse(result["exact_proof"])

    def test_chart_proves_rank_and_substitution(self):
        chart = AffineChart(chart_document(), [0, 1, 0])
        self.assertEqual(chart.verify_coverage(chart_equalities()), 2)
        self.assertEqual(chart.lift([Fraction(1, 4)]), [(1, 0), (0, Fraction(1, 4)), (Fraction(1, 4), 0)])
        bad = deepcopy(chart_document())
        bad["rows"][2] = [1, []]
        with self.assertRaisesRegex(ValueError, "substitution"):
            AffineChart(bad, [0, 1, 0]).verify_coverage(chart_equalities())

    def test_chart_cannot_restrict_free_coordinates(self):
        doc = chart_document()
        doc["parity"].append(0)
        doc["pivot_columns"].append(3)
        doc["selected_rows"].append(2)
        doc["rows"].append([1, []])
        chart = AffineChart(doc, [0, 1, 0, 0])
        first, second = chart_equalities()
        with self.assertRaisesRegex(ValueError, "rank"):
            chart.verify_coverage([first, second, second])
        bad = chart_document()
        bad["rows"][1] = [2, [[1, 1]]]
        with self.assertRaisesRegex(ValueError, "free coordinate"):
            AffineChart(bad, [0, 1, 0])

    def test_restricted_primal_coordinates_lift_into_the_full_chart(self):
        self.assertEqual(lift_numeric_free([0.25, -0.75], [3, 0], 5, 100),
                         [Fraction(-3, 4), 0, 0, Fraction(1, 4), 0])

    def test_psd_entry_trace_weights_off_diagonal_twice(self):
        entries = [[(0, 0, {0: -2}), (1, 0, {1: 3}), (1, 1, {2: 4})]]
        self.assertEqual(assemble_affine_objective(entries, [[[1], [2]]], 3), [-2, 12, 16])

    def test_nonzero_stationarity_certifies_with_bounded_moments(self):
        chart = AffineChart(chart_document(), [0, 1, 0])
        bounds = moment_bounds([(), (0,), (1, 1)])
        near_ray = chart.linear_form([-2, Fraction(1, 4), 0])
        self.assertLess(bounded_upper(chart, near_ray, bounds), 0)
        self.assertGreaterEqual(bounded_upper(chart, chart.linear_form([0, 1, 0]), bounds), 1)
        self.assertEqual(bounds[2], (0, 1))

    def test_lifted_box_bounds_can_rescue_a_dual(self):
        chart = AffineChart(chart_document(), [0, 1, 0])
        bounds = moment_bounds([(), (0,), (1, 1)])
        # m_2 >= 0 kills the otherwise positive free-box residual.
        objective = chart.linear_form([Fraction(-1, 4), 0, -1])
        self.assertGreater(bounded_upper(chart, objective, bounds), 0)
        weights = propose_box_weights(chart, objective, bounds, 10**6)
        self.assertTrue(weights)
        self.assertEqual(bounded_upper(chart, objective, bounds, weights), Fraction(-1, 4))
        with self.assertRaisesRegex(ValueError, "bound multiplier"):
            bounded_upper(chart, objective, bounds, {3: 1})

    def test_exact_dual_factors_and_weights_replay(self):
        chart = AffineChart(chart_document(), [0, 1, 0])
        chart.verify_coverage(chart_equalities())
        bounds = moment_bounds([(), (0,), (1, 1)])
        entries = [[(0, 0, {0: -1, 2: -4})]]
        result, weights = exact_dual_report(chart, entries, bounds, [[[1]]], 1)
        self.assertFalse(result["passed"])
        self.assertFalse(weights)
        result, weights = exact_dual_report(chart, entries, bounds, [[[1]]], 10**6, tighten=True)
        self.assertTrue(result["passed"])
        replayed, _ = exact_dual_report(chart, entries, bounds, [[[1]]], 10**6, weights=weights)
        self.assertEqual(result, replayed)
        with self.assertRaisesRegex(ValueError, "integer Gram factor"):
            exact_dual_report(chart, entries, bounds, [[[True]]], 1)

    def test_exact_primal_archive_requires_basis_and_moment_count(self):
        candidate = {"schema": "anchored-exact-primal-v1", "coefficient_basis": ["1", "sqrt(10)/5"],
                     "moments": [1, ["1/4", "2/3"]]}
        self.assertEqual(read_primal_moments(candidate, 2), [(1, 0), (Fraction(1, 4), Fraction(2, 3))])
        with self.assertRaisesRegex(ValueError, "moment count"):
            read_primal_moments(candidate, 1)
        with self.assertRaisesRegex(ValueError, "coefficient basis"):
            read_primal_moments({**candidate, "coefficient_basis": ["1", "sqrt(10)"]}, 2)

    def test_exact_psd_requires_symmetric_inputs(self):
        report = certify_matrix([[10, 1], [0, 10]], [[0, 0], [0, 0]], 1)
        self.assertFalse(report["holds"])
        self.assertEqual(report["reason"], "asymmetric")

    def test_exact_psd_and_negative_witness(self):
        self.assertTrue(certify_matrix([[1, 1], [1, 1]], [[0, 0], [0, 0]], 1)["holds"])
        report = certify_matrix([[1, 0], [0, 1]], [[0, 2], [2, 0]], 1)
        self.assertFalse(report["holds"])
        self.assertLess(Fraction(report["quadratic_form_upper"]), 0)
        self.assertTrue(report["integer_test_vector"])


if __name__ == "__main__":
    unittest.main()
