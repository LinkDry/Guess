"""Proof boundary regressions for the independent support-five verifier."""

from fractions import Fraction as Q
from pathlib import Path
import sys
import unittest

from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from verify_support5_rank import (audit_common_kernel, audit_image_rank, bounded_exact_zero, chart_identities,
                                  exact_dual, exact_primal, parse_chart, primitive_parity_row)


class Support5RankTests(unittest.TestCase):
    def test_primitive_scaling_and_mixed_beta_parity(self):
        self.assertEqual(primitive_parity_row({1: (0, -15), 2: (10, 0)}, [0, 1, 0]),
                         {1: 3, 2: -5})
        with self.assertRaisesRegex(ValueError, 'mixed beta parity'):
            primitive_parity_row({1: (1, 0), 2: (1, 0)}, [0, 1, 0])

    def test_bounded_crt_checks_integer_cancellation_and_extra_row(self):
        factor = 10**20
        with self.assertRaisesRegex(ValueError, 'row 1'):
            bounded_exact_zero(csr_matrix([[3, -2, 0], [0, 0, 1]]), [[2*factor, 3*factor, 1_000_003]])
        bounded_exact_zero(csr_matrix([[3, -2, 0], [0, 0, 1]]), [[2*factor, 3*factor, 0]])
        bounded_exact_zero(csr_matrix([[3, -2]]), [[Q(2, 10**20), Q(3, 10**20)]])

    def test_chart_cannot_change_free_coordinate(self):
        bad = {'parity': [0, 0], 'free_columns': [1], 'rows': [[1, [[0, 1]]], [1, []]]}
        with self.assertRaisesRegex(ValueError, 'free coordinate'):
            parse_chart(bad, [0, 0])

    def test_additional_minor_and_unselected_cross_upper_bound(self):
        old = ([{0: 1}, {1: 1}], [1])
        old_matrix = csr_matrix((0, 2), dtype=int)
        doc = {'eliminated_moment_coordinates': [1], 'selected_cross_rows': [0], 'additional_rank': 1}
        chart = ([{0: 1}, {1: Q(3, 2)}, {1: 1}], [2], doc)
        cross = csr_matrix([[0, 2, -3], [0, 4, -6]])
        self.assertEqual(chart_identities(chart, old, cross, old_matrix)['exact_rank_including_normalization'], 2)
        with self.assertRaisesRegex(ValueError, 'row 1'):
            chart_identities(chart, old, csr_matrix([[0, 2, -3], [0, 0, 65537]]), old_matrix)

    def test_psd_image_requires_exact_relation_on_every_block(self):
        certificate = {'rank': 1, 'retained_direction_columns': [0], 'witness_packed_rows': [[0, 0]],
                       'discarded_as_retained_combinations': [[1, [[0, '3/2']]]]}
        rows = [{0: 1}, {1: 1}, {2: 1}]
        blocks = [csr_matrix([[0, 2, 3]]), csr_matrix([[0, 4, 6]])]
        self.assertEqual(audit_image_rank(certificate, blocks, rows, 2)['exact_rank'], 1)
        blocks[1] = csr_matrix([[0, 4, 6 + 65537]])
        with self.assertRaisesRegex(ValueError, 'exact sparse identity'):
            audit_image_rank(certificate, blocks, rows, 2)

    def test_common_kernel_checks_every_affine_coefficient_and_converts(self):
        flags = [[0, 1]]
        matrix = csr_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        rows = [{0: 1}, {0: 1}, {0: Q(2, 5)}]
        cert = {'schema': 'support5-common-kernel-v1',
                'blocks': [{'size': 2, 'basis_parity': [0, 1], 'pivot_indices': [0],
                            'vectors': [[[0, '1'], [1, '-1']]]}]}
        kernels, counts = audit_common_kernel(cert, [matrix], flags, rows, 2)
        self.assertEqual(counts, [1])
        moments = [(1, 0), (0, 1), (Q(2, 5), 0)]
        empty = csr_matrix((0, 3), dtype=int)
        self.assertTrue(exact_primal(moments, [matrix], [2], [0, 1, 0], empty, empty, kernels)['passed'])
        rows[2][1] = Q(1, 65537)
        with self.assertRaisesRegex(ValueError, 'exact sparse identity'):
            audit_common_kernel(cert, [matrix], flags, rows, 2)

    def test_singular_primal_verifies_field_kernel_and_principal_complement(self):
        # M = [[1, beta], [beta, 2/5]], with q = [-beta, 1].
        mapping = csr_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        empty = csr_matrix((0, 3), dtype=int)
        moments = [(1, 0), (0, 1), (Q(2, 5), 0)]
        record = {'pivot_indices': [1], 'directions': [{'0': [0, -1], '1': 1}]}
        report = exact_primal(moments, [mapping], [2], [0, 1, 0], empty, empty, [record])
        self.assertTrue(report['passed'])
        self.assertEqual(report['blocks'][0]['verified_kernel_dimension'], 1)
        self.assertEqual(report['blocks'][0]['principal_order'], 1)
        record['directions'][0]['0'] = [0, 1]
        with self.assertRaisesRegex(ValueError, 'kernel annihilation fails'):
            exact_primal(moments, [mapping], [2], [0, 1, 0], empty, empty, [record])

    def test_exact_original_primal_and_dual_on_toy_cones(self):
        matrix = csr_matrix([[1]])
        empty = csr_matrix((0, 1), dtype=int)
        self.assertTrue(exact_primal([(1, 0)], [matrix], [1], [0], empty, empty)['passed'])
        self.assertFalse(exact_primal([(1, 0)], [-matrix], [1], [0], empty, empty)['passed'])
        factor = {'gram_factor_denominator': 1, 'factors': [[[1]]]}
        chart = ([{0: Q(1)}], [], {})
        self.assertTrue(exact_dual(factor, chart, [-matrix], [1], [()], [0], False)['etf_nonexistence_verified'])
        self.assertFalse(exact_dual(factor, chart, [matrix], [1], [()], [0], False)['passed'])
        self.assertFalse(exact_dual(factor, chart, [-matrix], [1], [()], [0], True)['etf_nonexistence_verified'])


if __name__ == '__main__':
    unittest.main()
