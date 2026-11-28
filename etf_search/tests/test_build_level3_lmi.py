import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from build_level3_lmi_4_9 import affine_chart, packed_lmi, rational_equalities


class AffineChartTest(unittest.TestCase):
    def setUp(self):
        self.model = {
            'coefficient_basis': ['1', 'sqrt(10)/5'],
            'moments': [[], [1], [2, 2]],
            'equalities': [{'0': [-2, 0], '1': [0, 5]}, {'0': [-4, 0], '1': [0, 10]}],
            'psd_blocks': [{'basis': [[], [1]], 'entries': [
                [0, 0, {'0': 1}], [1, 0, {'1': 1}], [1, 1, {'2': 2}],
            ]}],
        }

    def test_full_exact_chart_and_packed_sign(self):
        parity, equalities = rational_equalities(self.model)
        chart = affine_chart(equalities, [1, 0])
        self.assertEqual(chart['free_columns'], [2])
        self.assertEqual(chart['rows'], [[1, [[0, 1]]], [1, [[0, 1]]], [1, [[1, 1]]]])
        matrix, rhs, sizes = packed_lmi(self.model, parity, chart['rows'])
        self.assertEqual(sizes, [2])
        np.testing.assert_allclose(rhs - matrix @ np.array([3]), [1, np.sqrt(20)/5, 6])

    def test_incomplete_equalities_rejected(self):
        self.model['equalities'][1] = {'2': [1, 0]}
        _, equalities = rational_equalities(self.model)
        with self.assertRaises(ArithmeticError):
            affine_chart(equalities, [0, 1])

    def test_rejects_mixed_parity_row(self):
        self.model['equalities'][1] = {'0': [1, 0], '1': [1, 0]}
        with self.assertRaises(ValueError):
            rational_equalities(self.model)
