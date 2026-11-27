"""Checks specific to the antiunitary projection and phase coordinates."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import check_grad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from search_antiunitary import AntiunitaryModel, MU, SCALE


class AntiunitaryTests(unittest.TestCase):
    def test_dimensions_signs_and_projection(self):
        rng = np.random.default_rng(29)
        for n_fixed, sign, dimension in [(1, 1, 16), (3, 1, 15), (3, -1, 15)]:
            model = AntiunitaryModel(n_fixed, sign)
            self.assertEqual(model.dimension, dimension)
            gram = model.from_angles(rng.uniform(-np.pi, np.pi, dimension))
            permutation = model.permutation
            np.testing.assert_array_equal(gram[np.ix_(permutation, permutation)], gram.conj())
            np.testing.assert_allclose(model.project_modulus(gram), gram, atol=1e-15, rtol=0)
            np.testing.assert_allclose(gram[0, 1:n_fixed], MU)
            np.testing.assert_allclose(gram[0, n_fixed:], MU)
            if n_fixed == 3:
                self.assertEqual(gram[1, 2], sign * MU)
            rng_input = rng.normal(size=(9, 9)) + 1j * rng.normal(size=(9, 9))
            projected = model.project_spectrum(rng_input)
            np.testing.assert_allclose(projected[np.ix_(permutation, permutation)], projected.conj(), atol=2e-15)
            np.testing.assert_allclose(np.linalg.eigvalsh(projected), [0] * 5 + [SCALE] * 4, atol=2e-15)
            np.testing.assert_allclose(model.project_spectrum(projected), projected, atol=3e-15)

    def test_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(70)
        for n_fixed, sign in [(1, 1), (3, -1)]:
            model = AntiunitaryModel(n_fixed, sign)
            phase = rng.uniform(-np.pi, np.pi, model.dimension)
            error = check_grad(lambda x: model.objective(x)[0], lambda x: model.objective(x)[1], phase)
            self.assertLess(error, 2e-6)


if __name__ == "__main__":
    unittest.main()
