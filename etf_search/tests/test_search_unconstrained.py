"""Independent numerical checks for the dephased unrestricted search."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))
import search_unconstrained as search


class UnconstrainedSearchTests(unittest.TestCase):
    def test_phase_gauge_and_modulus_projection(self):
        angles = np.linspace(-2, 2, 28)
        gram = search.phase_gram(angles)
        self.assertEqual(gram.shape, (9, 9))
        np.testing.assert_array_equal(gram, gram.conj().T)
        np.testing.assert_array_equal(np.diag(gram), 1)
        np.testing.assert_array_equal(gram[0, 1:], search.MU)
        np.testing.assert_allclose(np.abs(gram[np.triu_indices(9, 1)]), np.sqrt(5/32))
        np.testing.assert_allclose(search.project_modulus(gram), gram, atol=1e-15)
        np.testing.assert_array_equal(search.project_modulus(np.zeros((9, 9), complex)),
                                      search.phase_gram(np.zeros(28)))

    def test_spectrum_projection_selects_four_largest_algebraic_eigenvalues(self):
        source = np.diag([-11, -4, -2, 0, 1, 2, 3, 4, 5])
        target = np.diag([0] * 5 + [9/4] * 4)
        projected = search.project_spectrum(np.stack([source, source]))
        np.testing.assert_allclose(projected, np.stack([target, target]), atol=1e-15)
        np.testing.assert_allclose(search.project_spectrum(projected), projected, atol=1e-15)

    def test_analytic_gradient_matches_central_differences(self):
        angles = np.random.default_rng(9).uniform(-np.pi, np.pi, 28)
        gram = search.phase_gram(angles)
        loss, gradient = search.phase_objective(angles)
        self.assertAlmostEqual(loss, 0.5 * np.linalg.norm(gram @ gram - (9/4) * gram, 'fro')**2)
        difference = []
        for axis in 1e-6 * np.eye(28):
            difference.append((search.phase_objective(angles + axis)[0] -
                               search.phase_objective(angles - axis)[0]) / 2e-6)
        np.testing.assert_allclose(gradient, difference, rtol=2e-6, atol=1e-8)

    def test_batched_drs_matches_three_dense_reference_steps(self):
        seeds = [7, 41]
        result = search.drs_batch(seeds, steps=3, relaxation=1.0)
        for index, seed in enumerate(seeds):
            angles = np.random.default_rng(seed).uniform(-np.pi, np.pi, 28)
            state = shadow = search.phase_gram(angles)
            scores = [np.linalg.norm(shadow @ shadow - (9/4) * shadow, 'fro')]
            for _ in range(3):
                _, vectors = np.linalg.eigh(2 * shadow - state)
                top = vectors[:, -4:]
                state = state + (9/4) * top @ top.conj().T - shadow
                shadow = search.project_modulus(state)
                scores.append(np.linalg.norm(shadow @ shadow - (9/4) * shadow, 'fro'))
            np.testing.assert_allclose(result['last_grams'][index], shadow, atol=1e-12)
            self.assertEqual(result['best_steps'][index], int(np.argmin(scores)))
            gram = result['best_grams'][index]
            self.assertAlmostEqual(np.linalg.norm(gram @ gram - (9/4) * gram, 'fro'), min(scores))

    def test_cli_saves_all_optimizer_endpoints_and_independently_checks_them(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            subprocess.run([sys.executable, '-B', str(SOURCE / 'search_unconstrained.py'),
                            '--seeds', '3', '--seed-start', '5', '--steps', '7',
                            '--workers', '2', '--batch-size', '2', '--refine-maxiter', '40',
                            '--output-dir', str(out)], capture_output=True, text=True, timeout=60, check=True)
            result = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
            matrices = np.load(out / 'all_final_grams.npy', allow_pickle=False)
            self.assertEqual(matrices.shape, (3, 9, 9))
            self.assertEqual([row['seed'] for row in result['runs']], [5, 6, 7])
            for row, gram in zip(result['runs'], matrices, strict=True):
                self.assertAlmostEqual(row['final']['tightness_frobenius'],
                                       np.linalg.norm(gram @ gram - (9/4) * gram, 'fro'))
                self.assertAlmostEqual(row['independent_verification']['metrics']['tightness_frobenius'],
                                       row['final']['tightness_frobenius'])
            best = result['top10'][0]['matrix_index']
            np.testing.assert_array_equal(np.load(out / 'best_gram.npy', allow_pickle=False), matrices[best])
            self.assertEqual(result['double_gate_hits'], [])


if __name__ == '__main__':
    unittest.main()
