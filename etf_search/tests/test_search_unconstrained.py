"""Independent numerical checks for the dephased unrestricted search."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

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


class FourTenRegressionTests(unittest.TestCase):
    def test_36_phase_objective_and_rank_four_projection_use_new_parameters(self):
        model = search.GramModel(4, 10)
        angles = np.random.default_rng(61).uniform(-np.pi, np.pi, 36)
        gram = search.phase_gram(angles, model=model)
        self.assertEqual(gram.shape, (10, 10))
        np.testing.assert_array_equal(gram[0, 1:], np.sqrt(1/6))
        np.testing.assert_allclose(gram, search.project_modulus(gram, model=model), atol=1e-15)
        loss, gradient = search.phase_objective(angles, model=model)
        self.assertAlmostEqual(loss, 0.5 * np.linalg.norm(gram @ gram - 2.5 * gram, 'fro') ** 2)
        direction = np.random.default_rng(8).normal(size=36)
        step = 1e-6 * direction
        finite_difference = (search.phase_objective(angles + step, model=model)[0]
                             - search.phase_objective(angles - step, model=model)[0]) / 2e-6
        self.assertAlmostEqual(gradient @ direction, finite_difference, places=7)
        projected = search.project_spectrum(gram, model=model)
        np.testing.assert_allclose(np.linalg.eigvalsh(projected), [0] * 6 + [2.5] * 4, atol=1e-14)

    def test_four_ten_cli_and_saved_verifier_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            subprocess.run([sys.executable, '-B', str(SOURCE / 'search_unconstrained.py'),
                            '--d', '4', '--n', '10', '--seeds', '2', '--steps', '4',
                            '--workers', '1', '--output-dir', str(out)],
                           capture_output=True, text=True, timeout=60, check=True)
            result = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
            matrices = np.load(out / 'all_final_grams.npy', allow_pickle=False)
            self.assertEqual(matrices.shape, (2, 10, 10))
            self.assertEqual(result['model']['free_real_phases'], 36)
            self.assertEqual(result['model']['mu'], np.sqrt(1/6))
            self.assertEqual(result['model']['lambda'], 2.5)
            for row, gram in zip(result['runs'], matrices, strict=True):
                verifier = row['independent_verification']
                self.assertEqual(verifier['mu'], np.sqrt(1/6))
                expected = np.linalg.norm(gram @ gram - 2.5 * gram, 'fro')
                self.assertAlmostEqual(verifier['metrics']['tightness_frobenius'], expected)

    def test_tightness_crossing_invokes_independent_verifier_before_recovery(self):
        model = search.GramModel(4, 10)
        spectral_gram = np.diag([2.5] * 4 + [0.] * 6).astype(complex)
        tightness_hits = []
        with patch.object(search, 'project_modulus', return_value=spectral_gram[None, ...]):
            search.drs_batch([43], steps=1, relaxation=1.0, model=model,
                             on_hit=lambda seed, stage, gram: tightness_hits.append((seed, stage, gram)))
        self.assertEqual([(s, p) for s, p, _ in tightness_hits], [(43, 'drs_step_1')])
        with tempfile.TemporaryDirectory() as directory, patch.object(search, 'recover_exact') as recovery:
            report = search.record_hit(*tightness_hits[0], Path(directory), model=model)
            recovery.assert_not_called()
            self.assertFalse(report['independent_pass'])
            self.assertEqual(report['recovery_status'], 'independent_gate_failed')

    def test_high_precision_recovery_uses_the_selected_parameters(self):
        # Known (4,5) simplex with first row rephased to +mu.
        model = search.GramModel(4, 5)
        gram = np.full((5, 5), -0.25, dtype=complex)
        np.fill_diagonal(gram, 1)
        gram[0, 1:] = gram[1:, 0] = 0.25
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            report = search.record_hit(7, 'test_simplex', gram, path, model=model)
            self.assertTrue(report['independent_pass'])
            self.assertEqual(report['recovery_status'], 'exact_verified')
            exact = json.loads((path / 'double_gate_hits/seed_7_test_simplex/candidate_exact.json').read_text())
            self.assertEqual(exact['gram'][0][1], 'sqrt(1/16)')


if __name__ == '__main__':
    unittest.main()
