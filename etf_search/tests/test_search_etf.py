"""Tests for the free-C3 search, using dense Gram checks as a reference."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE))
import search_etf as search


class C3ModelTests(unittest.TestCase):
    def setUp(self):
        self.phases = np.random.default_rng(118).uniform(-np.pi, np.pi, 10)
        self.coeff = search.phase_coefficients(self.phases)
        self.gram = search.gram_from_coefficients(self.coeff)

    def test_coefficients_phase_round_trip_and_structure(self):
        self.assertEqual(self.coeff.shape, (3, 3, 3))
        self.assertEqual(self.gram.shape, (9, 9))
        self.assertEqual(len(search.PHASE_POSITIONS), 10)
        np.testing.assert_allclose(search.coefficient_phases(self.coeff), self.phases, atol=1e-14)
        np.testing.assert_array_equal(self.gram, self.gram.conj().T)
        np.testing.assert_array_equal(np.diag(self.gram), 1)
        np.testing.assert_allclose(np.abs(self.gram[~np.eye(9, dtype=bool)]), np.sqrt(5/32), atol=1e-14)
        self.assertEqual(self.gram[0, 3], search.MU)
        self.assertEqual(self.gram[0, 6], search.MU)
        shift = [(i // 3) * 3 + (i + 1) % 3 for i in range(9)]
        np.testing.assert_array_equal(self.gram[np.ix_(shift, shift)], self.gram)

    def test_fourier_sign_and_sector_order(self):
        omega = np.exp(2j * np.pi / 3)
        unitary = np.array([
            [int(i // 3 == j % 3) * omega ** ((i % 3) * (j // 3)) / np.sqrt(3) for j in range(9)]
            for i in range(9)
        ])
        ordered = unitary.conj().T @ self.gram @ unitary
        blocks = search.to_sectors(self.coeff)
        np.testing.assert_allclose(blocks[1], sum(omega ** t * self.coeff[:, :, t] for t in range(3)), atol=1e-14)
        for k in range(3):
            np.testing.assert_allclose(blocks[k], ordered[3*k:3*k+3, 3*k:3*k+3], atol=1e-14)
        np.testing.assert_allclose(search.from_sectors(blocks), self.coeff, atol=1e-14)

    def test_modulus_projection_has_deterministic_zero_phase_and_is_idempotent(self):
        zero = np.zeros((3, 3, 3), dtype=complex)
        expected = search.phase_coefficients(np.zeros(10))
        np.testing.assert_array_equal(search.project_modulus(zero), expected)
        np.testing.assert_allclose(search.project_modulus(self.coeff), self.coeff, atol=1e-14)

    def test_spectrum_projection_selects_largest_algebraic_eigenvalues(self):
        sectors = np.array([np.diag([-10, 1, 2]), np.diag([-7, -4, -1]), np.diag([3, 2, 1])])
        dat = search.from_sectors(sectors)
        got = search.project_spectrum(dat)
        expect = search.LAMBDA * np.array([np.diag([0, 1, 1]), np.diag([0, 0, 1]), np.diag([1, 0, 0])])
        np.testing.assert_allclose(search.to_sectors(got), expect, atol=1e-14)
        np.testing.assert_allclose(search.project_spectrum(got), got, atol=1e-14)
        gram = search.gram_from_coefficients(got)
        np.testing.assert_allclose(gram @ gram, search.LAMBDA * gram, atol=1e-14)

    def test_phase_gradient_matches_dense_central_differences(self):
        loss, gradient = search.phase_objective(self.phases)
        residual = self.gram @ self.gram - (9/4) * self.gram
        self.assertAlmostEqual(loss, 0.5 * np.linalg.norm(residual, "fro") ** 2, places=12)
        step = 1e-6
        numeric = np.array([
            (search.phase_objective(self.phases + step * e)[0] - search.phase_objective(self.phases - step * e)[0])
            / (2 * step) for e in np.eye(10)
        ])
        np.testing.assert_allclose(gradient, numeric, rtol=1e-6, atol=1e-7)

    def test_gate_is_strict_and_uses_both_residuals(self):
        good = {"tightness_frobenius": 9e-13, "welch_max_error": 9e-13,
                "hermitian_max_error": 0.0, "diagonal_max_error": 0.0}
        self.assertTrue(search.passes_numeric(good))
        for key in ("tightness_frobenius", "welch_max_error"):
            with self.subTest(key=key):
                self.assertFalse(search.passes_numeric({**good, key: 1e-12}))

    def test_seed_runs_are_reproducible_and_keep_best(self):
        for method in ("drs", "ap"):
            with self.subTest(method=method):
                options = dict(steps=8, relaxation=1.0, refine_maxiter=30)
                first, gram = search.run_seed(method, 23, **options)
                again, repeated_gram = search.run_seed(method, 23, **options)
                self.assertEqual(first, again)
                np.testing.assert_array_equal(gram, repeated_gram)
                initial = first["initial"]["tightness_frobenius"]
                projected = first["best_projected"]["tightness_frobenius"]
                best = first["best"]["tightness_frobenius"]
                self.assertLessEqual(projected, initial)
                self.assertLessEqual(best, projected)
                self.assertAlmostEqual(best, np.linalg.norm(gram @ gram - (9/4) * gram, "fro"))
                self.assertLess(first["best"]["welch_max_error"], 1e-12)

    def test_cli_saves_unmodified_gram_and_all_seed_records(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            process = subprocess.run([
                sys.executable, "-B", str(SOURCE / "search_etf.py"),
                "--seeds", "2", "--seed-start", "23", "--steps", "8",
                "--refine-maxiter", "30", "--output-dir", str(out),
            ], text=True, capture_output=True, timeout=45, check=True)
            self.assertIn("best_gram.npy", process.stdout)
            report = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            gram = np.load(out / "best_gram.npy", allow_pickle=False)
            self.assertEqual(report["model"]["sector_ranks"], [2, 1, 1])
            self.assertEqual([(r["seed"], r["method"]) for r in report["runs"]],
                             [(23, "drs"), (23, "ap"), (24, "drs"), (24, "ap")])
            best = report["best"]["tightness_frobenius"]
            self.assertEqual(best, min(row["best"]["tightness_frobenius"] for row in report["runs"]))
            self.assertAlmostEqual(best, np.linalg.norm(gram @ gram - (9/4) * gram, "fro"))
            self.assertEqual(report["symbolic"]["status"], "numeric_gate_not_met")
            self.assertFalse((out / "candidate_exact.json").exists())


if __name__ == "__main__":
    unittest.main()
