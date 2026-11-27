"""Exercise the independent verifier through its public CLI."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "src" / "verify_etf.py"
CHECKS = {"hermitian", "unit_diagonal", "welch_modulus", "tightness"}


def harmonic_gram():
    """The cyclic (7, 3, 1) difference set gives a nonreal ETF Gram matrix."""
    samples = np.arange(7)
    frequencies = np.array([0, 1, 3])
    frame = np.exp(2j * np.pi * np.outer(frequencies, samples) / 7) / np.sqrt(3)
    return frame.conj().T @ frame


def symbolic_harmonic_gram():
    entries = [
        [
            1 if i == j else f"(1+zeta(7)**{(j-i)%7}+zeta(7)**{(3*(j-i))%7})/3"
            for j in range(7)
        ]
        for i in range(7)
    ]
    return {"format": "etf-gram-v1", "gram": entries}


class VerificationCLI(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def run_cli(self, path, d, n, *flags):
        result = subprocess.run(
            [sys.executable, "-B", str(SCRIPT), str(path), "--d", str(d), "--n", str(n), *flags],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.stderr, "")

        def reject_constant(token):
            self.fail(f"Nonstandard JSON number: {token}")

        return result.returncode, json.loads(result.stdout, parse_constant=reject_constant)

    def numeric(self, gram, d=3, n=7):
        path = self.directory / "gram.npy"
        np.save(path, gram)
        return self.run_cli(path, d, n)

    def symbolic(self, document, d=3, n=7):
        path = self.directory / "gram.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return self.run_cli(path, d, n, "--symbolic")

    def test_nonreal_numeric_etf(self):
        gram = harmonic_gram()
        self.assertGreater(np.max(np.abs(gram.imag)), 0.1)
        code, report = self.numeric(gram)
        self.assertEqual(code, 0)
        self.assertTrue(report["passed"])
        self.assertEqual(report["mode"], "numeric")
        self.assertEqual(report["checks"], dict.fromkeys(CHECKS, True))
        self.assertLess(report["metrics"]["tightness_frobenius"], 1e-12)
        self.assertLess(report["metrics"]["welch_max_error"], 1e-12)

    def test_phase_corruption_preserves_moduli_but_fails_tightness(self):
        gram = harmonic_gram()
        gram[0, 1] *= 1j
        gram[1, 0] = gram[0, 1].conjugate()
        code, report = self.numeric(gram)
        self.assertEqual(code, 1)
        self.assertFalse(report["passed"])
        self.assertEqual(report["checks"], dict.fromkeys(CHECKS, True) | {"tightness": False})

    def test_lower_triangle_is_checked_on_supplied_matrix(self):
        gram = harmonic_gram()
        gram[1, 0] = 0
        code, report = self.numeric(gram)
        self.assertEqual(code, 1)
        self.assertFalse(report["checks"]["hermitian"])
        self.assertFalse(report["checks"]["welch_modulus"])
        self.assertTrue(report["checks"]["unit_diagonal"])

    def test_wrong_welch_parameter_fails(self):
        code, report = self.numeric(harmonic_gram(), d=4)
        self.assertEqual(code, 1)
        self.assertFalse(report["checks"]["welch_modulus"])
        self.assertFalse(report["checks"]["tightness"])

    def test_invalid_array_inputs(self):
        cases = (
            (np.eye(6), 3, 7),
            (np.eye(7)[:, :, None], 3, 7),
            (np.eye(7, dtype=object), 3, 7),
            (np.full((7, 7), np.nan), 3, 7),
            (np.full((7, 7), -np.inf), 3, 7),
            (np.eye(7), 7, 7),
        )
        for gram, d, n in cases:
            with self.subTest(shape=gram.shape, dtype=gram.dtype, value=gram.flat[0], d=d):
                code, report = self.numeric(gram, d=d, n=n)
                self.assertEqual(code, 2)
                self.assertFalse(report["passed"])
                self.assertTrue(report["error"])

    def test_finite_input_whose_residual_overflows_fails_with_valid_json(self):
        gram = np.eye(7)
        gram[0, 0] = 1e300
        code, report = self.numeric(gram)
        self.assertEqual(code, 1)
        self.assertFalse(report["checks"]["tightness"])
        self.assertIsNone(report["metrics"]["tightness_frobenius"])

    def test_nonreal_symbolic_etf(self):
        code, report = self.symbolic(symbolic_harmonic_gram())
        self.assertEqual(code, 0)
        self.assertTrue(report["passed"])
        self.assertEqual(report["mode"], "symbolic")
        self.assertEqual(report["checks"], dict.fromkeys(CHECKS, True))
        self.assertEqual(report["first_failure"], {})

    def test_radical_symbolic_etf(self):
        scale = "-1/sqrt(2)"
        document = {"format": "etf-gram-v1", "gram": [[1, scale, scale], [scale, 1, scale], [scale, scale, 1]]}
        code, report = self.symbolic(document, d=2, n=3)
        self.assertEqual(code, 1)
        self.assertFalse(report["checks"]["welch_modulus"])
        document["gram"] = [[1, "-sqrt(1/4)", "-1/2"], ["-1/2", 1, "-1/2"], ["-1/2", "-1/2", 1]]
        code, report = self.symbolic(document, d=2, n=3)
        self.assertEqual(code, 0)
        self.assertTrue(report["passed"])

    def test_symbolic_rejects_perturbation_that_passes_numerically(self):
        gram = harmonic_gram()
        gram[0, 0] += 1e-15
        self.assertEqual(self.numeric(gram)[0], 0)
        document = symbolic_harmonic_gram()
        document["gram"][0][0] = "1+1/10**15"
        code, report = self.symbolic(document)
        self.assertEqual(code, 1)
        self.assertTrue(report["checks"]["hermitian"])
        self.assertFalse(report["checks"]["unit_diagonal"])
        self.assertTrue(report["checks"]["welch_modulus"])
        self.assertFalse(report["checks"]["tightness"])
        self.assertEqual(report["first_failure"]["unit_diagonal"], [0, 0])

    def test_symbolic_rejects_nonexact_or_outside_grammar(self):
        invalid_entries = (
            1.0, "1.0", "1j", "x", "I.real", "open('anything')", "1/0", "sqrt(-1)", True,
            "0/(1+zeta(3)+zeta(3)**2)", "(I-I)**-1",
        )
        for invalid in invalid_entries:
            with self.subTest(entry=invalid):
                document = {"format": "etf-gram-v1", "gram": [[invalid, 0, 0], [0, 1, 0], [0, 0, 1]]}
                code, report = self.symbolic(document, d=2, n=3)
                self.assertEqual(code, 2)
                self.assertFalse(report["passed"])
                self.assertIn("[0,0]", report["error"])


if __name__ == "__main__":
    unittest.main()
