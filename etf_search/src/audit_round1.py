"""Reproduce the independent audit of Round 1's best Gram matrix on stdout.

Run with the same NumPy and SymPy versions for a byte-for-byte comparison
of the numeric diagnostics to the archived report.
"""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import sympy as sp


def abs_squared(value):
    real, imag = value.as_real_imag()
    return real**2 + imag**2


def main():
    src = Path(__file__).resolve().parent
    matrix = src.parent / "results" / "round1" / "best_gram.npy"
    original = matrix.read_bytes()
    verifier = subprocess.run(
        [sys.executable, "-B", str(src / "verify_etf.py"), str(matrix), "--d", "4", "--n", "9"],
        capture_output=True,
        text=True,
    )
    if verifier.returncode not in (0, 1) or verifier.stderr:
        raise SystemExit(f"Independent verifier failed: {verifier.stderr or verifier.stdout}")
    if matrix.read_bytes() != original:
        raise SystemExit("The input matrix changed during verification")
    report = json.loads(verifier.stdout)

    raw = np.load(io.BytesIO(original), allow_pickle=False)
    if raw.shape != (9, 9) or raw.dtype != np.complex128 or not np.isfinite(raw).all():
        raise SystemExit("The Round 1 audit requires a finite 9-by-9 complex128 Gram matrix")
    gram = sp.Matrix(9, 9, lambda i, j: (
        sp.Rational(float(raw[i, j].real)) + sp.I * sp.Rational(float(raw[i, j].imag))
    ))
    permutation = [3 * orbit + (x + 1) % 3 for orbit in range(3) for x in range(3)]
    if gram != gram.H or gram.diagonal() != sp.ones(1, 9) or gram.extract(permutation, permutation) != gram:
        raise SystemExit("The Gram matrix violates the exact Hermitian, diagonal, or free C3 audit conditions")

    mu_squared = sp.Rational(5, 32)
    scale = sp.Rational(9, 4)
    modulus_error = max(
        abs(abs_squared(gram[i, j]) - mu_squared)
        for i in range(9) for j in range(9) if i != j
    )
    residual_squared = sum(abs_squared(entry) for entry in gram**2 - scale * gram)
    residual = sp.sqrt(residual_squared)

    # The radial correction is at most 3E per off-diagonal entry since mu > 1/3.
    # Its Frobenius norm is bounded by 27E since sqrt(72) < 9.
    projection_distance = 27 * modulus_error
    if sum(abs_squared(entry) for entry in gram) >= 25:
        raise SystemExit("The Gram matrix exceeds the norm bound used in the audit")
    residual_change = (5 + sp.Rational(9, 2) + scale) * projection_distance
    gamma = (7 - 2 * sp.sqrt(10)) / 18
    lower_bound = scale**2 * gamma / (2 * (2 * sp.sqrt(3) + sp.sqrt(2)))
    above_bound = bool(residual_squared > (lower_bound + residual_change)**2)

    report.update(
        verifier_exit_code=verifier.returncode,
        gram_file=matrix.name,
        gram_sha256=hashlib.sha256(original).hexdigest(),
        stored_dtype=raw.dtype.name,
        exact_arithmetic_on_stored_entries={
            "method": "Treat real and imaginary binary64 components as exact dyadic rationals.",
            "hermitian": True,
            "unit_diagonal": True,
            "c3_invariant": True,
            "c3_permutation_zero_based": permutation,
            "max_squared_modulus_error_exact": str(modulus_error),
            "max_squared_modulus_error_approx": str(modulus_error.evalf(35)),
            "tightness_residual_squared_exact": str(residual_squared),
            "tightness_residual_frobenius_approx": str(residual.evalf(45)),
        },
        c3_residual_bound={
            "exact_modulus_lower_bound_formula": "(9/4)^2*(7-2*sqrt(10))/(36*(2*sqrt(3)+sqrt(2)))",
            "exact_modulus_lower_bound_approx": str(lower_bound.evalf(35)),
            "distance_to_radial_projection_upper_bound_exact": str(projection_distance),
            "residual_change_upper_bound_exact": str(residual_change),
            "residual_change_upper_bound_approx": str(residual_change.evalf(35)),
            "projected_candidate_above_lower_bound": above_bound,
        },
    )
    output = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    sys.stdout.write(output)


if __name__ == "__main__":
    main()
