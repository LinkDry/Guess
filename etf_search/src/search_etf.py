"""Fixed-seed calibration of free-C3 complex ETF(4,9) projections."""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
import scipy
from scipy.optimize import minimize


D, N = 4, 9
LAMBDA = N / D
MU = np.sqrt(5 / 32)
TOLERANCE = 1e-12
C3_RESIDUAL_LOWER_BOUND = 1e-2  # See ../ansatz.md for its scope and proof.
RANKS = (2, 1, 1)
METHODS = ("drs", "ap")
GAUGE_POSITIONS = ((0, 1, 0), (0, 2, 0))
INDEPENDENT_POSITIONS = tuple((a, a, 1) for a in range(3)) + tuple(
    (a, b, t) for a in range(3) for b in range(a + 1, 3) for t in range(3)
)
PHASE_POSITIONS = tuple(pos for pos in INDEPENDENT_POSITIONS if pos not in GAUGE_POSITIONS)
PHASE_INDEX = tuple(np.array(PHASE_POSITIONS).T)
PHASE_CONJUGATES = (PHASE_INDEX[1], PHASE_INDEX[0], (-PHASE_INDEX[2]) % 3)
ROW_ORBIT = (np.arange(N) // 3)[:, None]
COL_ORBIT = ROW_ORBIT.T
OFFSET = (np.arange(N)[None, :] - np.arange(N)[:, None]) % 3
OFF_DIAGONAL = ~np.eye(N, dtype=bool)


def phase_coefficients(phases):
    """Build g[a,b,t] with two representatives fixed to +mu by rephasing."""
    coefficients = np.zeros((3, 3, 3), dtype=np.complex128)
    coefficients[np.arange(3), np.arange(3), 0] = 1.0
    for a, b, t in GAUGE_POSITIONS:
        coefficients[a, b, t] = coefficients[b, a, (-t) % 3] = MU
    values = MU * np.exp(1j * phases)
    coefficients[PHASE_INDEX] = values
    coefficients[PHASE_CONJUGATES] = values.conj()
    return coefficients


def coefficient_phases(coefficients):
    return np.angle(coefficients[PHASE_INDEX])


def gram_from_coefficients(coefficients):
    return coefficients[ROW_ORBIT, COL_ORBIT, OFFSET]


def to_sectors(coefficients):
    """Forward transform with positive exponent, ordered [k,a,b]."""
    return np.moveaxis(3 * np.fft.ifft(coefficients, axis=2), 2, 0)


def from_sectors(sectors):
    return np.fft.fft(np.moveaxis(sectors, 0, 2), axis=2) / 3


def project_modulus(coefficients):
    """Frobenius projection onto unit diagonal and fixed Welch modulus.

    Invariant Hermitian entries have equal multiplicities in each conjugate
    pair. Average the pair first to remove any floating-point asymmetry.
    A zero representative maps to phase zero.
    """
    entries = (coefficients[PHASE_INDEX] + coefficients[PHASE_CONJUGATES].conj()) / 2
    return phase_coefficients(np.angle(entries))


def project_spectrum(coefficients):
    """Frobenius projection onto Hermitian sectors with specified ranks."""
    sectors = to_sectors(coefficients)
    sectors = (sectors + sectors.conj().swapaxes(1, 2)) / 2
    _, vectors = np.linalg.eigh(sectors)
    projected = np.empty_like(sectors)
    for k, rank in enumerate(RANKS):
        top = vectors[k, :, -rank:]
        projected[k] = LAMBDA * (top @ top.conj().T)
    return from_sectors(projected)


def phase_objective(phases):
    """Return half the squared raw Gram residual and its real phase gradient."""
    coefficients = phase_coefficients(phases)
    gram = gram_from_coefficients(coefficients)
    residual = gram @ gram - LAMBDA * gram
    derivative = gram @ residual + residual @ gram - LAMBDA * residual
    a, b, t = PHASE_INDEX
    # Three translated copies and three conjugate copies of each representative.
    gradient = 6 * np.real(derivative[3 * a, 3 * b + t].conj() * 1j * coefficients[PHASE_INDEX])
    return 0.5 * float(np.vdot(residual, residual).real), gradient


def gram_metrics(gram):
    """Measure the same unmodified full Gram matrix that is saved to disk."""
    return {
        "tightness_frobenius": float(np.linalg.norm(gram @ gram - LAMBDA * gram, "fro")),
        "welch_max_error": float(np.max(np.abs(np.abs(gram[OFF_DIAGONAL]) - MU))),
        "hermitian_max_error": float(np.max(np.abs(gram - gram.conj().T))),
        "diagonal_max_error": float(np.max(np.abs(gram.diagonal() - 1))),
    }


def passes_numeric(metrics):
    return all(metrics[key] < TOLERANCE for key in (
        "tightness_frobenius", "welch_max_error", "hermitian_max_error", "diagonal_max_error",
    ))


def run_seed(method, seed, *, steps, relaxation, refine_maxiter):
    """Score every modulus projection and retain the best before refinement."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    rng = np.random.Generator(np.random.PCG64(seed))
    state = phase_coefficients(rng.uniform(-np.pi, np.pi, len(PHASE_POSITIONS)))
    shadow = state
    initial = gram_metrics(gram_from_coefficients(shadow))
    best_projected = last_projected = {"step": 0, **initial}
    best_coefficients = shadow

    for step in range(1, steps + 1):
        if method == "drs":
            spectral = project_spectrum(2 * shadow - state)
            state = state + relaxation * (spectral - shadow)
            shadow = project_modulus(state)
        else:
            shadow = project_modulus(project_spectrum(shadow))
        last_projected = {"step": step, **gram_metrics(gram_from_coefficients(shadow))}
        if last_projected["tightness_frobenius"] < best_projected["tightness_frobenius"]:
            best_projected, best_coefficients = last_projected, shadow

    refined = minimize(
        phase_objective, coefficient_phases(best_coefficients), jac=True, method="L-BFGS-B",
        options={"maxiter": refine_maxiter, "ftol": np.finfo(float).eps, "gtol": 1e-12, "maxls": 40},
    )
    refined_gram = gram_from_coefficients(phase_coefficients(refined.x))
    refined_metrics = gram_metrics(refined_gram)
    if not all(np.isfinite(list(refined_metrics.values()))):
        raise RuntimeError(f"{method} seed {seed}: nonfinite phase refinement")
    if refined_metrics["tightness_frobenius"] <= best_projected["tightness_frobenius"]:
        best = {"stage": "lbfgs", **refined_metrics}
        best_gram = refined_gram
    else:
        best = {"stage": "projection", **best_projected}
        best_gram = gram_from_coefficients(best_coefficients)
    return {
        "method": method,
        "seed": seed,
        "initial": {**initial, "numeric_gate_passed": passes_numeric(initial)},
        "best_projected": {**best_projected, "numeric_gate_passed": passes_numeric(best_projected)},
        "last_projected": {**last_projected, "numeric_gate_passed": passes_numeric(last_projected)},
        "refined": {**refined_metrics, "numeric_gate_passed": passes_numeric(refined_metrics)},
        "best": {**best, "numeric_gate_passed": passes_numeric(best)},
        "lbfgs": {
            "iterations": int(refined.nit), "function_evaluations": int(refined.nfev),
            "success": bool(refined.success), "status": int(refined.status), "message": str(refined.message),
            "gradient_inf_norm": float(np.linalg.norm(refined.jac, ord=np.inf)),
        },
    }, best_gram


def attempt_symbolic(gram, directory):
    """Try bounded PSLQ in a small radical basis, then call the exact verifier."""
    if not passes_numeric(gram_metrics(gram)):
        return {"status": "numeric_gate_not_met"}

    import mpmath as mp

    def expression(value):
        if abs(value) < 2e-14:
            return "0"
        terms = ("1", "sqrt(2)", "sqrt(3)", "sqrt(5)")
        basis = [mp.mpf(1), mp.sqrt(2), mp.sqrt(3), mp.sqrt(5)]
        relation = mp.pslq(mp.matrix([mp.mpf(float(value)), *basis]), tol=mp.mpf("2e-14"), maxcoeff=64)
        if relation is None or relation[0] == 0:
            return None
        constant, *coefficients = relation
        numerator = "+".join(f"({-c})*{term}" for c, term in zip(coefficients, terms) if c) or "0"
        return f"({numerator})/({constant})"

    exact = [[1 for _ in range(N)] for _ in range(N)]
    with mp.workdps(60):
        for a, b, offset in INDEPENDENT_POSITIONS:
            phase = gram[3 * a, 3 * b + offset] / MU
            real, imag = expression(phase.real), expression(phase.imag)
            if real is None or imag is None:
                return {"status": "phase_recognition_failed", "coefficient_index": [a, b, offset]}
            for x in range(3):
                i, j = 3 * a + x, 3 * b + (x + offset) % 3
                exact[i][j] = f"sqrt(5/32)*(({real})+I*({imag}))"
                exact[j][i] = f"sqrt(5/32)*(({real})-I*({imag}))"

    path = directory / "candidate_exact.json"
    path.write_text(json.dumps({"format": "etf-gram-v1", "gram": exact}, indent=2) + "\n", encoding="utf-8")
    command = [sys.executable, "-B", str(Path(__file__).with_name("verify_etf.py")),
               str(path), "--d", str(D), "--n", str(N), "--symbolic"]
    check = subprocess.run(command, capture_output=True, text=True)
    return {
        "status": "exact_passed" if check.returncode == 0 else "exact_verifier_rejected",
        "path": path.name, "verifier_returncode": check.returncode,
        "verifier_stdout": check.stdout, "verifier_stderr": check.stderr,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=48, help="number of seeds used by each method")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=1800, help="projection updates per method per seed")
    parser.add_argument("--relaxation", type=float, default=1.0, help="DRS relaxation in (0, 2)")
    parser.add_argument("--refine-maxiter", type=int, default=750)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.seeds < 1 or args.seed_start < 0 or args.steps < 1 or args.refine_maxiter < 1:
        parser.error("seeds, steps and refine-maxiter must be positive and seed-start must be nonnegative")
    if not 0 < args.relaxation < 2:
        parser.error("relaxation must be in (0, 2)")

    start = perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    overall = None
    overall_gram = None
    symbolic = {"status": "numeric_gate_not_met"}
    for count, seed in enumerate(range(args.seed_start, args.seed_start + args.seeds), start=1):
        for method in METHODS:
            row, gram = run_seed(method, seed, steps=args.steps, relaxation=args.relaxation,
                                 refine_maxiter=args.refine_maxiter)
            runs.append(row)
            if overall is None or row["best"]["tightness_frobenius"] < overall["tightness_frobenius"]:
                overall = {"method": method, "seed": seed, **row["best"]}
                overall_gram = gram
            if row["best"]["numeric_gate_passed"] and symbolic["status"] == "numeric_gate_not_met":
                symbolic = {"method": method, "seed": seed, **attempt_symbolic(gram, args.output_dir)}
        if count % 8 == 0 or count == args.seeds:
            print(f"{count}/{args.seeds} seeds per method | best raw Gram residual "
                  f"{overall['tightness_frobenius']:.16g} ({overall['method']}, seed {overall['seed']})", flush=True)

    out_path = args.output_dir / "best_gram.npy"
    np.save(out_path, overall_gram, allow_pickle=False)
    by_method = {}
    for method in METHODS:
        rows = [row for row in runs if row["method"] == method]
        by_method[method] = {}
        for stage in ("initial", "best_projected", "last_projected", "refined", "best"):
            ranked = sorted(rows, key=lambda row: row[stage]["tightness_frobenius"])
            values = [row[stage]["tightness_frobenius"] for row in ranked]
            by_method[method][stage] = {
                "seed_of_min": ranked[0]["seed"], "min": values[0],
                "median": float(np.median(values)), "max": values[-1],
            }
        by_method[method]["numeric_gate_pass_count"] = sum(row["best"]["numeric_gate_passed"] for row in rows)

    summary = {
        "model": {
            "d": D, "n": N, "lambda": LAMBDA, "mu": float(MU), "symmetry": "free C3",
            "vertex_order": "zero-based orbit-major 3*a+x", "offset": "(y-x) mod 3",
            "fourier_forward": "3*ifft(g, axis=2), output moved to [k,a,b]",
            "sector_ranks": list(RANKS), "gauge_positions_positive_mu": GAUGE_POSITIONS,
            "phase_positions_in_order": PHASE_POSITIONS, "free_real_phases": len(PHASE_POSITIONS),
            "analytic_raw_residual_lower_bound": C3_RESIDUAL_LOWER_BOUND,
            "bound_scope": "free C3, exact unit diagonal and Welch modulus; see ../../ansatz.md",
        },
        "settings": {
            "methods_in_order": METHODS, "seed_start": args.seed_start, "seeds_per_method": args.seeds,
            "steps": args.steps, "drs_relaxation": args.relaxation,
            "initial_rng": "NumPy Generator(PCG64(seed)), independent phases uniform [-pi, pi)",
            "refiner": "L-BFGS-B (unbounded) with analytic gradient of half squared raw Gram residual",
            "refine_maxiter": args.refine_maxiter, "refine_ftol": float(np.finfo(float).eps),
            "refine_gtol": 1e-12, "refine_maxls": 40, "numeric_strict_tolerance": TOLERANCE,
            "projection_scoring": "measure every modulus shadow, including step 0",
        },
        "reproduction": {
            "argv_from_repo_root": ["python", "-B", "etf_search/src/search_etf.py", *sys.argv[1:]],
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "thread_environment": {key: os.environ.get(key) for key in (
                "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            )},
        },
        "runtime_seconds": perf_counter() - start,
        "best_gram_file": out_path.name,
        "best": overall,
        "by_method": by_method,
        "symbolic": symbolic,
        "runs": runs,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    print(f"Saved {out_path} and {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
