"""Search the reflection-invariant Gram models for complex ETF(4,9)."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from search_etf import MU, LAMBDA as SCALE, gram_metrics


class AntiunitaryModel:
    """A Gram is invariant under conjugation followed by a fixed involution."""

    def __init__(self, n_fixed, sign=1):
        if n_fixed not in (1, 3) or sign not in (-1, 1) or (n_fixed == 1 and sign != 1):
            raise ValueError("Choose one fixed point or three fixed points with triangle sign ±1")
        self.permutation = np.arange(9)
        basis = np.eye(9, dtype=complex)
        for i in range(n_fixed, 9, 2):
            self.permutation[i:i + 2] = [i + 1, i]
            basis[i:i + 2, i:i + 2] = np.array([[1, 1j], [1, -1j]]) / np.sqrt(2)
        self.basis = basis
        self.template = np.eye(9, dtype=complex)
        self.template[0, 1:] = self.template[1:, 0] = MU
        if n_fixed == 3:
            self.template[1, 2] = self.template[2, 1] = sign * MU
        index, rows, cols, directions, representatives = [], [], [], [], []
        visited = set()
        for i in range(1, 9):
            for j in range(i + 1, 9):
                if (i, j) in visited or j < n_fixed:
                    continue
                u, v = self.permutation[[i, j]]
                image = (min(u, v), max(u, v))
                number = len(representatives)
                representatives.append((i, j))
                orbit = [((i, j), 1)]
                if image != (i, j):
                    orbit.append((image, -1 if u < v else 1))
                for (row, col), direction in orbit:
                    rows.append(row)
                    cols.append(col)
                    index.append(number)
                    directions.append(direction)
                    visited.add((row, col))
        self.dimension = len(representatives)
        self.phase_index = np.asarray(index)
        self.rows, self.cols = np.asarray(rows), np.asarray(cols)
        self.directions = np.asarray(directions)
        self.representatives = tuple(np.asarray(representatives).T)
        self.average = np.eye(self.dimension)[self.phase_index]

    def from_angles(self, phases):
        gram = np.broadcast_to(self.template, (*phases.shape[:-1], 9, 9)).copy()
        values = MU * np.exp(1j * phases[..., self.phase_index] * self.directions)
        gram[..., self.rows, self.cols] = values
        gram[..., self.cols, self.rows] = values.conj()
        return gram

    def project_modulus(self, gram):
        upper = (gram[..., self.rows, self.cols] + gram[..., self.cols, self.rows].conj()) / 2
        oriented = np.where(self.directions == 1, upper, upper.conj())
        return self.from_angles(np.angle(oriented @ self.average))

    def project_spectrum(self, gram):
        real = (self.basis.conj().T @ gram @ self.basis).real
        real = (real + real.swapaxes(-2, -1)) / 2
        _, vectors = np.linalg.eigh(real)
        lifted = self.basis @ vectors[..., -4:]
        return SCALE * (lifted @ lifted.conj().swapaxes(-2, -1))

    def objective(self, phases):
        gram = self.from_angles(phases)
        residual = gram @ gram - SCALE * gram
        derivative = gram @ residual + residual @ gram - SCALE * residual
        terms = 2 * np.real(derivative[self.rows, self.cols].conj()
                            * 1j * self.directions * gram[self.rows, self.cols])
        return 0.5 * float(np.vdot(residual, residual).real), terms @ self.average


def norms(grams):
    return np.linalg.norm(grams @ grams - SCALE * grams, axis=(-2, -1))


def search_batch(model, seeds, steps, maxiter):
    phases = np.array([np.random.default_rng(seed).uniform(-np.pi, np.pi, model.dimension) for seed in seeds])
    state = model.from_angles(phases)
    shadow = state
    best = state.copy()
    best_steps = np.zeros(len(seeds), dtype=int)
    best_residuals = norms(best)
    for step in range(1, steps + 1):
        state = state + model.project_spectrum(2 * shadow - state) - shadow
        shadow = model.project_modulus(state)
        residuals = norms(shadow)
        improved = residuals < best_residuals
        best_residuals[improved] = residuals[improved]
        best_steps[improved] = step
        best[improved] = shadow[improved]

    for index, seed in enumerate(seeds):
        projected = best[index]
        phase = np.angle(projected[model.representatives])
        refined = minimize(model.objective, phase, method="L-BFGS-B", jac=True,
                           options={"maxiter": maxiter, "ftol": np.finfo(float).eps,
                                    "gtol": 1e-12, "maxls": 40})
        final = model.from_angles(refined.x)
        projected_metrics, final_metrics = gram_metrics(projected), gram_metrics(final)
        # Preserve a projected numerical hit even if refinement increases its residual.
        retained = final if final_metrics["tightness_frobenius"] <= projected_metrics["tightness_frobenius"] else projected
        yield ({"seed": seed, "best_projected_step": int(best_steps[index]),
                "best_projected": projected_metrics, "final": final_metrics,
                "optimizer_status": int(refined.status), "optimizer_iterations": int(refined.nit),
                "gradient_inf_norm": float(np.linalg.norm(refined.jac, np.inf)),
                "retained": gram_metrics(retained)}, final, retained)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed", type=int, choices=[1, 3], required=True)
    parser.add_argument("--sign", type=int, choices=[-1, 1], default=1)
    parser.add_argument("--seeds", type=int, default=128)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.seeds, args.steps, args.maxiter) < 1:
        parser.error("seeds, steps and maxiter must be positive")
    model = AntiunitaryModel(args.fixed, args.sign)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    begin = perf_counter()
    records, final_grams = [], []
    best_gram, best_record = None, None
    for start in range(0, args.seeds, 32):
        for record, final, retained in search_batch(model, range(start, min(start + 32, args.seeds)), args.steps, args.maxiter):
            records.append(record)
            final_grams.append(final)
            if best_record is None or record["retained"]["tightness_frobenius"] < best_record["retained"]["tightness_frobenius"]:
                best_gram, best_record = retained, record
            if record["retained"]["tightness_frobenius"] < 1e-12:
                np.save(args.output_dir / f"hit_{record['seed']}.npy", retained, allow_pickle=False)
                print(f"NUMERIC GATE seed {record['seed']}", flush=True)
        print(f"{len(records)}/{args.seeds} seeds, best {best_record['retained']['tightness_frobenius']:.16g}", flush=True)
    ordered = sorted(records, key=lambda row: row["final"]["tightness_frobenius"])
    values = np.array([row["final"]["tightness_frobenius"] for row in ordered])
    output = {
        "model": {"fixed_points": args.fixed, "triangle_sign": args.sign,
                  "permutation": model.permutation.tolist(), "phase_dimension": model.dimension},
        "settings": {"seed_start": 0, "seeds": args.seeds, "steps": args.steps, "maxiter": args.maxiter},
        "seconds": perf_counter() - begin, "best": best_record,
        "distribution": {"quantiles": dict(zip(["min", "q25", "median", "q75", "max"],
                                                np.quantile(values, [0, .25, .5, .75, 1]).tolist())),
                         "count_within_1e-8_of_min": int(np.count_nonzero(values - values[0] < 1e-8))},
        "top_ten_final": [{"seed": row["seed"], **row["final"]} for row in ordered[:10]], "runs": records,
    }
    np.save(args.output_dir / "best_gram.npy", best_gram, allow_pickle=False)
    np.save(args.output_dir / "final_grams.npy", np.stack(final_grams), allow_pickle=False)
    (args.output_dir / "summary.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
