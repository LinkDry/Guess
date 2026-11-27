"""Exact feasible moments for an S9-invariant Bargmann triangle relaxation."""
import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from fractions import Fraction


def build_witness():
    """Construct centered Gram covariances as rational projection matrices."""
    vertices = range(9)
    edges = list(combinations(vertices, 2))
    triangles = list(combinations(vertices, 3))
    edge_index = {edge: i for i, edge in enumerate(edges)}
    unsigned = np.zeros((len(edges), len(triangles)), dtype=np.int64)
    boundary = np.zeros_like(unsigned)
    for index, (i, j, k) in enumerate(triangles):
        for sign, edge in [(1, (i, j)), (-1, (i, k)), (1, (j, k))]:
            unsigned[edge_index[edge], index] = 1
            boundary[edge_index[edge], index] = sign

    edge_gram = unsigned @ unsigned.T
    inverse_numerator = edge_gram @ edge_gram - 38 * edge_gram + 417 * np.eye(36, dtype=np.int64)
    real_denominator, imaginary_denominator = 1260, 9
    real_projection = real_denominator * np.eye(84, dtype=np.int64) - unsigned.T @ inverse_numerator @ unsigned
    imaginary_projection = imaginary_denominator * np.eye(84, dtype=np.int64) - boundary.T @ boundary
    real_rank = int(np.trace(real_projection)) // real_denominator
    imaginary_rank = int(np.trace(imaginary_projection)) // imaginary_denominator

    beta_squared = Fraction(2, 5)
    mean_squared = beta_squared / 49
    real_weight = (1 - mean_squared) * 84 / (2 * real_rank)
    imaginary_weight = (1 - mean_squared) * 84 / (2 * imaginary_rank)
    four_cycle_mean = Fraction(-11, 70)
    triangulations = {}
    for i, j, k, l in combinations(vertices, 4):
        # Compare the means of both triangulations of the cycle i-j-k-l-i.
        for a, b in [((i, j, k), (i, k, l)), ((i, j, l), (j, k, l))]:
            t, u = triangles.index(a), triangles.index(b)
            overlap = set(a) & set(b)
            edge = edge_index[tuple(sorted(overlap))]
            orientation = -int(boundary[edge, t] * boundary[edge, u])
            value = (mean_squared
                     + real_weight * Fraction(int(real_projection[t, u]), real_denominator)
                     - orientation * imaginary_weight * Fraction(int(imaginary_projection[t, u]), imaginary_denominator))
            triangulations[value] = triangulations.get(value, 0) + 1
    trace_four = 9 * 8**2 + beta_squared * 9 * 8
    repeated_vertex_terms = 2 * 9 * 8**2 - 9 * 8
    four_cycle_terms = 9 * 8 * 7 * 6
    exact_checks = {
        "unsigned_edge_gram_inverse": bool(np.array_equal(edge_gram @ inverse_numerator, 1260 * np.eye(36, dtype=np.int64))),
        "real_covariance_projection": bool(np.array_equal(real_projection @ real_projection, real_denominator * real_projection)),
        "imaginary_covariance_projection": bool(np.array_equal(imaginary_projection @ imaginary_projection, imaginary_denominator * imaginary_projection)),
        "real_linear_constraints": bool(not np.any(unsigned @ real_projection)),
        "imaginary_linear_constraints": bool(not np.any(boundary @ imaginary_projection)),
        "unit_modulus_moments": all(mean_squared + real_weight * Fraction(int(real_projection[i, i]), real_denominator)
                                     + imaginary_weight * Fraction(int(imaginary_projection[i, i]), imaginary_denominator) == 1
                                     for i in range(84)),
        "four_cycle_means": triangulations == {four_cycle_mean: 252},
        "fourth_trace": trace_four == repeated_vertex_terms + four_cycle_terms * four_cycle_mean,
    }
    return {"scope": "S9-invariant first and second moments of all triangles with four-cycle first moments",
            "psd_block_sizes": [85, 84], "triangle_mean": "sqrt(2/5)/7",
            "centered_covariance_ranks": {"real": real_rank, "imaginary": imaginary_rank},
            "centered_covariance_weights": {"real": str(real_weight), "imaginary": str(imaginary_weight)},
            "four_cycle_mean": str(four_cycle_mean), "trace_signature_fourth_power": str(trace_four),
            "four_cycle_triangulations_tested": sum(triangulations.values()), "exact_checks": exact_checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    path = parser.parse_args().output
    data = build_witness()
    if not all(data["exact_checks"].values()):
        raise SystemExit("Exact moment witness check failed")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
