"""Compare the switch-equivalence classes of the first two rounds numerically."""

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from search_unconstrained import FREE_I, FREE_J, phase_objective


def anchor(gram):
    gauge = np.ones(len(gram), dtype=complex)
    gauge[1:] = np.conj(gram[0, 1:]) / np.abs(gram[0, 1:])
    return gauge.conj()[:, None] * gram * gauge[None, :]


def fingerprints(gram):
    """Sorted real parts of the 28 triangle products through each vertex."""
    triangles = np.einsum('ij,jk,ki->ijk', gram, gram, gram)
    return np.array([
        np.sort([triangles[i, j, k].real for j, k in combinations((v for v in range(9) if v != i), 2)])
        for i in range(9)
    ])


def match_switching(reference, reference_fingerprints, candidate, tolerance):
    differences = np.max(abs(reference_fingerprints[:, None] - fingerprints(candidate)[None, :]), axis=-1)
    options = [np.flatnonzero(row < tolerance) for row in differences]
    anchored = anchor(reference)
    best = (float('inf'), None, None)
    def injective_orders(chosen=(), seen=frozenset()):
        if len(chosen) == 9:
            yield chosen
            return
        for vertex in options[len(chosen)]:
            if vertex not in seen:
                yield from injective_orders((*chosen, vertex), seen | {vertex})

    for permutation in injective_orders():
        ordered = anchor(candidate[np.ix_(permutation, permutation)])
        for conjugate in (False, True):
            aligned = ordered.conj() if conjugate else ordered
            error = float(np.max(np.abs(anchored - aligned)))
            if error < best[0]:
                best = (error, list(map(int, permutation)), conjugate)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parents[1] / 'results')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    first = np.load(args.results / 'round1' / 'best_gram.npy', allow_pickle=False)
    second = np.load(args.results / 'round2_unconstrained' / 'best_gram.npy', allow_pickle=False)
    endpoints = np.load(args.results / 'round2_unconstrained' / 'all_final_grams.npy', allow_pickle=False)
    tolerance = 1e-6
    reference_fingerprints = fingerprints(first)
    error, permutation, conjugation = match_switching(first, reference_fingerprints, second, tolerance)
    errors = [match_switching(first, reference_fingerprints, item, tolerance)[0] for item in endpoints]
    matches = [value for value in errors if value < tolerance]
    angles = np.angle(second[FREE_I, FREE_J])
    hessian_checks = []
    for step in (1e-4, 1e-5, 1e-6):
        offsets = step * np.eye(len(angles))
        hessian = np.column_stack([(phase_objective(angles + offset)[1] - phase_objective(angles - offset)[1])
                                   / (2 * step) for offset in offsets])
        values = np.linalg.eigvalsh((hessian + hessian.T) / 2)
        hessian_checks.append({'finite_difference_step': step, 'min_eigenvalue': float(values[0]),
                               'max_eigenvalue': float(values[-1]),
                               'max_asymmetry': float(np.max(abs(hessian - hessian.T)))})
    report = {
        'round1_best_vs_round2_best': {'max_entry_error': error if np.isfinite(error) else None,
                                       'permutation_zero_based': permutation, 'conjugation': conjugation},
        'round2_endpoints_vs_round1_best': {'tolerance': tolerance, 'tested': len(endpoints),
                                            'matched': len(matches), 'max_error_among_matches': max(matches, default=None)},
        'round2_best_gram_eigenvalues': np.linalg.eigvalsh(second).tolist(),
        'round2_best_gradient_max_absolute': float(np.linalg.norm(phase_objective(angles)[1], np.inf)),
        'round2_best_phase_hessian': hessian_checks,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
