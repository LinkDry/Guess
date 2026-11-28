"""Recover exact equality-feasible degree-six moments from a numerical SDP point."""
import argparse
from collections import defaultdict
from fractions import Fraction
from math import lcm
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sympy import QQ
from sympy.polys.matrices import DomainMatrix


def fixed_quartic_kernel_rows(model, parity):
    """Use the exact kernel of each pinned quartic block in its cubic columns."""
    from flint import fmpz_mat

    quartic = {0: Fraction(1)}
    for witness in model['equality_witnesses']:
        if not isinstance(witness, dict) or witness.get('type') != 'fixed_moment':
            continue
        index = model['moments'].index(witness['monomial'])
        coeff = tuple(map(Fraction, witness['value']))
        if coeff[1 - parity[index]]:
            raise ValueError('pinned quartic coordinates have mixed parity')
        quartic[index] = coeff[parity[index]]
    if len(quartic) != 107:
        raise ValueError('quartic facial reconstruction requires all 107 quartic moments pinned')
    constraints = set()
    for block in model['psd_blocks']:
        degrees = [len(polynomial[0][0]) for polynomial in block['basis']]
        low = [index for index, degree in enumerate(degrees) if degree < 3]
        high = [index for index, degree in enumerate(degrees) if degree == 3]
        positions = {index: number for number, index in enumerate(low)}
        top = [[Fraction(0) for _ in low] for _ in low]
        cross = defaultdict(dict)
        for row, col, incidence in block['entries']:
            # Divide the i-th row and column by beta for odd-degree basis polynomials.
            scale = Fraction(5, 2) if degrees[row] % 2 and degrees[col] % 2 else Fraction(1)
            weights = {int(index): coefficient * scale for index, coefficient in incidence.items()}
            if row in positions and col in positions:
                i, j = positions[row], positions[col]
                top[i][j] = top[j][i] = sum(quartic[index] * value for index, value in weights.items())
            elif row in positions or col in positions:
                inside, outside = (row, col) if row in positions else (col, row)
                cross[outside][positions[inside]] = weights
        common = lcm(*(value.denominator for line in top for value in line))
        kernel, nullity = fmpz_mat(len(low), len(low), [int(value * common) for line in top for value in line]).nullspace()
        for column in range(nullity):
            for outside in high:
                combined = defaultdict(Fraction)
                for inside, weights in cross[outside].items():
                    weight = int(kernel[inside, column])
                    for index, coefficient in weights.items():
                        combined[index] += coefficient * weight
                row = sorted((index, value) for index, value in combined.items() if value)
                if row:
                    constraints.add(tuple((index, value / row[0][1]) for index, value in row))
    return sorted(constraints)


def recover(model, point, denominator, quartic_face=False):
    count = len(model['moments'])
    parity = [len(monomial) % 2 for monomial in model['moments']]
    rows = {0: {0: QQ.one, count: QQ.one}}
    for row_number, row in enumerate(model['equalities'], 1):
        terms = {}
        equation_parity = set()
        for index, coefficients in row.items():
            i = int(index)
            a, b = map(Fraction, coefficients)
            active = int(bool(b))
            if a and b:
                raise ValueError('a coefficient has mixed parity')
            coefficient = b if active else a
            equation_parity.add((active + parity[i]) % 2)
            if active and parity[i]:
                coefficient *= Fraction(2, 5)
            terms[i] = QQ(coefficient.numerator, coefficient.denominator)
        if len(equation_parity) != 1:
            raise ValueError('an equation has mixed parity')
        rows[row_number] = terms

    face = fixed_quartic_kernel_rows(model, parity) if quartic_face else []
    for row in face:
        rows[len(rows)] = {i: QQ(value.numerator, value.denominator) for i, value in row}
    matrix = DomainMatrix(rows, (len(rows), count + 1), QQ)
    reduced, pivots = matrix.rref(method="GJ_dense")
    if len(point) != count or not np.isfinite(point).all() or denominator < 1:
        raise ValueError("invalid numerical candidate or rounding denominator")
    if count in pivots:
        raise ValueError('the linear equalities are inconsistent')
    beta = np.sqrt(10) / 5
    scaling = np.where(parity, beta, 1)
    rounded = [QQ(round(float(value) * denominator), denominator) for value in point / scaling]
    pivot_rows = reduced.to_sdm()
    for row_number, pivot in enumerate(pivots):
        row = pivot_rows[row_number]
        rounded[pivot] = row.get(count, QQ.zero) - sum(
            coefficient * rounded[index] for index, coefficient in row.items() if index not in (count, pivot)
        )
    if any(sum(coefficient * rounded[i] for i, coefficient in row.items() if i != count) != row.get(count, QQ.zero)
           for row in rows.values()):
        raise ValueError('reconstructed vector failed exact equality check')
    coefficients = [['0', str(value)] if high else [str(value), '0'] for value, high in zip(rounded, parity)]
    numeric = scaling * np.array(list(map(float, rounded)))
    return {'schema': 'anchored-exact-primal-v1', 'coefficient_basis': model['coefficient_basis'],
            'moments': coefficients, 'rational_equality_rank': len(pivots),
            'fixed_quartic_kernel_rows': len(face),
            'maximum_coordinate_change': float(np.max(abs(numeric - point)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--denominator', type=int, default=10**12)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--quartic-face', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    begin = perf_counter()
    model = json.loads((args.model or args.run / 'solver_model.json').read_text(encoding='utf-8'))
    with np.load(args.run / 'solver_primal_dual.npz', allow_pickle=False) as solution:
        point = solution['x']
    doc = recover(model, point, args.denominator, args.quartic_face)
    doc['wall_seconds'] = perf_counter() - begin
    path = args.output or args.run / 'exact_primal_candidate.json'
    path.write_text(json.dumps(doc, indent=2) + '\n', encoding='utf-8', newline='\n')
    print({key: value for key, value in doc.items() if key != 'moments'}, flush=True)


if __name__ == '__main__':
    main()
