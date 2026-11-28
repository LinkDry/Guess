"""Exact orbit representatives for the rank constraints of ETF(4,9)."""
from collections import defaultdict
from functools import cache
from itertools import combinations, permutations

from general_4_9_model import edge


ANCHOR = 8


def minor_representatives(size, anchored):
    """List row/column subset orbits using overlap and anchor membership."""
    indices = list(combinations(range(9 if anchored else 8), size))
    representatives = {}
    for position, rows in enumerate(indices):
        for columns in indices[position:]:
            flags = sorted((ANCHOR in rows, ANCHOR in columns))
            overlap = len(set(rows) & set(columns))
            representatives.setdefault((overlap, *flags), (rows, columns))
    return tuple(representatives.values())


def multiply(lvalue, rvalue):
    """Multiply Gaussian Q(beta) coefficients. The two pairs store real and imaginary parts."""
    ar, br, ai, bi = lvalue
    cr, dr, ci, di = rvalue
    # All beta coordinates in determinant products are multiples of 20.
    assert all(value % 20 == 0 for value in (br, bi, dr, di))
    return (ar * cr - ai * ci + (2 * (br * dr - bi * di)) // 5,
            ar * dr + br * cr - ai * di - bi * ci,
            ar * ci + ai * cr + (2 * (br * di + bi * dr)) // 5,
            ar * di + br * ci + ai * dr + bi * cr)


def matrix_entry(i, j, schur):
    if i == j:
        return [((), (27 if schur else 32, 0, 0, 0))]
    if ANCHOR in (i, j):
        return [((), (0, 20, 0, 0))]
    x, _ = edge('x', i, j)
    y, orientation = edge('y', i, j)
    entries = [((x,), (0, 20, 0, 0)), ((y,), (0, 0, 0, 20 * orientation))]
    if schur:
        entries.append(((), (-5, 0, 0, 0)))
    return entries


@cache
def column_orders(size):
    return tuple((order, (-1)**sum(a > b for i, a in enumerate(order) for b in order[i+1:]))
                 for order in permutations(range(size)))


def determinant(rows, columns, *, schur=False):
    """Expand the determinant of the minor of 32G or of 32H, where H is the anchor Schur complement."""
    matrix = [[matrix_entry(i, j, schur) for j in columns] for i in rows]
    total = defaultdict(lambda: [0, 0, 0, 0])
    for order, sign in column_orders(len(rows)):
        terms = [((), (sign, 0, 0, 0))]
        for row, col in enumerate(order):
            terms = [(left + right, multiply(lvalue, rvalue))
                     for left, lvalue in terms for right, rvalue in matrix[row][col]]
        for monomial, coefficient in terms:
            target = total[tuple(sorted(monomial))]
            for index, value in enumerate(coefficient):
                target[index] += value
    return {key: tuple(value) for key, value in total.items() if any(value)}


@cache
def rank_equations():
    equations = {}
    for name, size, schur in [('gram5', 5, False), ('schur4', 4, True)]:
        for rows, cols in minor_representatives(size, not schur):
            stem = name + '_' + ''.join(map(str, rows)) + '_' + ''.join(map(str, cols))
            coefficients = determinant(rows, cols, schur=schur)
            for offset, suffix in [(0, 'real'), (2, 'imag')]:
                part = {m: c[offset:offset+2] for m, c in coefficients.items() if any(c[offset:offset+2])}
                if part:
                    equations[stem + '_' + suffix] = part
    return equations
