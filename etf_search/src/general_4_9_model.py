"""Complete signed S8-invariant order-two moment model for complex ETF(4,9)."""

from collections import defaultdict
from functools import cache
from itertools import combinations, combinations_with_replacement, permutations, product
from math import gcd

import numpy as np


N = 8
VARIABLES = tuple((kind, a, b) for a, b in combinations(range(N), 2) for kind in ('x', 'y'))
EDGE_IDS = {edge: i for i, edge in enumerate(VARIABLES)}
IDENTITY = tuple(range(N))
# This convention also works as an integer code for signed moment indices.
ZERO = 0
NUMBER_OF_VARIABLES = len(VARIABLES)


def edge(kind, a, b):
    """Return the variable index and the sign of its directed version."""
    if a < b:
        return EDGE_IDS[kind, a, b], 1
    return EDGE_IDS[kind, b, a], (-1 if kind == 'y' else 1)


def edge_action(vertex_map):
    """Encode each transformed variable as a signed, one-based index."""
    return tuple(sign * (index + 1) for index, sign in (
        edge(kind, vertex_map[a], vertex_map[b]) for kind, a, b in VARIABLES
        if a < len(vertex_map) and b < len(vertex_map)
    ))


def apply_action(monomial, action):
    image = [action[i] for i in monomial]
    sign = -1 if sum(i < 0 for i in image) % 2 else 1
    return tuple(sorted(abs(i) - 1 for i in image)), sign


@cache
def support_actions(size):
    # Variables on an initial segment of vertices are not an initial segment
    # of VARIABLES, so store the images at their original variable indices.
    domain = [i for i, (_, a, b) in enumerate(VARIABLES) if max(a, b) < size]
    return tuple(dict(zip(domain, edge_action(p))) for p in permutations(range(size)))


class SignedMoments:
    def __init__(self):
        self.representatives = [()]
        self._index = {(0, ()): 1}
        self._raw = {(): 1}

    def orbit(self, monomial):
        """Return sign*(moment_id+1), or zero for a sign-reversing stabilizer."""
        known = self._raw.get(monomial)
        if known is not None:
            return known
        if sum(i % 2 for i in monomial) % 2:
            self._raw[monomial] = ZERO
            return ZERO
        edges = [VARIABLES[i] for i in monomial]
        vertices = {vertex for _, a, b in edges for vertex in (a, b)}
        relabel = {vertex: i for i, vertex in enumerate(sorted(vertices))}
        size = len(vertices)
        pattern = tuple(sorted(EDGE_IDS[kind, relabel[a], relabel[b]] for kind, a, b in edges))
        compressed = size, pattern
        value = self._index.get(compressed)
        if value is None:
            signs = {}
            reversed_stabilizer = False
            for action in support_actions(size):
                image, sign = apply_action(pattern, action)
                previous = signs.setdefault(image, sign)
                if previous != sign:
                    reversed_stabilizer = True
            canonical = min(signs)
            if reversed_stabilizer:
                for image in signs:
                    self._index[size, image] = ZERO
            else:
                self.representatives.append(canonical)
                one_based = len(self.representatives)
                canonical_sign = signs[canonical]
                for image, sign in signs.items():
                    self._index[size, image] = canonical_sign * sign * one_based
            value = self._index[compressed]
        self._raw[monomial] = value
        return value


def base_equations():
    """One equation of each S8 type. Coefficients are a+b*beta with beta=sqrt(10)/5."""
    x01, y01 = EDGE_IDS['x', 0, 1], EDGE_IDS['y', 0, 1]
    circle = {(): (-1, 0), (x01, x01): (1, 0), (y01, y01): (1, 0)}
    row_real = {(): (0, -1), **{(EDGE_IDS['x', 0, j],): (1, 0) for j in range(1, N)}}
    row_imag = {(EDGE_IDS['y', 0, j],): (1, 0) for j in range(1, N)}
    tight_real = {(): (1, 0), (x01,): (0, -1)}
    tight_imag = {(y01,): (0, -1)}
    for k in range(2, N):
        for a_kind, b_kind, coefficient, equation in (
            ('x', 'x', 1, tight_real), ('y', 'y', -1, tight_real),
            ('x', 'y', 1, tight_imag), ('y', 'x', 1, tight_imag),
        ):
            a, a_sign = edge(a_kind, 0, k)
            b, b_sign = edge(b_kind, k, 1)
            equation[tuple(sorted((a, b)))] = (coefficient * a_sign * b_sign, 0)
    return {
        'circle': circle, 'row_real': row_real, 'row_imag': row_imag,
        'tight_real': tight_real, 'tight_imag': tight_imag,
    }


def build_equalities(orbits, monomials):
    rows = []
    witnesses = []
    seen = set()
    equations = base_equations()
    for name, coefficients in equations.items():
        degree = max(map(len, coefficients))
        for multiplier in monomials[4 - degree]:
            if sum(i % 2 for i in multiplier) % 2 != int(name.endswith('_imag')):
                continue
            reduced = defaultdict(lambda: [0, 0])
            for term, (a, b) in coefficients.items():
                moment = orbits.orbit(tuple(sorted(term + multiplier)))
                if moment:
                    pair = reduced[abs(moment) - 1]
                    sign = 1 if moment > 0 else -1
                    pair[0] += sign * a
                    pair[1] += sign * b
            nonzero = sorted((index, a, b) for index, (a, b) in reduced.items() if a or b)
            if not nonzero:
                continue
            divisor = gcd(*(coefficient for _, a, b in nonzero for coefficient in (a, b)))
            if (nonzero[0][1], nonzero[0][2]) < (0, 0):
                divisor = -divisor
            normalized = tuple((index, a // divisor, b // divisor) for index, a, b in nonzero)
            if normalized not in seen:
                rows.append({index: (a, b) for index, a, b in normalized})
                witnesses.append((name, multiplier, divisor))
                seen.add(normalized)
    return rows, witnesses, equations


def transposition(i, j):
    permutation = list(IDENTITY)
    permutation[i], permutation[j] = permutation[j], permutation[i]
    return edge_action(permutation)


def basis_blocks(monomials, split, *, fixed_vertices=4):
    """Build integer bases of the tail S4 fixed space and its character spaces."""
    if not 4 <= fixed_vertices <= N:
        raise ValueError("fixed_vertices must be between four and eight")
    generators = [(transposition(i, i + 1), 1) for i in range(fixed_vertices, N - 1)]
    heads = [transposition(0, 1), transposition(2, 3)]
    characters = product((1, -1), repeat=3) if split else [(1, 1, 0)]
    for first, second, conj in characters:
        actions = generators + list(zip(heads, (first, second))) if split else generators
        visited = set()
        basis = []
        for monomial in monomials:
            if monomial in visited or (conj and (-1)**sum(i % 2 for i in monomial) != conj):
                continue
            coefficients = {monomial: 1}
            pending = [monomial]
            consistent = True
            for term in pending:
                for action, character in actions:
                    image, sign = apply_action(term, action)
                    weight = sign * character * coefficients[term]
                    if image not in coefficients:
                        coefficients[image] = weight
                        pending.append(image)
                    elif coefficients[image] != weight:
                        consistent = False
            visited.update(coefficients)
            if consistent:
                basis.append(tuple(sorted(coefficients.items())))
        yield (first, second, conj) if split else None, basis


def build_psd_blocks(orbits, monomials, split):
    indexed_blocks = list(basis_blocks(monomials, split))
    monomial_ids = {monomial: i for i, monomial in enumerate(monomials)}
    pair_moments = np.zeros((len(monomials), len(monomials)), dtype=np.int32)
    # Skip rows absent from every retained basis. Every product of retained
    # polynomials can be read from the remaining principal submatrix.
    indices = sorted({monomial_ids[m] for _, block in indexed_blocks for p in block for m, _ in p})
    for offset, i in enumerate(indices):
        monomial = monomials[i]
        for j in indices[:offset + 1]:
            moment = orbits.orbit(tuple(sorted(monomial + monomials[j])))
            pair_moments[i, j] = pair_moments[j, i] = moment
    result = []
    for character, basis in indexed_blocks:
        data = [(np.array([monomial_ids[m] for m, _ in polynomial]),
                 np.array([sign for _, sign in polynomial])) for polynomial in basis]
        entries = []
        for row, (row_ids, row_signs) in enumerate(data):
            for col, (col_ids, col_signs) in enumerate(data[:row + 1]):
                submatrix = pair_moments[np.ix_(row_ids, col_ids)]
                weights = row_signs[:, None] * col_signs[None, :] * np.sign(submatrix)
                totals = defaultdict(int)
                for signed_id, weight in zip(submatrix.ravel().tolist(), weights.ravel().tolist()):
                    if signed_id:
                        totals[abs(signed_id) - 1] += weight
                nonzero = {index: coefficient for index, coefficient in sorted(totals.items()) if coefficient}
                if nonzero:
                    entries.append((row, col, nonzero))
        result.append({'character': character, 'basis': basis, 'entries': entries})
    return result


def build_model(*, split=True):
    """Return exact equality rows and PSD blocks, without calling a solver.

    Every PSD entry is (row, col, {moment_id: int_weight}) in the lower
    triangle. Omitted entries vanish. Bases contain (monomial, int_weight)
    terms. An equality row is a mapping from moment id to (a,b) representing
    a+b*sqrt(10)/5. Set moments[0] to 1 when solving.
    """
    orbits = SignedMoments()
    by_degree = [tuple(combinations_with_replacement(range(NUMBER_OF_VARIABLES), degree))
                 for degree in range(4)]
    cumulative = [sum(by_degree[:degree + 1], ()) for degree in range(4)]
    # Enumerate the entire quotient moment space, including moments absent
    # from any particular PSD basis or equality row.
    for degree in range(1, 5):
        for monomial in combinations_with_replacement(range(NUMBER_OF_VARIABLES), degree):
            orbits.orbit(monomial)
    equalities, witnesses, equations = build_equalities(orbits, cumulative)
    blocks = build_psd_blocks(orbits, cumulative[2], split)
    return {
        'variables': VARIABLES,
        'coefficient_basis': ('1', 'sqrt(10)/5'),
        'moments': tuple(orbits.representatives),
        'equalities': equalities,
        'equality_witnesses': witnesses,
        'base_equations': equations,
        'psd_blocks': blocks,
        'full_moment_matrix_size': len(cumulative[2]),
        'scope': 'complete S8 and conjugation invariant order-two relaxation',
    }
