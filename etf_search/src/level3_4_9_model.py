"""S8-invariant partial order-three ETF(4,9) moment relaxations."""

from collections import defaultdict
from functools import lru_cache
from itertools import combinations, combinations_with_replacement, permutations, product
from math import gcd

from general_4_9_model import (
    EDGE_IDS, N, NUMBER_OF_VARIABLES, VARIABLES, SignedMoments, apply_action,
    basis_blocks, build_equalities, transposition,
)


EDGE_MASKS = tuple((1 << a) | (1 << b) for _, a, b in VARIABLES)
STAGES = {'support3': 3, 'support4': 4}


def odd(word):
    return sum(i % 2 for i in word) % 2


@lru_cache(maxsize=256)
def compressed_edges(mask):
    labels = {vertex: index for index, vertex in enumerate(i for i in range(N) if mask & (1 << i))}
    return tuple(EDGE_IDS[kind, labels[a], labels[b]] if a in labels and b in labels else -1
                 for kind, a, b in VARIABLES)


@lru_cache(maxsize=180_000)
def high_degree_canonical(pattern):
    """Canonical signed representative using invariant vertex-color refinement."""
    size = 1 + max(b for _, _, b in (VARIABLES[i] for i in pattern))
    neighbors = [defaultdict(lambda: [0, 0]) for _ in range(size)]
    for variable in pattern:
        _, a, b = VARIABLES[variable]
        neighbors[a][b][variable % 2] += 1
        neighbors[b][a][variable % 2] += 1
    colors = (0,) * size
    while True:
        signatures = tuple((colors[v], tuple(sorted((colors[w], *counts)
                                                    for w, counts in edges.items())))
                           for v, edges in enumerate(neighbors))
        palette = {signature: i for i, signature in enumerate(sorted(set(signatures)))}
        refined = tuple(palette[signature] for signature in signatures)
        if refined == colors:
            break
        colors = refined
    groups = [[] for _ in range(max(colors) + 1)]
    for vertex, color in enumerate(colors):
        groups[color].append(vertex)
    best = None
    best_sign = 1
    edges = tuple(VARIABLES[i] for i in pattern)
    for ordering in product(*(permutations(group) for group in groups)):
        labels = [0] * size
        for label, vertex in enumerate(v for group in ordering for v in group):
            labels[vertex] = label
        image = []
        sign = 1
        for kind, a, b in edges:
            u, v = labels[a], labels[b]
            if u > v:
                u, v = v, u
                if kind == 'y':
                    sign = -sign
            image.append(EDGE_IDS[kind, u, v])
        candidate = tuple(sorted(image))
        if best is None or candidate < best:
            best, best_sign = candidate, sign
        elif candidate == best and sign != best_sign:
            # Two labelings give the same word with opposite signs.
            return (), 0
    return best, best_sign


class ExtendedMoments(SignedMoments):
    """Use the quartic indexing convention and extend it on demand."""

    def __init__(self):
        super().__init__()
        self._higher = {}

    @classmethod
    def from_representatives(cls, representatives):
        """Reconstruct signed indices from a saved model's moment list."""
        indexing = cls()
        for index, representative in enumerate(representatives, start=1):
            word = tuple(representative)
            if indexing.orbit(word) != index or indexing.representatives[index - 1] != word:
                raise ValueError(f'Invalid moment representative at index {index - 1}')
        return indexing

    def orbit(self, word):
        if len(word) <= 4:
            return super().orbit(word)
        if len(word) > 6:
            raise ValueError('Moment degree exceeds six')
        if odd(word):
            return 0
        mask = 0
        for variable in word:
            mask |= EDGE_MASKS[variable]
        labels = compressed_edges(mask)
        compressed = tuple(sorted(labels[i] for i in word))
        return self._canonical_id(compressed)

    @lru_cache(maxsize=180_000)
    def _canonical_id(self, compressed):
        representative, sign = high_degree_canonical(compressed)
        if not sign:
            return 0
        index = self._higher.get(representative)
        if index is None:
            self.representatives.append(representative)
            index = self._higher[representative] = len(self.representatives)
        return index * sign


def cubic_basis(stage):
    cutoff = STAGES[stage]
    return [word for word in combinations_with_replacement(range(NUMBER_OF_VARIABLES), 3)
            if (EDGE_MASKS[word[0]] | EDGE_MASKS[word[1]] | EDGE_MASKS[word[2]]).bit_count() <= cutoff]


def reduced_psd(orbits, monomials):
    """Form PSD blocks using one representative per row basis orbit."""
    result = []
    for char, basis in basis_blocks(monomials, split=True):
        entries = []
        for row, polynomial in enumerate(basis):
            representative, sign = polynomial[0]
            row_weight = len(polynomial) * sign
            for col, other in enumerate(basis[:row + 1]):
                totals = defaultdict(int)
                for word, weight in other:
                    index = orbits.orbit(tuple(sorted(representative + word)))
                    if index:
                        totals[abs(index) - 1] += weight if index > 0 else -weight
                coefficients = {index: row_weight * coefficient for index, coefficient in sorted(totals.items())
                                if coefficient}
                if coefficients:
                    entries.append((row, col, coefficients))
        result.append({'character': char, 'basis': basis, 'entries': entries})
    return result


def add_row(orbits, equations, witnesses, seen, name, equation, multiplier):
    totals = defaultdict(lambda: [0, 0])
    for word, (a, b) in equation.items():
        signed_id = orbits.orbit(tuple(sorted(word + multiplier)))
        if signed_id:
            sign = 1 if signed_id > 0 else -1
            pair = totals[abs(signed_id) - 1]
            pair[0] += sign * a
            pair[1] += sign * b
    coefficients = sorted((i, a, b) for i, (a, b) in totals.items() if a or b)
    if not coefficients:
        return
    divisor = gcd(*(coefficient for _, a, b in coefficients for coefficient in (a, b)))
    if (coefficients[0][1], coefficients[0][2]) < (0, 0):
        divisor = -divisor
    normalized = tuple((i, a // divisor, b // divisor) for i, a, b in coefficients)
    if normalized not in seen:
        equations.append({i: (a, b) for i, a, b in normalized})
        witnesses.append((name, multiplier, divisor))
        seen.add(normalized)


def multiplier_orbits(equation, max_degree):
    """One word per orbit under transpositions preserving equation up to sign."""
    generators = []
    for a, b in combinations(range(N), 2):
        action = transposition(a, b)
        overall = None
        for word, coefficient in equation.items():
            image, sign = apply_action(word, action)
            mapped = equation.get(image)
            positive = tuple(sign * c for c in coefficient)
            if mapped == positive:
                current = 1
            elif mapped == tuple(-c for c in positive):
                current = -1
            else:
                break
            if overall is not None and overall != current:
                break
            overall = current
        else:
            generators.append(action)
    parities = {odd(word) for word in equation}
    for degree in range(max_degree + 1):
        visited = set()
        for word in combinations_with_replacement(range(NUMBER_OF_VARIABLES), degree):
            if word in visited or odd(word) not in parities:
                continue
            pending = [word]
            visited.add(word)
            for term in pending:
                for action in generators:
                    image, _ = apply_action(term, action)
                    if image not in visited:
                        visited.add(image)
                        pending.append(image)
            yield word


def build_model(*, stage='support4', extra_equations=None):
    """Build a targeted degree-six model in the Round 4 solver format.

    Existing quartic moment IDs are preserved. Each optional extra equation
    maps a sorted word to exact integer coordinates in the basis (1, beta).
    Gram-five minors use linear multipliers and Schur-four minors use
    multipliers through degree two, with full support modulo relabeling.
    """
    if stage not in STAGES:
        raise ValueError(f'Unknown model stage: {stage}')
    orbits = ExtendedMoments()
    by_degree = [tuple(combinations_with_replacement(range(NUMBER_OF_VARIABLES), degree))
                 for degree in range(4)]
    for degree in range(1, 5):
        for word in combinations_with_replacement(range(NUMBER_OF_VARIABLES), degree):
            orbits.orbit(word)
    cumulative = [sum(by_degree[:degree + 1], ()) for degree in range(4)]
    rows, witnesses, equations = build_equalities(orbits, cumulative)
    seen = {tuple((i, *coefficient) for i, coefficient in row.items()) for row in rows}
    cubics = cubic_basis(stage)
    blocks = reduced_psd(orbits, (*cumulative[2], *cubics))
    psd_moment_count = len(orbits.representatives)

    cutoff = STAGES[stage]
    local_variables = [i for i, (_, _, b) in enumerate(VARIABLES) if b < cutoff]
    for name, equation in equations.items():
        degree = max(map(len, equation))
        for multiplier_degree in range(5 - degree, 7 - degree):
            for word in combinations_with_replacement(local_variables, multiplier_degree):
                if odd(word) == int(name.endswith('_imag')):
                    add_row(orbits, rows, witnesses, seen, name, equation, word)
    core_equalities = len(rows)
    for name, polynomial in (extra_equations or {}).items():
        if name in equations or not polynomial:
            raise ValueError(f'Expected a nonempty new equation: {name}')
        equation = {tuple(sorted(word)): tuple(coefficient) for word, coefficient in polynomial.items()
                    if any(coefficient)}
        if not equation:
            raise ValueError(f'Zero equation: {name}')
        degree = max(map(len, equation))
        if degree > 6:
            raise ValueError(f'Equation degree exceeds six: {name}')
        budget = 6 - degree
        if name.startswith('gram5_'):
            budget = min(budget, 1)
        elif name.startswith('schur4_'):
            budget = min(budget, 2)
        elif budget > 2:
            raise ValueError(f'Extras must use multipliers through degree two: {name}')
        equations[name] = equation
        for word in multiplier_orbits(equation, budget):
            add_row(orbits, rows, witnesses, seen, name, equation, word)
    # The memoized lookup is only needed during construction. Avoid retaining
    # the orbit object in a method cache after returning the serialized model.
    orbits._canonical_id.cache_clear()
    high_degree_canonical.cache_clear()
    return {
        'variables': VARIABLES,
        'coefficient_basis': ('1', 'sqrt(10)/5'),
        'moments': tuple(orbits.representatives),
        'equalities': rows,
        'equality_witnesses': witnesses,
        'base_equations': equations,
        'psd_blocks': blocks,
        'full_moment_matrix_size': len(cumulative[2]) + len(cubics),
        'scope': f'S8 and conjugation invariant partial order-three relaxation ({stage})',
        'maximum_monomial_degree': 6,
        'stage': stage,
        'quartic_moment_count': 107,
        'core_equality_count': core_equalities,
        'extra_equality_count': len(rows) - core_equalities,
        'psd_moment_count': psd_moment_count,
    }
