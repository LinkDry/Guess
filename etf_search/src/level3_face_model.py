"""Exact structural facial restriction for the ETF(4,9) moment model."""

from collections import defaultdict
from fractions import Fraction
from itertools import combinations, combinations_with_replacement
from math import gcd, lcm

from general_4_9_model import (N, NUMBER_OF_VARIABLES, apply_action, base_equations,
                              basis_blocks, edge_action)


FIELD_ZERO = (0, 0)
FIELD_ONE = (1, 0)
BASE_FOCAL_COUNT = {'circle': 2, 'row_real': 1, 'row_imag': 1,
                    'tight_real': 2, 'tight_imag': 2}


def primitive(vector):
    """Remove a common integer factor and canonicalize the overall sign."""
    items = sorted((i, (a, b)) for i, (a, b) in vector.items() if a or b)
    if not items:
        return {}, 1
    divisor = gcd(*(v for _, pair in items for v in pair))
    if items[0][1] < FIELD_ZERO:
        divisor = -divisor
    return {i: (a // divisor, b // divisor) for i, (a, b) in items}, divisor


def multiply_by_five(left, right):
    """Integer coordinates of five times the product, using beta²=2/5."""
    a, b = left
    c, d = right
    return 5*a*c + 2*b*d, 5*(a*d + b*c)


def independent_insert(vector, echelon, *, high_first=False):
    """Insert a row if independent, over the exact quadratic number field."""
    work = dict(vector)
    pivot_selector = max if high_first else min
    while work:
        pivot = pivot_selector(work)
        if pivot not in echelon:
            echelon[pivot] = work
            return True
        previous = echelon[pivot]
        leading = work[pivot]
        reducer = previous[pivot]
        updated = {}
        for column in work.keys() | previous.keys():
            a, b = multiply_by_five(work.get(column, FIELD_ZERO), reducer)
            c, d = multiply_by_five(previous.get(column, FIELD_ZERO), leading)
            updated[column] = (a - c, b - d)
        work, _ = primitive(updated)
    return False


def monomial_projection_maps(blocks):
    """Map each monomial to its coordinate and weight in each projection."""
    maps = []
    for block in blocks:
        mapping = {}
        for coordinate, polynomial in enumerate(block['basis']):
            for word, sign in polynomial:
                term = tuple(word)
                if term in mapping:
                    raise ValueError('Character block has overlapping monomial orbits')
                mapping[term] = coordinate, sign, len(polynomial)
        maps.append(mapping)
    return maps


def null_polynomials(model, monomial_space):
    """Generate permutations of h*m whose polynomial support lies in V."""
    multipliers = [tuple(combinations_with_replacement(range(NUMBER_OF_VARIABLES), d))
                   for d in range(3)]
    for name, focal_count in BASE_FOCAL_COUNT.items():
        source = model['base_equations'][name]
        degree = max(map(len, source))
        for focal in combinations(range(N), focal_count):
            permutation = (*focal, *(i for i in range(N) if i not in focal))
            action = edge_action(permutation)
            polynomial = []
            for term, (a, b) in source.items():
                image, sign = apply_action(tuple(term), action)
                polynomial.append((image, (a*sign, b*sign)))
            for d in range(4 - degree):
                for multiplier in multipliers[d]:
                    shifted = [(tuple(sorted(term + multiplier)), coefficient)
                               for term, coefficient in polynomial]
                    if all(term in monomial_space for term, _ in shifted):
                        yield (name, permutation, multiplier), shifted


def character_projection(polynomial, projection_map):
    """Return a primitive integer field vector and its exact projection scale."""
    coefficients = defaultdict(lambda: [0, 0])
    lengths = {}
    for monomial, (a, b) in polynomial:
        coordinate = projection_map.get(monomial)
        if coordinate is None:
            continue
        index, sign, length = coordinate
        weights = coefficients[index]
        weights[0] += a*sign
        weights[1] += b*sign
        lengths[index] = length
    if not coefficients:
        return {}, Fraction(1)
    scale = lcm(*lengths.values())
    vector, divisor = primitive({index: (a * (scale // lengths[index]), b * (scale // lengths[index]))
                                 for index, (a, b) in coefficients.items()})
    return vector, Fraction(scale, divisor)


def kernel_directions(model, blocks, *, high_first=False):
    maps = monomial_projection_maps(blocks)
    monomial_space = set().union(*(projection.keys() for projection in maps))
    # A monomial can have zero projection in every selected character. For
    # detecting membership in V, include the entire original basis from the stage.
    from level3_4_9_model import cubic_basis
    monomial_space.update(cubic_basis(model['stage']))
    monomial_space.update(term for d in range(3)
                          for term in combinations_with_replacement(range(NUMBER_OF_VARIABLES), d))
    echelon = [{} for _ in blocks]
    directions = [[] for _ in blocks]
    seen = [set() for _ in blocks]
    for source, polynomial in null_polynomials(model, monomial_space):
        for index, projection_map in enumerate(maps):
            vector, scale = character_projection(polynomial, projection_map)
            if not vector:
                continue
            key = tuple(vector.items())
            if key in seen[index]:
                continue
            seen[index].add(key)
            if independent_insert(vector, echelon[index], high_first=high_first):
                directions[index].append({'coordinates': vector, 'origin': source,
                                          'projection_scale': (scale.numerator, scale.denominator)})
    return directions, echelon


def quotient_block(block, pivots):
    size = len(block['basis'])
    keep = [i for i in range(size) if i not in pivots]
    positions = {old: new for new, old in enumerate(keep)}
    return {
        **block,
        'basis': [block['basis'][i] for i in keep],
        'entries': [(positions[i], positions[j], weights) for i, j, weights in block['entries']
                    if i in positions and j in positions],
    }, keep


def append_cross_constraints(block, directions, block_index, rows, witnesses, seen):
    size = len(block['basis'])
    by_column = [[] for _ in range(size)]
    for i, j, weights in block['entries']:
        by_column[j].append((i, weights))
        if i != j:
            by_column[i].append((j, weights))
    start = len(rows)
    for direction_index, direction in enumerate(directions):
        constraints = [defaultdict(lambda: [0, 0]) for _ in range(size)]
        for col, (a, b) in direction['coordinates'].items():
            for row, terms in by_column[col]:
                result = constraints[row]
                for moment, coefficient in terms.items():
                    entry = result[moment]
                    entry[0] += a*coefficient
                    entry[1] += b*coefficient
        for index, constraint in enumerate(constraints):
            primitive_row, divisor = primitive(constraint)
            if not primitive_row:
                continue
            key = tuple(primitive_row.items())
            if key not in seen:
                rows.append(primitive_row)
                witnesses.append(('face_kernel', block_index, direction_index, index, divisor))
                seen.add(key)
    return len(rows) - start


def divide(left, right):
    """Exact division in Q(beta)."""
    a, b = left
    c, d = right
    norm = c*c - Fraction(2, 5)*d*d
    return ((a*c - Fraction(2, 5)*b*d) / norm, (b*c - a*d) / norm)


def orthogonal_basis(block, echelon):
    """Return integer field vectors orthogonal to the kernel in monomial space."""
    norms = [len(polynomial) for polynomial in block['basis']]
    basis = []
    for column in range(len(norms)):
        if column in echelon:
            continue
        # Solve Q^T diag(norms) u=0 with the chosen free coordinate set to one.
        vector = {column: (Fraction(1), Fraction(0))}
        for pivot, equation in sorted(echelon.items(), reverse=True):
            real = imag = Fraction(0)
            for index, coefficient in equation.items():
                if index != pivot and index in vector:
                    a, b = multiply_by_five(vector[index], coefficient)
                    real += a * Fraction(norms[index], 5)
                    imag += b * Fraction(norms[index], 5)
            if real or imag:
                vector[pivot] = divide((-real, -imag), tuple(norms[pivot] * c for c in equation[pivot]))
        denominator = lcm(*(v.denominator for pair in vector.values() for v in pair))
        exact, _ = primitive({i: tuple(int(v*denominator) for v in pair) for i, pair in vector.items()})
        basis.append(exact)
    return basis


def face_dimensions(stage):
    """Count the exact reduction from basis data, without building any PSD entries."""
    from level3_4_9_model import STAGES, cubic_basis

    words = [term for degree in range(3)
             for term in combinations_with_replacement(range(NUMBER_OF_VARIABLES), degree)]
    words.extend(cubic_basis(stage))
    setup = {'stage': stage, 'base_equations': base_equations()}
    blocks = [{'character': character, 'basis': basis}
              for character, basis in basis_blocks(words, split=True,
                                                     fixed_vertices=max(4, STAGES[stage]))]
    _, echelon = kernel_directions(setup, blocks, high_first=True)
    sizes = [len(block['basis']) for block in blocks]
    ranks = list(map(len, echelon))
    reduced = [size - rank for size, rank in zip(sizes, ranks)]
    return {'stage': stage, 'original_sizes': sizes, 'kernel_dimensions': ranks,
            'reduced_sizes': reduced,
            'original_psd_rows': sum(size*(size+1)//2 for size in sizes),
            'reduced_psd_rows': sum(size*(size+1)//2 for size in reduced)}


def reduce_model(model, *, include_orthogonal_basis=False):
    """Append structural kernel equations and impose PSD on a complement."""
    blocks = model['psd_blocks']
    if len(blocks) != 8 or model.get('stage') not in ('support3', 'support4'):
        raise ValueError('Expected support3 or support4 with the eight character blocks')
    if 'face_reduction' in model:
        raise ValueError('The model already has a structural face reduction')
    directions, echelon = kernel_directions(model, blocks)
    rows = list(model['equalities'])
    witnesses = list(model['equality_witnesses'])
    seen = {tuple(sorted(equation.items())) for equation in rows}
    quotients = []
    metadata = []
    for index, (block, block_directions, pivots) in enumerate(zip(blocks, directions, echelon)):
        count = append_cross_constraints(block, block_directions, index, rows, witnesses, seen)
        quotient, keep = quotient_block(block, pivots)
        record = {
            'character': block.get('character'),
            'original_size': len(block['basis']),
            'kernel_dimension': len(pivots),
            'pivot_indices': sorted(pivots),
            'retained_indices': keep,
            'directions': block_directions,
            'added_cross_constraints': count,
        }
        if include_orthogonal_basis:
            record['orthogonal_basis'] = orthogonal_basis(block, pivots)
        metadata.append(record)
        quotients.append(quotient)
    return {
        **model,
        'equalities': rows,
        'equality_witnesses': witnesses,
        'psd_blocks': quotients,
        'scope': model['scope'] + ' with exact structural facial restriction',
        'face_reduction': {
            'schema': 'level3-structural-face-v1',
            'complement': 'principal coordinates disjoint from exact kernel pivots',
            'input_equality_count': len(model['equalities']),
            'added_equality_count': len(rows) - len(model['equalities']),
            'blocks': metadata,
        },
    }
