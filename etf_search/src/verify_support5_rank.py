"""Audit Round 8 support-five equations, PSD entries, and exact certificates.

The source chart is linked to the hash-matched Round 6 face audit. New rows
and PSD stencils are reconstructed with independently checked signed orbit
averaging. Over Q, nonsingular minors certify lower rank bounds and full
exact chart substitution and PSD column identities certify upper bounds.
"""

import argparse
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
from itertools import permutations
import json
from math import gcd, lcm
from pathlib import Path

from flint import nmod_mat
import numpy as np
from scipy.sparse import load_npz, csr_matrix
from sympy import nextprime

from verify_sdp_anchored import VARIABLES, EDGE_INDEX
from verify_sdp_face import shifted_origin
from verify_sdp_level3 import (BASIS, MomentManifest, canonical_psd_bases, equation_terms, row_terms,
                               selected_monomials)
from verify_sdp_factor_cert import monomial, rational
from verify_sdp_lmi import (AffineChart, certify_matrix, model_sha256, moment_bounds, read_json,
                           read_primal_moments, scaled_moments, selected_equalities)


def primitive_parity_row(row, parity):
    """Apply m_i=beta**parity[i] u_i and remove the common field factor."""
    scaled = {}
    tags = set()
    for column, (real, radical) in row.items():
        odd = parity[column]
        if real:
            tags.add(odd)
            scaled[column] = 5 * real
        if radical:
            tags.add(1 - odd)
            scaled[column] = (2 if odd else 5) * radical
    if len(tags) > 1:
        raise ValueError('row has mixed beta parity')
    if not scaled:
        return {}
    denominator = lcm(*(Fraction(value).denominator for value in scaled.values()))
    integers = {column: int(value * denominator) for column, value in scaled.items()}
    divisor = gcd(*integers.values())
    if integers[min(integers)] < 0:
        divisor = -divisor
    return {column: value // divisor for column, value in integers.items()}


def verify_complement_minor(directions, pivots):
    """Specialize beta to a square root of 2/5 and certify the pivot minor."""
    if not pivots:
        return
    # Each pair is (prime, image of beta) with 5 * image**2 = 2 mod prime.
    for prime, beta in ((1000000009, 184714173), (1000003, 78843), (1000037, 375517)):
        if (5 * beta**2 - 2) % prime:
            continue
        n = len(pivots)
        matrix = nmod_mat(n, n, prime)
        slot = {column: i for i, column in enumerate(pivots)}
        for i, direction in enumerate(directions):
            retained = {column: (Fraction(a), Fraction(b)) for column, (a, b) in direction.items()
                        if column in slot}
            denominator = lcm(*(c.denominator for coefficient in retained.values() for c in coefficient))
            for column, (a, b) in retained.items():
                matrix[i, slot[column]] = int(denominator * (a + beta * b)) % prime
        if matrix.rank() == n:
            return
    raise ValueError('kernel pivot minor has no full modular rank certificate')


FOCAL = {'circle': 2, 'row_real': 1, 'row_imag': 1, 'tight_real': 2, 'tight_imag': 2}


def parity(word):
    return sum(i % 2 for i in word) % 2


def raw_cross(manifest, name, multiplier, test, allowed):
    """Validate L(h*m*p)=0 as a shifted canonical polynomial identity."""
    if name not in FOCAL or test not in allowed:
        raise ValueError('cross witness uses an unsupported equation or test word')
    shifted = {tuple(sorted(term + multiplier)): value for term, value in equation_terms(name).items()}
    if any(term not in allowed for term in shifted):
        raise ValueError('kernel polynomial leaves the support-five monomial space')
    row = defaultdict(lambda: [0, 0])
    for term, coefficients in shifted.items():
        index, sign = manifest.lookup(tuple(sorted(term + test)))
        if sign:
            for component, value in enumerate(coefficients):
                row[index][component] += sign * value
    return {index: tuple(value) for index, value in row.items() if any(value)}


@lru_cache(maxsize=None)
def edge_map(perm):
    return tuple(EDGE_INDEX[min(perm[i], perm[j]), max(perm[i], perm[j])] + (kind == 'y')
                 for kind, i, j in VARIABLES)


def tail_representatives(words, fixed):
    """Enumerate orbits directly with every tail permutation, ignoring word signs."""
    maps = [edge_map(tuple(range(fixed)) + tail) for tail in permutations(range(fixed, 8))]
    seen = set()
    for word in words:
        if word in seen:
            continue
        yield word
        seen.update(tuple(sorted(mapping[var] for var in word)) for mapping in maps)


def independently_generate_crosses(manifest, words, degree_parity):
    """Generate all raw cross rows modulo focal and multiplier stabilizers."""
    allowed = set(words)

    @lru_cache(maxsize=8)
    def tests(fixed):
        return tuple(tail_representatives(words, fixed))

    for name, focal in FOCAL.items():
        equation = equation_terms(name)
        max_multiplier_degree = 3 - max(map(len, equation))
        equation_parity = parity(next(iter(equation)))
        for multiplier in tests(focal):
            if len(multiplier) > max_multiplier_degree:
                # Representatives are enumerated by degree.
                break
            used = sorted(set(range(focal)) | {v for x in multiplier for v in VARIABLES[x][1:]})
            complement = [v for v in range(8) if v not in used]
            relabel = edge_map(tuple((used + complement).index(v) for v in range(8)))
            multiplier = tuple(sorted(relabel[x] for x in multiplier))
            if any(tuple(sorted(term + multiplier)) not in allowed for term in equation):
                continue
            required_parity = equation_parity ^ parity(multiplier)
            for test in tests(len(used)):
                if parity(test) == required_parity:
                    yield primitive_parity_row(raw_cross(manifest, name, multiplier, test, allowed), degree_parity)


def csr_row(matrix, index):
    lo, hi = matrix.indptr[index:index+2]
    return {int(i): int(a) for i, a in zip(matrix.indices[lo:hi], matrix.data[lo:hi], strict=True)}


def read_integer_map(path, width, height):
    matrix = load_npz(path).tocsr()
    if (matrix.shape[0] != height or matrix.shape[1] > width or matrix.data.dtype.kind != 'i'
            or not matrix.has_canonical_format or np.any(matrix.data == 0)):
        raise ValueError(f'invalid integer CSR stencil {path.name}')
    matrix.check_format(full_check=True)
    matrix.resize((height, width))
    return matrix.astype(np.int64, copy=False)


def audit_cross_rows(root, meta, manifest, degree_parity):
    witnesses = read_json(root/'cross_provenance.json.gz')
    if len(witnesses) != meta['unique_cross_rows']:
        raise ValueError('cross provenance count mismatch')
    words = selected_monomials('support5')
    allowed = set(words)
    matrix = read_integer_map(root/'cross_rows.npz', len(degree_parity), len(witnesses))
    declared = set()
    for i, (name, multiplier, test) in enumerate(witnesses):
        multiplier = monomial(multiplier, len(VARIABLES))
        test = monomial(test, len(VARIABLES))
        expected = primitive_parity_row(raw_cross(manifest, name, multiplier, test, allowed), degree_parity)
        if not expected or expected != csr_row(matrix, i):
            raise ValueError(f'cross row {i} disagrees with its witness')
        declared.add(tuple(sorted(expected.items())))
    regenerated = set()
    proposals = 0
    for row in independently_generate_crosses(manifest, words, degree_parity):
        proposals += 1
        if row:
            regenerated.add(tuple(sorted(row.items())))
    if declared != regenerated or len(declared) != len(witnesses) or proposals != meta['cross_row_proposals']:
        raise ValueError('declared cross rows differ from independent exhaustive enumeration')
    return matrix, witnesses, {'unique_cross_rows': len(declared), 'cross_row_proposals': proposals}


def check_projection(direction, origin, scale, projections, allowed):
    name, perm, multiplier = origin
    _, _, _, image = shifted_origin((name, perm, list(multiplier)), allowed)
    exact = defaultdict(lambda: [0, 0])
    for word, coefficients in image.items():
        if word in projections:
            index, sign, length = projections[word]
            for component, value in enumerate(coefficients):
                exact[index][component] += scale * sign * value / length
    expected = {i: tuple(v) for i, v in exact.items() if any(v)}
    if not scale or not direction or expected != direction:
        raise ValueError('structural kernel projection disagrees with its polynomial origin')


def complement_proposal(bases):
    """Recover producer proposals for older artifacts that omit retained indices.

    Each proposed origin, its projected coordinates, and the full pivot minor
    are checked by the caller using the independent canonical bases.
    """
    from level3_face_model import kernel_directions

    blocks = [{'basis': [tuple(sorted(poly.items())) for poly in basis]} for basis in bases.values()]
    equations = {name: equation_terms(name) for name in FOCAL}
    integer_equations = {name: {word: tuple(int(c) for c in coefficient)
                                for word, coefficient in equation.items()}
                         for name, equation in equations.items()}
    directions, pivots = kernel_directions({'stage': 'support5', 'base_equations': integer_equations},
                                           blocks, high_first=True)
    return [(sorted(pivot), block_directions) for pivot, block_directions in zip(pivots, directions, strict=True)]


def audit_psd_maps(root, meta, manifest):
    bases = canonical_psd_bases('support5')
    allowed = set(selected_monomials('support5'))
    proposal = complement_proposal(bases)
    maps, block_parities = [], []
    for block, ((_, basis), (pivots, directions)) in enumerate(zip(bases.items(), proposal, strict=True)):
        projections = {word: (i, sign, len(poly)) for i, poly in enumerate(basis)
                       for word, sign in poly.items()}
        for witness in directions:
            check_projection(witness['coordinates'], witness['origin'], Fraction(*witness['projection_scale']),
                             projections, allowed)
        verify_complement_minor([w['coordinates'] for w in directions], pivots)
        keep = sorted(set(range(len(basis))) - set(pivots))
        size = meta['block_sizes'][block]
        if size != len(keep) or len(pivots) != meta['kernel_dimensions'][block]:
            raise ValueError(f'wrong quotient or structural kernel size in block {block}')
        packed = size * (size + 1) // 2
        matrix = read_integer_map(root/f'psd_map_{block}.npz', len(manifest.representatives), packed)
        congruence = np.load(root/f'congruence_{block}.npy', allow_pickle=False)
        row = 0
        expected_scale = []
        for col, j in enumerate(keep):
            for i in keep[col:]:
                terms = csr_row(matrix, row)
                expected_parity = (len(next(iter(basis[i]))) + len(next(iter(basis[j])))) % 2
                if any(len(manifest.representatives[k]) % 2 != expected_parity for k in terms):
                    raise ValueError(f'mixed degree parity at block {block}, packed row {row}')
                if terms != manifest.character_product(basis[i], basis[j]):
                    raise ValueError(f'PSD stencil disagrees at block {block}, packed row {row}')
                expected_scale.append(1/np.sqrt(len(basis[i]) * len(basis[j])))
                row += 1
        if congruence.shape != (packed,) or not np.allclose(congruence, expected_scale, rtol=1e-14, atol=0):
            raise ValueError(f'incorrect congruence in block {block}')
        maps.append(matrix)
        block_parities.append([len(next(iter(basis[i]))) % 2 for i in keep])
    return maps, block_parities


def parse_chart(document, parity_flags):
    free = document['free_columns']
    if document['parity'] != parity_flags or len(document['rows']) != len(parity_flags):
        raise ValueError('chart coordinates or parity disagree with moment order')
    if (len(set(free)) != len(free) or
            any(type(i) is not int or not 0 < i < len(parity_flags) for i in free)):
        raise ValueError('invalid chart free columns')
    rows = []
    for den, terms in document['rows']:
        if type(den) is not int or den <= 0:
            raise ValueError('invalid chart denominator')
        parsed = {}
        for k, v in terms:
            if (type(k) is not int or not 0 <= k <= len(free) or k in parsed
                    or type(v) is not int or not v):
                raise ValueError('invalid chart coefficient')
            parsed[k] = Fraction(v, den)
        rows.append(parsed)
    if rows[0] != {0: 1} or any(rows[col] != {i: 1} for i, col in enumerate(free, 1)):
        raise ValueError('chart fails normalization or changes a free coordinate')
    return rows, free


def bounded_exact_zero(matrix, columns):
    """Clear each column denominator and prove every sparse dot product vanishes."""
    largest = max((abs(int(matrix.data.min())), abs(int(matrix.data.max()))), default=0) if matrix.nnz else 0
    row_bound = largest * int(np.diff(matrix.indptr).max(initial=0))
    if row_bound >= (1 << 62):
        raise ValueError('stencil coefficients exceed safe sparse proof bound')
    cleared = []
    max_entry = 0
    for column in columns:
        denominator = lcm(*(Fraction(v).denominator for v in column))
        integer = [int(v * denominator) for v in column]
        max_entry = max(max_entry, max(map(abs, integer), default=0))
        cleared.append(integer)
    bound = row_bound * max_entry
    if not bound:
        return
    primes = [None]
    if bound >= (1 << 62):
        primes = []
        product, prime = 1, 1_000_000
        while product <= bound:
            prime = int(nextprime(prime))
            if row_bound * prime >= 1 << 62:
                raise ValueError('CRT dot products exceed safe sparse proof bound')
            primes.append(prime)
            product *= prime
    for prime in primes:
        for start in range(0, len(cleared), 8):
            batch = cleared[start:start+8]
            if prime:
                batch = [[v % prime for v in column] for column in batch]
            product = matrix @ np.ascontiguousarray(np.asarray(batch, dtype=np.int64).T)
            invalid = product % prime if prime else product
            if np.any(invalid):
                i, j = np.argwhere(invalid != 0)[0]
                raise ValueError(f'exact sparse identity fails at row {i}, column {start+j}')


def chart_column(rows, slot):
    return [row.get(slot, 0) for row in rows]


def modular_minor(rows):
    n = len(rows)
    if any(len(row) != n for row in rows):
        raise ValueError('expected square rank minor')
    integers = []
    for row in rows:
        den = lcm(*(Fraction(x).denominator for x in row))
        integers.extend(int(x * den) for x in row)
    for prime in (65537, 1_000_000_007, 1_000_000_009):
        if nmod_mat(n, n, [x % prime for x in integers], prime).rank() == n:
            return prime
    raise ValueError('rank minor was singular under all tested reductions')


def row_on_slots(matrix, i, chart, slots):
    row = csr_row(matrix, i)
    return [sum(weight * chart[j].get(k, 0) for j, weight in row.items()) for k in slots]


def chart_identities(chart, old, cross, old_matrix):
    """Verify E x=0 and an independent additional pivot minor inside the old chart."""
    old_rows, old_free = old
    rows, free, document = chart
    prior_free = old_free + list(range(len(old_rows), len(rows)))
    eliminated = document['eliminated_moment_coordinates']
    selected = document['selected_cross_rows']
    if (len(set(free + eliminated)) != len(prior_free) or sorted(free + eliminated) != sorted(prior_free)
            or len(selected) != len(eliminated) or len(set(selected)) != len(selected)
            or any(type(i) is not int or not 0 <= i < cross.shape[0] for i in selected)
            or document['additional_rank'] != len(selected)):
        raise ValueError('invalid additional equality pivot selection')
    slots = [prior_free.index(i) + 1 for i in eliminated]
    extended_old = old_rows + [{len(old_free)+i+1: 1} for i in range(len(rows)-len(old_rows))]
    prime = modular_minor([row_on_slots(cross, i, extended_old, slots) for i in selected])
    columns = [chart_column(rows, i) for i in range(len(free)+1)]
    bounded_exact_zero(cross, columns)
    bounded_exact_zero(old_matrix, [column[:len(old_rows)] for column in columns])
    rank = len(rows) - len(free)
    return {'additional_rank': len(selected), 'additional_minor_prime': prime,
            'exact_rank_including_normalization': rank, 'free_dimension': len(free)}


def audit_image_rank(document, maps, chart_rows, dimension):
    retained = document['retained_direction_columns']
    witnesses = document['witness_packed_rows']
    relations = document['discarded_as_retained_combinations']
    discarded = sorted(set(range(dimension)) - set(retained))
    if (any(type(i) is not int or not 0 <= i < dimension for i in retained)
            or len(set(retained)) != len(retained) or document['rank'] != len(retained)
            or len(witnesses) != len(retained) or sorted(discarded + retained) != list(range(dimension))
            or sorted(i for i, _ in relations) != discarded):
        raise ValueError('invalid PSD image pivot or discarded direction indices')
    minor = []
    for block, packed in witnesses:
        if not 0 <= block < len(maps) or not 0 <= packed < maps[block].shape[0]:
            raise ValueError('PSD minor witness outside the pencil')
        minor.append(row_on_slots(maps[block], packed, chart_rows, [j+1 for j in retained]))
    prime = modular_minor(minor)
    transformed = []
    for direction, terms in relations:
        weights = {}
        for index, weight in terms:
            if index not in retained or index in weights:
                raise ValueError('invalid discarded PSD direction combination')
            weights[index] = rational(weight)
        transformed.append([row.get(direction+1, 0) - sum(value * row.get(index+1, 0)
                                                          for index, value in weights.items())
                            for row in chart_rows])
    for matrix in maps:
        bounded_exact_zero(matrix, transformed)
    return {'exact_rank': len(retained), 'minor_prime': prime,
            'exact_discarded_direction_relations': len(transformed)}


def old_integer_rows(source, flags):
    """Read the prior face equations once, preserving source row numbering."""
    from array import array

    columns, values, indptr = array('i'), array('q'), array('i', [0])
    for supplied in source['equalities']:
        parsed = {int(i): ((c, 0) if type(c) is int else c) for i, c in supplied.items()}
        row = primitive_parity_row(parsed, flags)
        for index, coefficient in sorted(row.items()):
            columns.append(index)
            values.append(coefficient)
        indptr.append(len(columns))
    return csr_matrix((np.frombuffer(values, dtype=np.int64), np.frombuffer(columns, dtype=np.int32),
                       np.frombuffer(indptr, dtype=np.int32)), shape=(len(indptr)-1, len(flags)))


def audit_common_kernel(document, maps, block_parities, chart_rows, width):
    """Verify the entire affine pencil on D^-1 q for rational q and D=diag(beta^parity)."""
    if document['schema'] != 'support5-common-kernel-v1':
        raise ValueError('unsupported common-kernel schema')
    chart_columns = [chart_column(chart_rows, i) for i in range(width)]
    kernels, dimensions = [], []
    for record, matrix, flags in zip(document['blocks'], maps, block_parities, strict=True):
        size = len(flags)
        if record['size'] != size or record['basis_parity'] != flags:
            raise ValueError('common kernel disagrees with canonical retained basis')
        pivots = record['pivot_indices']
        if (len(set(pivots)) != len(pivots) or len(pivots) != len(record['vectors'])
                or any(type(i) is not int or not 0 <= i < size for i in pivots)):
            raise ValueError('invalid common-kernel pivots')
        vectors = []
        for terms in record['vectors']:
            vector = {}
            for coordinate, coefficient in terms:
                if type(coordinate) is not int or not 0 <= coordinate < size or coordinate in vector:
                    raise ValueError('invalid common-kernel coordinate')
                vector[coordinate] = rational(coefficient)
            vectors.append(vector)
        modular_minor([[vector.get(i, 0) for i in pivots] for vector in vectors])
        # In the u-moment chart, 2 D^-1 M D^-1 has integer stencils. The factor
        # is 5 for an odd-odd entry and 2 for every other entry.
        for vector in vectors:
            column_scale = lcm(*(v.denominator for v in vector.values()))
            entries = [defaultdict(int) for _ in range(size)]
            direction = {i: int(v * column_scale) for i, v in vector.items() if v}
            for src, weight in direction.items():
                for dest in range(size):
                    i, j = max(src, dest), min(src, dest)
                    packed = j*size - j*(j-1)//2 + i-j
                    factor = 5 if flags[src] and flags[dest] else 2
                    for k, v in csr_row(matrix, packed).items():
                        entries[dest][k] += weight * factor * v
            values, indices, indptr = [], [], [0]
            for entry in entries:
                divisor = gcd(*entry.values()) or 1
                for k, v in sorted(entry.items()):
                    if v:
                        if abs(v // divisor) >= 1 << 62:
                            raise ValueError('common-kernel contraction exceeds sparse integer range')
                        indices.append(k)
                        values.append(v // divisor)
                indptr.append(len(values))
            contraction = csr_matrix((np.asarray(values, dtype=np.int64), indices, indptr),
                                     shape=(size, len(chart_rows)))
            bounded_exact_zero(contraction, chart_columns)
        # Convert rational kernel of D^-1 M D^-1 into a field kernel of M.
        directions = [{str(i): [0, str(Fraction(5, 2)*v)] if flags[i] else [str(v), 0]
                       for i, v in vector.items() if v} for vector in vectors]
        kernels.append({'pivot_indices': pivots, 'directions': directions})
        dimensions.append(len(vectors))
    return kernels, dimensions


def restrict_primal_block(real, radical, record):
    """Check M q = 0 and return the principal complement of independent q."""
    size = len(real)
    pivots = record['pivot_indices']
    directions = [row_terms(direction, size) for direction in record['directions']]
    if (len(pivots) != len(directions) or len(set(pivots)) != len(pivots)
            or any(type(i) is not int or not 0 <= i < size for i in pivots)):
        raise ValueError('invalid primal kernel pivot indices')
    verify_complement_minor(directions, pivots)
    for index, direction in enumerate(directions):
        denominator = lcm(*(value.denominator for coefficient in direction.values() for value in coefficient))
        integer = [(j, int(a * denominator), int(b * denominator))
                   for j, (a, b) in direction.items()]
        for i, (a_row, b_row) in enumerate(zip(real, radical, strict=True)):
            a = sum(5 * a_row[j] * x + 2 * b_row[j] * y for j, x, y in integer)
            b = sum(a_row[j] * y + b_row[j] * x for j, x, y in integer)
            if a or b:
                raise ValueError(f'kernel annihilation fails for direction {index} at matrix row {i}')
    keep = sorted(set(range(size)) - set(pivots))
    return [[real[i][j] for j in keep] for i in keep], [[radical[i][j] for j in keep] for i in keep]


def exact_primal(moments, maps, sizes, flags, old_matrix, crosses, kernels=None):
    """Check original moments against all exact rows and certify retained PSD."""
    if moments[0] != (1, 0):
        raise ValueError('constant primal moment must be 1')
    u = [(b, Fraction(5, 2)*a) if odd else (a, b) for (a, b), odd in zip(moments, flags, strict=True)]
    coordinates = list(zip(*u, strict=True))
    bounded_exact_zero(crosses, coordinates)
    bounded_exact_zero(old_matrix, [column[:old_matrix.shape[1]] for column in coordinates])
    scaled, denominator = scaled_moments(moments)
    reports = []
    if kernels is None:
        kernels = [None] * len(maps)
    for matrix, size, kernel in zip(maps, sizes, kernels, strict=True):
        real, radical = [[[0] * size for _ in range(size)] for _ in range(2)]
        for packed, (i, j) in enumerate((i, j) for j in range(size) for i in range(j, size)):
            entry = csr_row(matrix, packed)
            real[i][j] = real[j][i] = sum(coef * scaled[k][0] for k, coef in entry.items())
            radical[i][j] = radical[j][i] = sum(coef * scaled[k][1] for k, coef in entry.items())
        if kernel is not None:
            real, radical = restrict_primal_block(real, radical, kernel)
        psd = (certify_matrix(real, radical, denominator) if real else
               {'holds': True, 'rank': 0, 'size': 0, 'method': 'exact_full_kernel'})
        reports.append({**psd, 'original_block_order': size, 'principal_order': len(real),
                        'verified_kernel_dimension': size - len(real)})
    return {'passed': all(item['holds'] for item in reports), 'blocks': reports,
            'inherited_equations_checked': old_matrix.shape[0], 'cross_equations_checked': crosses.shape[0]}


def exact_dual(document, chart, maps, sizes, words, flags, conditional):
    """Bound the nonnegative trace pairing on the exact affine chart."""
    from flint import fmpz_mat
    from verify_sdp_lmi import bounded_upper

    denominator = document['gram_factor_denominator']
    if type(denominator) is not int or denominator <= 0:
        raise ValueError('invalid dual Gram factor denominator')
    coefficients = [0] * len(words)
    for matrix, size, factor in zip(maps, sizes, document['factors'], strict=True):
        if (len(factor) != size or any(len(row) != len(factor[0]) for row in factor)
                or any(type(v) is not int for row in factor for v in row)):
            raise ValueError('invalid dual Gram factor')
        if not factor[0]:
            continue
        f = fmpz_mat(factor)
        gram = f * f.transpose()
        for packed, (i, j) in enumerate((i, j) for j in range(size) for i in range(j, size)):
            scale = int(gram[i, j]) * (1 if i == j else 2)
            if scale:
                for k, value in csr_row(matrix, packed).items():
                    coefficients[k] += value * scale
    rows, free, _ = chart
    pivots = sorted(set(range(len(words))) - set(free))
    objective = [[Fraction(0)] * 2 for _ in range(len(free)+1)]
    encoded_rows = []
    for moment, row in enumerate(rows):
        den = lcm(*(v.denominator for v in row.values()))
        encoded_rows.append([den, [[k, int(v*den)] for k, v in row.items()]])
        for k, value in row.items():
            objective[k][flags[moment]] += value * coefficients[moment]
    chart_view = AffineChart({'parity': flags, 'pivot_columns': pivots, 'free_columns': free,
                              'selected_rows': list(range(len(pivots))), 'rows': encoded_rows}, flags)
    weights = {int(i): rational(w) for i, w in document.get('box_weights', {}).items()}
    upper = min(bounded_upper(chart_view, objective, moment_bounds(words)),
                bounded_upper(chart_view, objective, moment_bounds(words), weights)) / denominator**2
    return {'passed': upper < 0, 'trace_upper': str(upper), 'trace_upper_approx': float(upper),
            'conditional': conditional, 'etf_nonexistence_verified': upper < 0 and not conditional}


def verify_artifacts(root, prior_face, prior_chart, inherited_audit, options):
    meta = read_json(root/'model.json')
    if meta['stage'] != 'support5' or len(meta['block_sizes']) != 8 or len(meta['kernel_dimensions']) != 8:
        raise ValueError('invalid Round 8 model stage or number of PSD blocks')
    flags = [len(w) % 2 for w in meta['moments']]
    manifest = MomentManifest({**meta, 'variables': list(map(list, VARIABLES)), 'coefficient_basis': BASIS})
    cross, _, cross_report = audit_cross_rows(root, meta, manifest, flags)
    maps, block_parities = audit_psd_maps(root, meta, manifest)
    model_hash = model_sha256(root/'model.json')
    face_hash = model_sha256(prior_face)
    old_doc = read_json(prior_chart)
    new_doc = read_json(root/'affine_chart.json.gz')
    if (old_doc['model_sha256'] != face_hash or new_doc['model_sha256'] != model_hash
            or new_doc['source_chart_sha256'] != model_sha256(prior_chart)):
        raise ValueError('model or source chart hashes disagree')
    if (inherited_audit.get('certificate_sha256') != face_hash
            or not inherited_audit.get('equivalence_to_original_relaxation_verified')):
        raise ValueError('hash-matched exact Round 6 face provenance is required')
    source = read_json(prior_face)
    count = meta['old_moments']
    if source['stage'] != 'support4' or source['moments'] != meta['moments'][:count]:
        raise ValueError('incorrect inherited moment prefix')
    old = parse_chart(old_doc, flags[:count])
    inherited = AffineChart(old_doc, flags[:count])
    inherited_rank = inherited.verify_coverage(selected_equalities(inherited, source))
    old_matrix = old_integer_rows(source, flags[:count])
    conditional = any(isinstance(w, dict) and w.get('type') == 'fixed_moment'
                      for w in source['equality_witnesses'])
    chart = (*parse_chart(new_doc, flags), new_doc)
    equality = chart_identities(chart, old, cross, old_matrix)
    if equality['exact_rank_including_normalization'] != inherited_rank + equality['additional_rank']:
        raise ValueError('prior and additional rank certificates disagree with chart dimension')
    rank_doc = read_json(root/'pencil_rank_certificate.json')
    image = audit_image_rank(rank_doc, maps, chart[0], len(chart[1]))
    report = {'passed': True, 'stage': 'support5', 'model_sha256': model_hash,
              'inherited_face_sha256': face_hash, 'conditional': conditional,
              'moment_count': len(flags), 'inherited_equality_rank': inherited_rank,
              'inherited_homogeneous_rows_checked': old_matrix.shape[0],
              'equations': {**cross_report, **equality}, 'psd_image': image,
              'psd_stencils_checked': sum(matrix.shape[0] for matrix in maps),
              'psd_block_sizes': meta['block_sizes'],
              'scope': ('Inherited support-four face equalities, plus support-five raw kernel cross rows '
                        'and principal-coordinate PSD blocks in the eight selected characters. '
                        'Every anchored ETF satisfies these constraints, subject to any fixed moment assumptions. '
                        'Raw cross rows imply the projected kernel cross rows, so PSD lifts by congruence '
                        'to the original eight support-five character blocks.')}
    primal_kernels = None
    if options.common_kernel:
        kernel_source = read_json(options.common_kernel)
        primal_kernels, common_dimensions = audit_common_kernel(kernel_source, maps, block_parities, chart[0],
                                                               len(chart[1])+1)
        report['common_kernel'] = {'dimensions': common_dimensions, 'affine_coefficients_checked': len(chart[1])+1}
    if options.primal:
        candidate = read_json(options.primal)
        if candidate.get('model_sha256', model_hash) != model_hash:
            raise ValueError('primal certificate model hash mismatch')
        moments = read_primal_moments(candidate, len(flags))
        report['exact_primal'] = exact_primal(moments, maps, meta['block_sizes'], flags, old_matrix, cross,
                                              primal_kernels if primal_kernels is not None else candidate.get('kernels'))
    if options.dual:
        candidate = read_json(options.dual)
        if candidate['model_sha256'] != model_hash:
            raise ValueError('dual certificate model hash mismatch')
        report['exact_dual'] = exact_dual(candidate, chart, maps, meta['block_sizes'],
                                          manifest.representatives, flags, conditional)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Round 8 artifact directory')
    parser.add_argument('--prior-face', required=True, type=Path, help='audited Round 6 support-four manifest')
    parser.add_argument('--prior-chart', required=True, type=Path, help='Round 7 exact affine chart')
    parser.add_argument('--prior-audit', type=Path, help='hash-matched exact audit of the prior face')
    parser.add_argument('--dual', type=Path, help='integer Gram factors and optional rational box weights')
    parser.add_argument('--common-kernel', type=Path, help='support5-common-kernel-v1 certificate')
    parser.add_argument('--primal', type=Path, help='exact original-moment vector')
    args = parser.parse_args()
    inherited = read_json(args.prior_audit or args.prior_face.with_name('independent_face_verification.json'))
    report = verify_artifacts(args.input, args.prior_face, args.prior_chart, inherited, args)
    print(json.dumps(report, indent=2, allow_nan=False))
    return int(any(not report[side]['passed'] for side in ('exact_primal', 'exact_dual') if side in report))


if __name__ == '__main__':
    raise SystemExit(main())
