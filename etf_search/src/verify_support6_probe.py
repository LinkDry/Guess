"""Check exact inherited or rounded primal points of the support-six sparse pencil.

The default checks the model's sparse artifacts. --audit-stencils additionally
reconstructs the sparse rows from canonical polynomials and moment orbits.
All floating eigenspaces and Cholesky factors are certificate proposals only.
"""

import argparse
from collections import defaultdict
from fractions import Fraction
from hashlib import file_digest
import json
from math import gcd, isfinite, lcm
from pathlib import Path
from random import Random

from flint import fmpq_mat, fmpz_mat, nmod_mat
import numpy as np
from scipy.linalg import eigh, qr, solve_triangular
from scipy.sparse import csr_matrix
from sympy import nextprime

from level3_face_model import kernel_directions
from verify_sdp_anchored import VARIABLES
from verify_sdp_factor_cert import monomial
from verify_sdp_level3 import BASIS, MomentManifest, canonical_psd_bases, equation_terms, selected_monomials
from verify_sdp_lmi import model_sha256, read_json, read_primal_moments
from verify_support5_rank import (
    FOCAL, audit_common_kernel, audit_image_rank, chart_column, chart_identities, check_projection,
    csr_row, parse_chart, primitive_parity_row, raw_cross, read_integer_map, verify_complement_minor,
)


def chart_point(rows, slots):
    return [sum(coefficient * slots[j] for j, coefficient in row.items()) for row in rows]


def verify_parent_chart(rows, prior_rows, prior_free):
    """Verify child chart inclusion without casting parent coefficients to int64."""
    lifted_free = [{0: 1}, *(rows[j] for j in prior_free)]
    for i, prior_row in enumerate(prior_rows):
        composed = defaultdict(Fraction)
        for j, coefficient in prior_row.items():
            for slot, value in lifted_free[j].items():
                composed[slot] += coefficient * value
        if rows[i] != {slot: value for slot, value in composed.items() if value}:
            raise ValueError(f'child chart disagrees with parent chart at moment {i}')


def inherited_slots(rows, free, prefix):
    """Find a rational extension of the entire prefix, or return None if inconsistent."""
    unknown = [j for j, i in enumerate(free, 1) if i >= len(prefix)]
    slots = [Fraction(1), *(Fraction(prefix[i]) if i < len(prefix) else Fraction(0) for i in free)]
    existing = chart_point(rows[:len(prefix)], slots)
    equations = []
    for row, target, value in zip(rows, prefix, existing):
        coefficients = [row.get(j, 0) for j in unknown]
        rhs = target - value
        if any(coefficients):
            equations.append([*coefficients, rhs])
        elif rhs:
            return None
    if equations:
        width = len(unknown) + 1
        reduced, rank = fmpq_mat(len(equations), width, [str(x) for row in equations for x in row]).rref()
        for i in range(rank):
            pivot = next(j for j in range(width) if reduced[i, j])
            if pivot == width - 1:
                return None
            slots[unknown[pivot]] = Fraction(str(reduced[i, width - 1]))
    if chart_point(rows[:len(prefix)], slots) != list(prefix):
        raise ArithmeticError('exact inherited extension failed its substitution check')
    return slots


def rounded_slots(point, retained, dimension, denominator):
    if point.shape != (len(retained),) or not np.isfinite(point).all() or denominator <= 0:
        raise ValueError('expected a finite solver vector and positive rounding denominator')
    slots = [Fraction(1)] + [Fraction(0)] * dimension
    for j, value in zip(retained, point, strict=True):
        slots[j + 1] = Fraction(round(Fraction.from_float(float(value)) * denominator), denominator)
    return slots


def specialize_block_ranks(maps, sizes, rows, flags, free_dimension):
    """Bound each common nullity by ranks at one exact affine specialization."""
    for prime, beta in ((1_000_000_009, 184714173), (1_000_003, 78843)):
        if (5 * beta**2 - 2) % prime:
            continue
        generator = Random(1924)
        point = [1, *(generator.randrange(1, prime) for _ in range(free_dimension))]
        moments = list(map(Fraction, chart_point(rows, point)))
        if any(moment.denominator % prime == 0 for moment in moments):
            continue
        image = np.asarray([(q.numerator * pow(q.denominator, -1, prime) * (beta if odd else 1)) % prime
                            for q, odd in zip(moments, flags, strict=True)], dtype=np.int64)
        ranks = []
        for matrix, size in zip(maps, sizes, strict=True):
            if matrix.nnz:
                bound = max(abs(int(matrix.data.min())), abs(int(matrix.data.max())))
                bound *= int(np.diff(matrix.indptr).max()) * (prime - 1)
                if bound >= 2**63:
                    raise ValueError('sparse modular specialization exceeds exact integer range')
            packed = (matrix @ image) % prime
            col, row = np.triu_indices(size)
            dense = np.empty((size, size), dtype=np.int64)
            dense[row, col] = dense[col, row] = packed
            ranks.append(nmod_mat(size, size, dense.ravel().tolist(), prime).rank())
        return {'prime': prime, 'beta_image': beta, 'free_point': point[1:], 'specialized_block_ranks': ranks,
                'common_nullity_upper_bounds': [size - rank for size, rank in zip(sizes, ranks, strict=True)]}
    raise ValueError('no valid modular specialization for this affine chart')


def retained_bases(meta):
    """Check structural polynomial origins and quotient pivots in canonical order."""
    basis = list(canonical_psd_bases(meta['stage']).values())
    words = set(selected_monomials(meta['stage']))
    equations = {name: {word: tuple(map(int, values)) for word, values in equation_terms(name).items()}
                 for name in FOCAL}
    blocks = [{'basis': [tuple(sorted(poly.items())) for poly in block]} for block in basis]
    directions, pivots = kernel_directions({**meta, 'base_equations': equations}, blocks, high_first=True)
    result = []
    for block, kernel, pivot, size, nullity in zip(
            basis, directions, pivots, meta['block_sizes'], meta['kernel_dimensions'], strict=True):
        projections = {word: (j, sign, len(poly)) for j, poly in enumerate(block) for word, sign in poly.items()}
        for witness in kernel:
            check_projection(witness['coordinates'], witness['origin'], Fraction(*witness['projection_scale']),
                             projections, words)
        verify_complement_minor([w['coordinates'] for w in kernel], sorted(pivot))
        keep = [poly for j, poly in enumerate(block) if j not in pivot]
        if len(keep) != size or len(pivot) != nullity:
            raise ValueError('canonical structural quotient disagrees with model')
        result.append(keep)
    return result


def audit_cross_stencils(root, cross, meta, manifest, flags):
    allowed = set(selected_monomials(meta['stage']))
    witnesses = read_json(root / 'cross_provenance.json.gz')
    if len(witnesses) != cross.shape[0]:
        raise ValueError('cross provenance count mismatch')
    for i, (name, multiplier, test) in enumerate(witnesses):
        multiplier = monomial(multiplier, len(VARIABLES))
        test = monomial(test, len(VARIABLES))
        row = primitive_parity_row(raw_cross(manifest, name, multiplier, test, allowed), flags)
        if not row or row != csr_row(cross, i):
            raise ValueError(f'cross row {i} disagrees with its polynomial origin')


def integer_block(matrix, parity, values, keep=None):
    """Evaluate the rational congruence 2 D^-1 M D^-1 with a positive integer scale."""
    denominator = lcm(*(Fraction(value).denominator for value in values))
    moments = [int(value * denominator) for value in values]
    size = len(parity)
    indices = list(range(size)) if keep is None else keep
    entries = [[0] * len(indices) for _ in indices]
    for j_out, j in enumerate(indices):
        for i_out in range(j_out, len(indices)):
            i = indices[i_out]
            packed = j * size - j * (j - 1) // 2 + i - j
            lo, hi = matrix.indptr[packed:packed + 2]
            value = sum(int(weight) * moments[column] for column, weight in
                        zip(matrix.indices[lo:hi], matrix.data[lo:hi], strict=True))
            factor = 5 if parity[i] and parity[j] else 2
            entries[i_out][j_out] = entries[j_out][i_out] = factor * value
    return entries, denominator


def exact_sign(entries, approximate, lowest_vector=None):
    """Certify positive definiteness by exact integer congruence or test a negative vector."""
    n = len(entries)
    if not n:
        return 'psd', 'exact_full_kernel'
    matrix = fmpz_mat(entries)
    if lowest_vector is not None:
        vector = fmpz_mat(n, 1, [round(float(value) * 10**10) for value in lowest_vector])
        if (vector.transpose() * matrix * vector)[0, 0] < 0:
            return 'indefinite', 'exact_negative_quadratic_form'
    try:
        lower = np.linalg.cholesky(approximate)
    except np.linalg.LinAlgError:
        return 'inconclusive', 'floating_cholesky_failed'
    inverse = solve_triangular(lower, np.eye(n), lower=True, check_finite=False)
    largest = np.max(np.abs(inverse))
    if not np.isfinite(largest) or largest <= 0:
        return 'inconclusive', 'nonfinite_inverse_factor'
    weights = np.tril(np.rint(10**10 * (inverse / largest)).astype(np.int64))
    if np.any(np.diag(weights) == 0):
        return 'inconclusive', 'rounded_factor_singular'
    w = fmpz_mat(weights.tolist())
    congruence = w * matrix * w.transpose()
    for i in range(n):
        if congruence[i, i] <= sum(abs(congruence[i, j]) for j in range(n) if i != j):
            return 'inconclusive', 'exact_diagonal_dominance_failed'
    return 'psd', 'exact_diagonal_dominance'


def primitive_integer_column(vector):
    denominator = lcm(*(Fraction(value).denominator for value in vector))
    integer = [int(value * denominator) for value in vector]
    divisor = gcd(*integer) or 1
    return [value // divisor for value in integer]


def verify_common_batched(matrix, kernel, chart):
    """Prove M(c)Q=0 on all affine coefficients by bounded CRT in small panels."""
    parity = np.asarray(kernel['basis_parity'], dtype=bool)
    size = len(parity)
    if not kernel['vectors'] or not matrix.nnz:
        return
    vectors = []
    for terms in kernel['vectors']:
        sparse = {j: Fraction(value) for j, value in terms}
        vectors.append(primitive_integer_column([sparse.get(j, 0) for j in range(size)]))
    width = 1 + max(max(row, default=0) for row in chart)
    coefficients = [primitive_integer_column(chart_column(chart, j)) for j in range(width)]
    chart_bound = max(max(map(abs, column)) for column in coefficients)
    row_bound = int(np.diff(matrix.indptr).max()) * max(abs(int(matrix.data.min())), abs(int(matrix.data.max())))
    kernel_bound = max(sum(map(abs, vector)) for vector in vectors)
    bound = 5 * row_bound * chart_bound * kernel_bound
    col, row = np.triu_indices(size)
    factors = np.where(parity[row] & parity[col], 5, 2)
    dense = np.empty((size, size), dtype=np.float64)
    modulus_product, prime = 1, 997_000
    while modulus_product <= bound:
        prime = int(nextprime(prime))
        # All terms are nonnegative integers. Both sparse integer accumulation
        # and dense BLAS dot products are exact under these rigorous bounds.
        if row_bound * (prime - 1) >= 2**63 or size * (prime - 1)**2 >= 2**53:
            raise ValueError('common kernel CRT products would exceed exact arithmetic range')
        q = np.ascontiguousarray([[value % prime for value in v] for v in vectors], dtype=np.float64).T
        for start in range(0, width, 4):
            panel = np.ascontiguousarray([[c % prime for c in column]
                                          for column in coefficients[start:start + 4]], dtype=np.int64)
            packed = (matrix @ panel.T) % prime
            for offset in range(packed.shape[1]):
                entries = (packed[:, offset] * factors) % prime
                dense[row, col] = entries
                dense[col, row] = entries
                product = (dense @ q).astype(np.int64)
                if np.any(product % prime):
                    raise ValueError(f'common kernel identity fails at affine coefficient {start + offset}')
        modulus_product *= prime


def propose_kernel(entries, eigenspace, parity):
    """Recover a pivot-normalized kernel and check every pointwise matrix row."""
    nullity = eigenspace.shape[1]
    _, _, ordering = qr(eigenspace.T, pivoting=True, mode='economic')
    pivots = sorted(map(int, ordering[:nullity]))
    keep = sorted(set(range(len(entries))) - set(pivots))
    vectors = [[Fraction(int(i == pivot)) for i in range(len(entries))] for pivot in pivots]
    if keep:
        principal = fmpq_mat([[entries[i][j] for j in keep] for i in keep])
        rhs = fmpq_mat([[-entries[i][j] for j in pivots] for i in keep])
        try:
            solved = principal.solve(rhs, algorithm='dixon')
        except (ValueError, ZeroDivisionError):
            return None, None
        for k, vector in enumerate(vectors):
            for j, i in enumerate(keep):
                vector[i] = Fraction(str(solved[j, k]))
    integer_vectors = fmpz_mat([primitive_integer_column(vector) for vector in vectors])
    zero = fmpz_mat(len(entries), nullity)
    if fmpz_mat(entries) * integer_vectors.transpose() != zero:
        return None, None
    record = {'size': len(entries), 'basis_parity': parity, 'pivot_indices': pivots,
              'vectors': [[[j, str(value)] for j, value in enumerate(vector) if value] for vector in vectors]}
    return keep, record


def check_block(matrix, parity, chart, values, tolerance, common_nullity_bound=None):
    entries, denominator = integer_block(matrix, parity, values)
    approximate = np.asarray(np.asarray(entries, dtype=object) / denominator, dtype=float)
    eigenvalues, eigenvectors = eigh(approximate, check_finite=True, driver='evd')
    minimum = float(eigenvalues[0])
    nullity = int(np.count_nonzero(abs(eigenvalues) <= tolerance))
    size = len(entries)
    report = {'original_block_order': size, 'principal_order': size, 'min_eigenvalue_approx': minimum,
              'numerical_nullity': nullity, 'verified_kernel_dimension': 0, 'common_kernel_dimension': 0}
    common = None
    if minimum >= -tolerance and nullity:
        keep, proposal = propose_kernel(entries, eigenvectors[:, :nullity], parity)
        if proposal is not None:
            report.update(principal_order=len(keep), verified_kernel_dimension=nullity)
            if common_nullity_bound != 0:
                try:
                    if nullity > 16:
                        verify_common_batched(matrix, proposal, chart)
                    else:
                        schema = {'schema': 'support5-common-kernel-v1', 'blocks': [proposal]}
                        width = 1 + max(max(row, default=0) for row in chart)
                        audit_common_kernel(schema, [matrix], [parity], chart, width)
                except ValueError as exc:
                    report['common_kernel_check'] = str(exc)
                else:
                    common = proposal
                    report['common_kernel_dimension'] = nullity
            entries = [[entries[i][j] for j in keep] for i in keep]
            approximate = approximate[np.ix_(keep, keep)]
    proposal_vector = eigenvectors[:, 0] if report['principal_order'] == size and minimum < 0 else None
    report['status'], report['method'] = exact_sign(entries, approximate, proposal_vector)
    return report, common


def verify(root, prior, solution, inherited, denominator, tolerance, audit_stencils, include_certificates):
    meta = read_json(root / 'model.json')
    old_meta = read_json(prior / 'model.json')
    flags = [len(word) % 2 for word in meta['moments']]
    old_flags = [len(word) % 2 for word in old_meta['moments']]
    count = len(old_flags)
    if (meta['stage'] != 'support6' or old_meta['stage'] != 'support5' or meta['old_moments'] != count
            or meta['moments'][:count] != old_meta['moments'] or len(meta['block_sizes']) != 8):
        raise ValueError('model stage, blocks or support-five moment prefix disagree')
    model_hash = model_sha256(root / 'model.json')
    parent_hash = model_sha256(prior / 'model.json')
    chart_path = root / 'affine_chart.json.gz'
    parent_path = prior / 'affine_chart.json.gz'
    chart_doc = read_json(chart_path)
    parent_doc = read_json(parent_path)
    if (chart_doc['schema'] != 'support6-affine-chart-v1' or parent_doc['schema'] != 'support5-affine-chart-v1'
            or chart_doc['model_sha256'] != model_hash or parent_doc['model_sha256'] != parent_hash
            or chart_doc['source_chart_sha256'] != model_sha256(parent_path)):
        raise ValueError('model or source chart hash or schema mismatch')
    rows, free = parse_chart(chart_doc, flags)
    parent_rows, parent_free = parse_chart(parent_doc, old_flags)
    verify_parent_chart(rows, parent_rows, parent_free)
    cross = read_integer_map(root / 'cross_rows.npz', len(flags), meta['unique_cross_rows'])
    ranks = chart_identities((rows, free, chart_doc), (parent_rows, parent_free), cross,
                             csr_matrix((0, count), dtype=np.int64))
    report = {'stage': 'support6', 'model_sha256': model_hash, 'inherited_model_sha256': parent_hash,
              'moment_count': len(flags), 'equalities': ranks, 'cross_rows_checked': cross.shape[0],
              'canonical_stencils_checked': audit_stencils, 'candidates': {}}
    candidates = {}
    if inherited is not None:
        document = read_json(inherited)
        if document.get('model_sha256', parent_hash) != parent_hash:
            raise ValueError('inherited primal model hash mismatch')
        pairs = read_primal_moments(document, count)
        if any(value[1 - odd] for value, odd in zip(pairs, old_flags, strict=True)):
            raise ValueError('inherited primal moment violates the chart basis parity')
        prefix = [value[odd] for value, odd in zip(pairs, old_flags, strict=True)]
        if prefix[0] != 1 or chart_point(parent_rows, [1, *(prefix[j] for j in parent_free)]) != prefix:
            raise ValueError('inherited primal does not lie on the support-five chart')
        slots = inherited_slots(rows, free, prefix)
        if slots is None:
            failure = {'status': 'no_exact_equality_extension'}
            if count == len(flags):
                residuals = []
                for i in range(cross.shape[0]):
                    residual = sum(weight * prefix[j] for j, weight in csr_row(cross, i).items())
                    if residual:
                        residuals.append({'row': i, 'residual': str(residual)})
                failure['violated_cross_rows'] = residuals
            report['candidates']['inherited'] = failure
        else:
            candidates['inherited'] = (slots, {'prefix_preserved': count})
    maps = [read_integer_map(root / f'psd_map_{i}.npz', len(flags), size * (size + 1) // 2)
            for i, size in enumerate(meta['block_sizes'])]
    rank = read_json(root / 'pencil_rank_certificate.json')
    report['psd_image'] = audit_image_rank(rank, maps, rows, len(free))
    specialization = specialize_block_ranks(maps, meta['block_sizes'], rows, flags, len(free))
    report['affine_pencil_specialization'] = specialization
    if solution is not None:
        with solution.open('rb') as stream:
            source_hash = file_digest(stream, 'sha256').hexdigest()
            stream.seek(0)
            with np.load(stream, allow_pickle=False) as archive:
                slots = rounded_slots(archive['x'], rank['retained_direction_columns'], len(free), denominator)
        candidates['rounded'] = (slots, {'source_solution_sha256': source_hash,
                                         'source_chart_free_rounding_denominator': denominator})
    if not candidates and not audit_stencils:
        return report
    if audit_stencils:
        manifest = MomentManifest({**meta, 'variables': list(map(list, VARIABLES)), 'coefficient_basis': BASIS})
        audit_cross_stencils(root, cross, meta, manifest, flags)
    bases = retained_bases(meta)
    parities = [[len(next(iter(poly))) % 2 for poly in block] for block in bases]
    checked_blocks = []
    for k, (matrix, basis, parity) in enumerate(zip(maps, bases, parities, strict=True)):
        for position, (i, j) in enumerate((i, j) for j in range(len(basis)) for i in range(j, len(basis))):
            lo, hi = matrix.indptr[position:position + 2]
            if any(flags[column] != (parity[i] ^ parity[j]) for column in matrix.indices[lo:hi]):
                raise ValueError(f'moment parity mismatch at block {k}, packed row {position}')
            if audit_stencils and csr_row(matrix, position) != manifest.character_product(basis[i], basis[j]):
                raise ValueError(f'stencil disagrees with canonical basis at block {k}, packed row {position}')
        checked_blocks.append(matrix.shape[0])
    report['packed_psd_rows_checked'] = sum(checked_blocks)
    for name, (slots, source) in candidates.items():
        values = chart_point(rows, slots)
        results, kernels = [], []
        for k, (matrix, parity, bound) in enumerate(zip(
                maps, parities, specialization['common_nullity_upper_bounds'], strict=True)):
            result, kernel = check_block(matrix, parity, rows, values, tolerance, bound)
            results.append({'block': k, **result})
            kernels.append(kernel)
        statuses = {result['status'] for result in results}
        status = ('psd' if statuses == {'psd'} else 'indefinite' if 'indefinite' in statuses else 'inconclusive')
        entry = {'status': status, **source, 'blocks': results}
        if include_certificates and status == 'psd':
            entry['primal'] = {'schema': 'anchored-exact-primal-v1', 'coefficient_basis': BASIS,
                               'model_sha256': model_hash, **source,
                               'moments': [['0', str(x)] if odd else [str(x), '0']
                                           for x, odd in zip(values, flags, strict=True)]}
            entry['verified_common_kernels'] = kernels
        report['candidates'][name] = entry
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='support-six artifact directory')
    parser.add_argument('--prior', type=Path, required=True, help='support-five artifact directory')
    parser.add_argument('--solution', type=Path, help='Newton solver_primal_dual.npz checkpoint')
    parser.add_argument('--inherited-primal', type=Path, help='support-five exact witness to try to extend')
    parser.add_argument('--denominator', type=int, default=10**14)
    parser.add_argument('--kernel-tolerance', type=float, default=1e-8, help='numerical proposal threshold only')
    parser.add_argument('--audit-stencils', action='store_true', help='also reconstruct supplied sparse stencils')
    parser.add_argument('--include-certificates', action='store_true', help='embed successful exact witnesses in JSON')
    args = parser.parse_args()
    if args.denominator <= 0 or not isfinite(args.kernel_tolerance) or args.kernel_tolerance <= 0:
        parser.error('positive denominator and finite positive kernel tolerance required')
    report = verify(args.input, args.prior, args.solution, args.inherited_primal, args.denominator,
                    args.kernel_tolerance, args.audit_stencils, args.include_certificates)
    print(json.dumps(report, indent=2, allow_nan=False))
    return int(bool(report['candidates']) and all(c['status'] != 'psd' for c in report['candidates'].values()))


if __name__ == '__main__':
    raise SystemExit(main())
