"""Eliminate every affine equality in the facially reduced ETF(4,9) LMI."""
import argparse
import gzip
import hashlib
import json
from fractions import Fraction
from math import lcm
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix


BASIS = ('1', 'sqrt(10)/5')


def rational_equalities(model):
    """Set x_i=beta^(degree_i mod 2) z_i and clear each row's common beta."""
    if tuple(model['coefficient_basis']) != BASIS or model['moments'][0]:
        raise ValueError('expected anchored moment order, including the constant')
    parity = np.asarray([len(word) % 2 for word in model['moments']], dtype=np.int8)
    indices, offsets, coefficients = [0], [0, 1], [1]
    for row in model['equalities']:
        tag = None
        for column, (a, b) in row.items():
            i = int(column)
            if not (a or b):
                continue
            if a and b:
                raise ValueError('a coefficient has mixed parity')
            current = (int(bool(b)) + int(parity[i])) % 2
            if tag is not None and current != tag:
                raise ValueError('an equality has mixed parity')
            tag = current
            indices.append(i)
            coefficients.append(5*a if a else (2 if parity[i] else 5)*b)
        offsets.append(len(indices))
    equality = csr_matrix((np.asarray(coefficients, dtype=np.int64), indices, offsets),
                          shape=(len(offsets)-1, len(parity)))
    return parity, equality


def check_all_equalities(equality, columns, *, normalized=True):
    """Check E[X0,N]=[e0,0] with integer arithmetic and guarded accumulation."""
    row_bound = max(sum(abs(int(value)) for value in equality.data[a:b])
                    for a, b in zip(equality.indptr[:-1], equality.indptr[1:]))
    bounded = []
    for slot, column in enumerate(columns):
        denominator = lcm(*(value.denominator for value in column))
        integer_column = [int(value * denominator) for value in column]
        expected = denominator if normalized and slot == 0 else 0
        if max(row_bound * max(map(abs, integer_column)), expected) < 2**63:
            bounded.append((slot, integer_column, expected))
        else:
            # This branch has arbitrary-precision accumulators for both signs.
            for i, (a, b) in enumerate(zip(equality.indptr[:-1], equality.indptr[1:])):
                value = sum(int(equality.data[k])*integer_column[equality.indices[k]] for k in range(a, b))
                if value != (expected if i == 0 else 0):
                    raise ArithmeticError(f'exact equality failed in row {i}, chart slot {slot}')
    for begin in range(0, len(bounded), 16):
        batch = bounded[begin:begin+16]
        result = equality @ np.ascontiguousarray(np.asarray([v for _, v, _ in batch], dtype=np.int64).T)
        if np.any(result[1:]) or result[0].tolist() != [rhs for _, _, rhs in batch]:
            raise ArithmeticError(f'exact equality failed in chart slots {[i for i, _, _ in batch]}')


def affine_chart(equality, selected_rows):
    """Use QR only for pivot selection. Dixon and all-row checks prove the chart."""
    from flint import fmpq_mat
    from scipy.linalg import qr

    selected = [int(i) for i in selected_rows]
    if 0 not in selected or len(set(selected)) != len(selected) or min(selected) < 0 or max(selected) >= equality.shape[0]:
        raise ValueError('selected equality rows must be unique and include normalization')
    square_rows = equality[selected].toarray()
    normalized = square_rows / np.linalg.norm(square_rows.astype(float), axis=1)[:, None]
    _, triangle, order = qr(normalized, mode='economic', pivoting=True, check_finite=True)
    rank = len(selected)
    pivots = list(map(int, order[:rank]))
    pivot_set = set(pivots)
    free = [i for i in range(equality.shape[1]) if i not in pivot_set]
    right = np.column_stack(([int(i == 0) for i in selected], -square_rows[:, free]))
    left_q = fmpq_mat(rank, rank, square_rows[:, pivots].ravel().tolist())
    right_q = fmpq_mat(rank, len(free)+1, right.ravel().tolist())
    print(f'Exact Dixon solve for {rank} pivots and {len(free)+1} right-hand sides', flush=True)
    start = perf_counter()
    solved = left_q.solve(right_q, algorithm='dixon')
    seconds_dixon = perf_counter() - start
    n = equality.shape[1]
    columns = [[Fraction(0)]*n for _ in range(len(free)+1)]
    for j, free_index in enumerate(free, 1):
        columns[j][free_index] = Fraction(1)
    for i, pivot in enumerate(pivots):
        for j, column in enumerate(columns):
            value = solved[i,j]
            if value:
                column[pivot] = Fraction(int(value.numerator), int(value.denominator))
    print(f'Dixon complete in {seconds_dixon:.2f} s. Checking all {equality.shape[0]} equalities.', flush=True)
    start = perf_counter()
    check_all_equalities(equality, columns)
    seconds_check = perf_counter() - start
    chart_rows = []
    for coefficient_vector in zip(*columns, strict=True):
        denominator = lcm(*(v.denominator for v in coefficient_vector))
        chart_rows.append([denominator, [[j, int(v*denominator)] for j, v in enumerate(coefficient_vector) if v]])
    return {'pivot_columns': pivots, 'free_columns': free, 'selected_rows': selected, 'rows': chart_rows,
            'minimum_numeric_qr_pivot': float(abs(triangle[rank-1, rank-1])),
            'seconds_dixon': seconds_dixon, 'seconds_check_all_equalities': seconds_check}


def packed_lmi(model, parity, rows):
    """Raise each exact scaled coordinate into packed SCS order without thresholding."""
    from scipy.sparse import csc_matrix

    sizes = [len(block['basis']) for block in model['psd_blocks']]
    n_packed = sum(size*(size+1)//2 for size in sizes)
    n_slots = 1 + max(slot for _, terms in rows for slot, _ in terms)
    chart_rows, chart_cols, chart_values = [], [], []
    beta = np.sqrt(10)/5
    for i, (denominator, terms) in enumerate(rows):
        scale = beta if parity[i] else 1
        for slot, numerator in terms:
            chart_rows.append(i)
            chart_cols.append(slot)
            chart_values.append(scale * (numerator/denominator))
    chart = csc_matrix((chart_values, (chart_rows, chart_cols)), shape=(len(rows), n_slots))
    integer_map, factors = packed_integer_map(model, sizes, len(rows), n_packed)
    expanded = (integer_map.multiply(factors[:, None]).tocsr() @ chart).tocsc()
    return -expanded[:, 1:], expanded[:, 0].toarray().ravel(), sizes


def packed_integer_map(model, sizes, n_moments, n_packed):
    offsets, columns, coefficients = [], [], []
    factors = np.full(n_packed, np.sqrt(2))
    offset = 0
    for block, size in zip(model['psd_blocks'], sizes, strict=True):
        for i, j, entry in block['entries']:
            if not 0 <= j <= i < size:
                raise ValueError('entry outside lower triangle')
            packed_index = offset + j*size - j*(j-1)//2 + i-j
            if i == j:
                factors[packed_index] = 1
            for column, coefficient in entry.items():
                offsets.append(packed_index)
                columns.append(int(column))
                coefficients.append(coefficient)
        offset += size*(size+1)//2
    matrix = coo_matrix((np.asarray(coefficients, dtype=np.int64), (offsets, columns)),
                        shape=(n_packed, n_moments)).tocsr()
    return matrix, factors


def restrict_lmi(model, chart, matrix, rhs, sizes, target):
    """Select an exact spanning subfamily of the pencil directions."""
    from flint import fmpq_mat
    from scipy.linalg import qr
    from scipy.sparse import csc_matrix, vstack

    start = perf_counter()
    parity = np.asarray(chart['parity'])
    integer_map, _ = packed_integer_map(model, sizes, len(parity), matrix.shape[0])
    # Splitting coefficients into 1 and beta proves kernel identities over Q.
    rational_map = vstack([integer_map.multiply((parity == p)[None, :]) for p in (0, 1)], format='csr')
    rational_map.eliminate_zeros()
    zero = Fraction(0)
    columns = [[zero]*len(parity) for _ in range(matrix.shape[1])]
    row_indices, slots, approximations = [], [], []
    for row, (denominator, terms) in enumerate(chart['rows']):
        for slot, numerator in terms:
            if slot:
                value = Fraction(numerator, denominator)
                columns[slot-1][row] = value
                row_indices.append(row)
                slots.append(slot-1)
                approximations.append(float(value))
    approx_chart = csc_matrix((approximations, (row_indices, slots)), shape=(len(parity), matrix.shape[1]))
    approx_image = (rational_map @ approx_chart).tocsc()
    nonzero = np.flatnonzero(np.diff(approx_image.indptr)).tolist()
    approx_visible = approx_image[:, nonzero].toarray()
    _, triangle, column_order = qr(approx_visible, mode='economic', pivoting=True, check_finite=False)
    diagonal = np.abs(triangle.diagonal())
    rank = int(np.count_nonzero(diagonal > diagonal[0]*1e-11))
    kept = sorted(nonzero[i] for i in column_order[:rank])
    dropped = sorted(set(range(matrix.shape[1])) - set(kept))
    dependent = sorted(set(nonzero) - set(kept))
    _, _, row_order = qr(approx_image[:, kept].toarray().T, mode='economic', pivoting=True, check_finite=False)
    witness_rows = row_order[:rank]

    moment_stencils = [list(zip(rational_map.indices[a:b], map(int, rational_map.data[a:b])))
                       for a, b in zip(rational_map.indptr[witness_rows], rational_map.indptr[witness_rows+1])]
    def exact_entries(indices):
        return [str(sum(c*columns[j][i] for i, c in stencil)) for stencil in moment_stencils for j in indices]

    independent = fmpq_mat(rank, rank, exact_entries(kept))
    combinations = independent.solve(fmpq_mat(rank, len(dependent), exact_entries(dependent)), algorithm='dixon')
    coefficient_rows = {j: [] for j in dropped}
    for l, j in enumerate(dependent):
        for k, combination in enumerate(kept):
            value = Fraction(str(combinations[k,l]))
            if value:
                coefficient_rows[j].append([combination, str(value)])
    kernel = []
    for j in dropped:
        column = columns[j].copy()
        for k, coefficient in coefficient_rows[j]:
            value = Fraction(coefficient)
            column = [a-value*b if b else a for a,b in zip(column, columns[k], strict=True)]
        kernel.append(column)
    check_all_equalities(rational_map, kernel, normalized=False)
    restricted = matrix[:, kept]
    np.savez_compressed(target, A_data=restricted.data, A_indices=restricted.indices, A_indptr=restricted.indptr,
                        A_shape=restricted.shape, b=rhs, q=np.zeros(rank), block_sizes=sizes,
                        psd_congruence=congruence_weights(model), retained_chart_columns=kept,
                        full_chart_dimension=matrix.shape[1])
    metadata = {'image_rank_over_rationals_split': rank, 'retained_chart_columns': kept,
                'exact_witness_split_packed_rows': witness_rows.tolist(), 'discarded_columns_in_retained_span': coefficient_rows,
                'split_packed_rows_checked_exactly': rational_map.shape[0], 'restricted_conic_nnz': restricted.nnz,
                'seconds': perf_counter()-start}
    (target.parent/'pencil_rank_certificate.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print(f'Exact spanning check. {rank} retained directions, {restricted.nnz} packed coefficients.', flush=True)


def congruence_weights(model):
    return np.concatenate([
        np.concatenate([w[j] * w[j:] for j in range(len(w))])
        for block in model['psd_blocks']
        for w in [1/np.sqrt(np.array([sum(c*c for _, c in p) for p in block['basis']], dtype=float))]
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='Round 6 face-model archive directory')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--restrict-only', action='store_true', help='Restrict a saved pencil to an exact spanning set')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    raw_model = gzip.decompress((args.source/'solver_face_model.json.gz').read_bytes())
    digest = hashlib.sha256(raw_model).hexdigest()
    model = json.loads(raw_model)
    del raw_model
    if args.restrict_only:
        with gzip.open(args.output/'affine_chart.json.gz', 'rt', encoding='utf-8') as stream:
            chart = json.load(stream)
        if chart['model_sha256'] != digest:
            raise ValueError('chart manifest hash mismatch')
        from scipy.sparse import csc_matrix
        with np.load(args.output/'lmi_conic_data.npz', allow_pickle=False) as data:
            matrix = csc_matrix((data['A_data'], data['A_indices'], data['A_indptr']), shape=data['A_shape'])
            restrict_lmi(model, chart, matrix, data['b'], data['block_sizes'].tolist(),
                         args.output/'lmi_conic_data_rank97.npz')
        return
    parity, equality = rational_equalities(model)
    with np.load(args.source/'solver_row_transform.npz', allow_pickle=False) as data:
        selected = data['selected']
        selected = selected[selected < equality.shape[0]]
    chart = affine_chart(equality, selected)
    specification = {'schema': 'anchored-affine-chart-v1', 'model_sha256': digest, 'parity': parity.tolist(),
                     **{k: chart[k] for k in ('pivot_columns', 'free_columns', 'selected_rows', 'rows')}}
    with (args.output/'affine_chart.json.gz').open('wb') as target:
        with gzip.GzipFile(fileobj=target, mode='wb', filename='', mtime=0) as stream:
            stream.write(json.dumps(specification, separators=(',', ':')).encode() + b'\n')
    matrix, rhs, sizes = packed_lmi(model, parity, chart['rows'])
    shape = matrix.shape
    artifacts = args.output/'lmi_conic_data.npz'
    np.savez_compressed(artifacts, A_data=matrix.data, A_indices=matrix.indices, A_indptr=matrix.indptr,
                        A_shape=shape, b=rhs, q=np.zeros(shape[1]), block_sizes=sizes,
                        psd_congruence=congruence_weights(model))
    report = {'model_sha256': digest, 'lmi_sha256': hashlib.sha256(artifacts.read_bytes()).hexdigest(),
              'equalities_checked_exactly': equality.shape[0], 'exact_rank': len(chart['pivot_columns']),
              'free_dimension': len(chart['free_columns']), 'zero_cone_rows': 0, 'psd_blocks': sizes,
              'conic_shape': shape, 'conic_nnz': matrix.nnz, 'conic_density': matrix.nnz/(shape[0]*shape[1]),
              'conic_csc_bytes': matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes,
              'exact_chart_nnz': sum(len(terms) for _, terms in chart['rows']),
              'largest_chart_denominator_bits': max(d.bit_length() for d, _ in chart['rows']),
              'minimum_numeric_qr_pivot': chart['minimum_numeric_qr_pivot'],
              'seconds_dixon': chart['seconds_dixon'], 'seconds_check_all_equalities': chart['seconds_check_all_equalities'],
              'wall_seconds': perf_counter() - start}
    (args.output/'build_summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
