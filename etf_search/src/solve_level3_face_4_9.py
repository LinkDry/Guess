"""Solve the exact structural face of a saved targeted ETF(4,9) moment model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, sleep

from solve_general_4_9 import THREADS, diagnostics, windows_job, write_json
from solve_level3_4_9 import (DEADLINE_ENV, clarabel_order, cone_data,
                              diagonal_cone_scaling, read_saved_model, run_solver)


def compress_rows(matrix, rhs, cone, tolerance=1e-10):
    """Select original rows using a sparse Gram rank estimate and validate in batches."""
    import numpy as np
    from scipy.linalg import eigvalsh, qr, solve_triangular

    zero_count = cone['z']
    equality = matrix[:zero_count, :].tocsr()
    spectrum = eigvalsh((equality.T @ equality).toarray(), check_finite=False)
    rank = int(np.count_nonzero(spectrum > tolerance * max(1, spectrum[-1])))
    if not rank:
        raise ArithmeticError('normalization row was absent')
    rng = np.random.default_rng(6)
    candidates = np.unique(np.r_[np.arange(min(zero_count, 1400)),
                                     rng.choice(zero_count, size=min(zero_count, 3*rank), replace=False)])
    for attempt in range(12):
        basis = equality[candidates].toarray()
        _, triangle, pivots = qr(basis.T, mode='economic', pivoting=True, check_finite=False)
        diagonal = np.abs(triangle.diagonal())
        retained_rank = min(rank, int(np.count_nonzero(diagonal > 1e-9)))
        kept = candidates[pivots[:retained_rank]]
        original_subset = equality[kept].toarray()
        q, r = qr(original_subset.T, mode='economic', check_finite=False)
        rhs_error = row_error = 0.
        failing = []
        dual_rhs = solve_triangular(r.T, rhs[kept], lower=True, check_finite=False)
        for begin in range(0, zero_count, 2048):
            end = min(begin + 2048, zero_count)
            batch = equality[begin:end]
            projections = batch @ q
            dense = batch.toarray()
            dense -= projections @ q.T
            errors = np.max(np.abs(dense), axis=1)
            row_error = max(row_error, float(np.max(errors, initial=0)))
            if np.any(errors > 5e-8):
                worst = np.argsort(errors)[-min(16, len(errors)):]
                failing.extend(begin + int(i) for i in worst if errors[i] > 5e-8)
            rhs_error = max(rhs_error, float(np.max(abs(projections @ dual_rhs - rhs[begin:end]), initial=0)))
        if not failing and rhs_error <= 5e-8:
            selected = np.r_[kept, np.arange(zero_count, matrix.shape[0])]
            return selected, {'old_rows': zero_count, 'retained_rows': retained_rank,
                              'sparse_gram_rank': rank, 'selection_attempts': attempt+1,
                              'max_dropped_row_error': row_error, 'max_dropped_rhs_error': rhs_error,
                              'smallest_retained_qr_pivot': float(diagonal[retained_rank-1]),
                              'smallest_positive_gram_eigenvalue': float(spectrum[-rank]),
                              'largest_discarded_gram_eigenvalue': float(spectrum[-rank-1]) if rank < len(spectrum) else None}
        if not failing or retained_rank == rank:
            raise ArithmeticError(f'cannot reproduce omitted equalities: {row_error=}, {rhs_error=}')
        candidates = np.unique(np.r_[kept, failing])
    raise ArithmeticError('independent row selection exceeded its retry budget')


def reconstruct_point(model, x, selected_rows):
    """Use exact arithmetic on the row subset, then check *all* original rows."""
    from fractions import Fraction
    from math import lcm

    import numpy as np
    from flint import fmpq, fmpq_mat
    from scipy.linalg import qr

    rows = model['equalities']
    subset = [rows[int(i)-1] for i in selected_rows if 0 < i <= len(rows)]
    parity = [len(word) % 2 for word in model['moments']]
    rational_rows = [{0: 1}]
    for row in subset:
        transformed = {}
        tags = set()
        for col, (a, b) in row.items():
            index = int(col)
            if a and b:
                raise ValueError('a coefficient has mixed parity')
            tags.add((bool(b) + parity[index]) % 2)
            transformed[index] = 5*a if a else (2 if parity[index] else 5)*b
        if len(tags) != 1:
            raise ValueError('an equality has mixed parity')
        rational_rows.append(transformed)
    n = len(parity)
    if len(x) != n or not np.isfinite(x).all():
        raise ValueError('invalid numeric moment coordinates')
    numeric_rows = np.zeros((len(rational_rows), n))
    for i, row in enumerate(rational_rows):
        for j, value in row.items():
            numeric_rows[i,j] = value
    _, tri_rows, row_order = qr(numeric_rows.T, pivoting=True, mode='economic')
    pivoted_rows = abs(tri_rows.diagonal())
    rank = int(np.count_nonzero(pivoted_rows > 1e-10*max(1, pivoted_rows[0])))
    kept = row_order[:rank]
    _, _, col_order = qr(numeric_rows[kept], pivoting=True, mode='economic')
    pivot_columns = list(map(int, col_order[:rank]))
    pivot_set = set(pivot_columns)
    scaling = np.where(parity, np.sqrt(10)/5, 1)
    guesses = [fmpq(round(float(v)*10**12), 10**12) for v in x/scaling]
    square_rows = [rational_rows[int(i)] for i in kept]
    left = fmpq_mat(rank, rank, [row.get(j,0) for row in square_rows for j in pivot_columns])
    right = fmpq_mat(rank, 1, [int(i == 0) - sum(coef*guesses[j] for j,coef in row.items() if j not in pivot_set)
                               for i,row in zip(kept,square_rows,strict=True)])
    print(f'Solving rank-{rank} exact rational square system by Dixon lifting', flush=True)
    solution = left.solve(right, algorithm='dixon')
    for offset, col in enumerate(pivot_columns):
        guesses[col] = solution[offset,0]
    coordinates = [(Fraction(0),Fraction(str(v))) if odd else (Fraction(str(v)),Fraction(0))
                   for v,odd in zip(guesses,parity,strict=True)]
    approximation = np.asarray([float(Fraction(str(v))) for v in guesses]) * scaling
    candidate = {'schema': 'anchored-exact-primal-v1', 'coefficient_basis': model['coefficient_basis'],
                 'rational_equality_rank': rank, 'maximum_coordinate_change': float(max(abs(approximation-x))),
                 'moments': [[str(a),str(b)] for a,b in coordinates]}
    denominator = lcm(*(v.denominator for pair in coordinates for v in pair))
    integer_coordinates = [(int(a*denominator), int(b*denominator)) for a,b in coordinates]
    failures = []
    for i, row in enumerate(rows):
        real = imag = 0
        for column, (a,b) in row.items():
            c,d = integer_coordinates[int(column)]
            real += 5*a*c + 2*b*d
            imag += a*d + b*c
        if real or imag:
            failures.append(i)
    candidate['all_equalities_exact'] = not failures and coordinates[0] == (1,0)
    candidate['failed_equality_count'] = len(failures)
    candidate['first_failed_equality_rows'] = failures[:16]
    return candidate


def run_reconstruction(args):
    import numpy as np
    root = args.reconstruct
    manifest_root = args.face_manifest or root
    manifest_path = manifest_root / 'solver_face_model.json'
    if manifest_path.exists():
        model = json.loads(manifest_path.read_text(encoding='utf-8'))
    else:
        import gzip

        with gzip.open(manifest_path.with_suffix('.json.gz'), 'rt', encoding='utf-8') as stream:
            model = json.load(stream)
    with np.load(root/'solver_primal_dual.npz', allow_pickle=False) as raw:
        x = raw['x']
    with np.load(manifest_root/'solver_row_transform.npz', allow_pickle=False) as transform:
        rows = transform['selected']
    candidate = reconstruct_point(model, x, rows)
    write_json(args.output/'solver_exact_primal_candidate.json', candidate)
    print(json.dumps({key: value for key,value in candidate.items() if key != 'moments'}), flush=True)
    return 0 if candidate['all_equalities_exact'] else 1


def solve_clarabel(matrix, rhs, cone, args, remaining):
    import clarabel
    import numpy as np
    from scipy.sparse import csc_matrix

    order = clarabel_order(cone)
    inverse = np.argsort(order)
    cones = [clarabel.ZeroConeT(cone['z']), *[clarabel.PSDTriangleConeT(i) for i in cone['s']]]
    settings = clarabel.DefaultSettings()
    settings.verbose = True
    settings.max_threads = 1
    settings.max_iter = args.iterations
    settings.time_limit = remaining
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = args.eps
    settings.chordal_decomposition_enable = False
    settings.direct_solve_method = 'qdldl'
    solver = clarabel.DefaultSolver(csc_matrix((matrix.shape[1], matrix.shape[1])),
                                    np.zeros(matrix.shape[1]), matrix[order].tocsc(),
                                    rhs[order], cones, settings)
    solution = solver.solve()
    info = solver.get_info()
    return ({'x': np.asarray(solution.x), 'y': np.asarray(solution.z)[inverse],
             's': np.asarray(solution.s)[inverse]},
            {'status': str(solution.status), 'iter': solution.iterations, 'res_pri': solution.r_prim,
             'res_dual': solution.r_dual, 'solve_time_seconds': solution.solve_time,
             'linear_solver_nnz_L': info.linsolver.nnzL, 'chordal_decomposition': False},
            {'tol': args.eps, 'time_limit_secs': remaining, 'iterations': args.iterations})


def save_matrix(path, matrix, rhs, cone):
    import numpy as np

    np.savez_compressed(path, data=matrix.data, indices=matrix.indices, indptr=matrix.indptr,
                        shape=matrix.shape, b=rhs, zero_cone=cone['z'], psd_cones=cone['s'])


def solve_cached(args):
    import numpy as np
    from scipy.sparse import csc_matrix

    source = args.cached
    with np.load(source/'solver_conic_data.npz', allow_pickle=False) as saved:
        A = csc_matrix((saved['data'], saved['indices'], saved['indptr']), shape=tuple(saved['shape']))
        b = saved['b']
        cone = {'z': int(saved['zero_cone']), 's': saved['psd_cones'].tolist()}
    if args.psd_margin:
        offset = cone['z']
        for size in cone['s']:
            diagonal = [offset + j*size-j*(j-1)//2 for j in range(size)]
            b[diagonal] -= args.psd_margin
            offset += size*(size+1)//2
        np.save(args.output/'solver_shifted_rhs.npy', b, allow_pickle=False)
    remaining = float(os.environ[DEADLINE_ENV]) - perf_counter() - 20
    if args.backend == 'clarabel':
        raw, info, settings = solve_clarabel(A, b, cone, args, remaining)
        backend_order = clarabel_order(cone)
    else:
        raw, info, _, settings = run_solver(args, A, b, cone, remaining)
        backend_order = np.arange(A.shape[0])
    arrays = {k: np.asarray(raw[k]) for k in ('x', 'y', 's')}
    np.savez_compressed(args.output/'solver_primal_dual.npz', **arrays,
                        y_backend=arrays['y'][backend_order], s_backend=arrays['s'][backend_order])
    stats = diagnostics(A, b, np.zeros(A.shape[1]), cone, arrays)
    write_json(args.output/'solver_summary.json', {'backend': args.backend, 'raw_info': info,
               'settings': settings, 'cached_conic_source': str(source), 'cone': cone,
               'required_psd_margin_solver_scale': args.psd_margin,
               'solver_scale_diagnostics': stats, 'numeric_only': True})
    return 0


def worker(args):
    if sys.stdin.readline() != 'start\n':
        return 2
    start = perf_counter()
    if args.reconstruct:
        return run_reconstruction(args)
    if args.cached:
        return solve_cached(args)
    import numpy as np
    from level3_face_model import reduce_model
    from solve_general_4_9 import pack_model

    model = read_saved_model(args.input)
    model['moments'] = tuple(map(tuple, model['moments']))
    for block in model['psd_blocks']:
        block['basis'] = [tuple((tuple(word), c) for word, c in p) for p in block['basis']]
    out = args.output
    print(f"Loaded {model['stage']} with blocks {[len(b['basis']) for b in model['psd_blocks']]}", flush=True)
    reduced = reduce_model(model)
    face = reduced['face_reduction']
    full_A, full_b, full_cone, _, _, row_scales = cone_data(reduced)
    manifest = pack_model(reduced)
    manifest['scope'] = reduced['scope']
    manifest['maximum_monomial_degree'] = 6
    manifest['equality_row_scales'] = row_scales
    manifest['source_model'] = str(args.input/'solver_model.json')
    manifest['source_sha256'] = hashlib.sha256((args.input/'solver_model.json').read_bytes()).hexdigest()
    temporary_manifest = out/'solver_face_model.json.tmp'
    with temporary_manifest.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(manifest, stream, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
    temporary_manifest.replace(out/'solver_face_model.json')
    print(f"Exact face blocks {full_cone['s']}, {face['added_equality_count']} new equalities, A{full_A.shape}", flush=True)

    scale = diagonal_cone_scaling(full_A, reduced, full_cone)
    scaled_A = full_A.multiply(scale[:, None]).tocsc()
    scaled_b = full_b * scale
    if args.keep_equalities:
        selected = np.arange(full_A.shape[0])
        compression = {'old_rows': full_cone['z'], 'retained_rows': full_cone['z']}
    else:
        selected, compression = compress_rows(scaled_A, scaled_b, full_cone)
    cone = {'z': compression['retained_rows'], 's': full_cone['s']}
    A, b = scaled_A[selected].tocsc(), scaled_b[selected]
    save_matrix(out/'solver_conic_data.npz', A, b, cone)
    np.savez_compressed(out/'solver_row_transform.npz', selected=selected, scale=scale)
    print(f"Solver A{A.shape} nnz={A.nnz}, rank compression {compression}", flush=True)
    remaining = float(os.environ[DEADLINE_ENV]) - perf_counter() - min(60, max(15, args.seconds*0.1))
    if remaining <= 0:
        raise TimeoutError('no remaining solver time')
    if args.backend == 'clarabel':
        native, info, settings = solve_clarabel(A, b, cone, args, remaining)
    else:
        native, info, _, settings = run_solver(args, A, b, cone, remaining)
    native = {key: np.asarray(native[key], dtype=float) for key in ('x', 'y', 's')}
    adjusted_y = np.zeros(full_A.shape[0])
    adjusted_y[selected] = native['y'] * scale[selected]
    adjusted_s = np.zeros(full_A.shape[0])
    adjusted_s[selected] = native['s'] / scale[selected]
    if args.backend == 'clarabel':
        order = clarabel_order(cone)
    else:
        order = np.arange(A.shape[0])
    np.savez_compressed(out/'solver_primal_dual.npz', x=native['x'], y=adjusted_y, s=adjusted_s,
                        y_solver_scs_order=native['y'], s_solver_scs_order=native['s'],
                        y_backend=native['y'][order], s_backend=native['s'][order])
    print(f"{args.backend} returned {info['status']}; saved vectors", flush=True)
    stats = diagnostics(full_A, full_b, np.zeros(full_A.shape[1]), full_cone,
                        {'x': native['x'], 'y': adjusted_y, 's': adjusted_s})
    base_A, base_b, base_cone, *_ = cone_data(model)
    original_stats = None
    if np.isfinite(native['x']).all():
        from solve_general_4_9 import diagnostic_eigenvalues
        residual = base_b - base_A @ native['x']
        original_stats = {'equality_inf': np.max(np.abs(residual[:base_cone['z']])),
                          'moment_psd_min_eigenvalues': diagnostic_eigenvalues(residual, base_cone)}
    write_json(out/'solver_summary.json', {
        'scope': reduced['scope'], 'stage': model['stage'], 'backend': args.backend, 'raw_info': info,
        'settings': settings, 'equality_compression': compression, 'psd_blocks': cone['s'],
        'equality_row_count_full': full_cone['z'], 'moments': A.shape[1], 'solver_nnz': A.nnz,
        'reduced_unscaled_diagnostics': stats, 'original_model_diagnostics': original_stats,
        'numeric_only': True, 'worker_seconds': perf_counter()-start,
    })
    return 0


def supervise(args):
    import psutil

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob('solver_*')):
        raise FileExistsError(f'Choose a fresh run directory: {out}')
    command = [sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:], '--worker']
    start = perf_counter()
    deadline = start + args.seconds
    limit = args.memory_mib * 2**20
    env = dict(os.environ, PYTHONUNBUFFERED='1', **{name: '1' for name in THREADS})
    env[DEADLINE_ENV] = str(deadline)
    peak, stop, close_job, error = 0, None, None, None
    with (out/'solver_run.log').open('w', encoding='utf-8') as log:
        child = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=log,
                                 stderr=subprocess.STDOUT, text=True)
        try:
            if os.name == 'nt':
                try:
                    close_job = windows_job(child, limit)
                except OSError as exc:
                    error = str(exc)
                    stop = 'memory_guard_setup_failed'
            if stop is None:
                child.stdin.write('start\n')
                child.stdin.close()
                process = psutil.Process(child.pid)
                while child.poll() is None:
                    group = [process, *process.children(recursive=True)]
                    rss = 0
                    for proc in group:
                        try:
                            rss += proc.memory_info().rss
                        except psutil.NoSuchProcess:
                            pass
                    peak = max(peak, rss)
                    if peak >= limit or perf_counter() >= deadline:
                        stop = 'rss_limit' if peak >= limit else 'wall_limit'
                        break
                    sleep(0.04)
        finally:
            if child.poll() is None:
                if close_job is not None:
                    close_job()
                else:
                    child.kill()
            elif close_job is not None:
                close_job()
            status = child.wait()
    guard = {'status': status, 'stop_reason': stop, 'peak_sampled_rss_bytes': peak,
             'wall_seconds': perf_counter()-start, 'rss_and_commit_limit_bytes': limit,
             'windows_job_commit_limit_enforced': close_job is not None, 'windows_job_error': error,
             'wall_limit_seconds': args.seconds, 'command': command}
    write_json(out/'solver_guard.json', guard)
    print(json.dumps(guard), flush=True)
    return status if not stop else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path)
    source.add_argument('--cached', type=Path, help='Re-solve an existing reduced cone in its saved row order')
    source.add_argument('--reconstruct', type=Path, help='Solve retained equations exactly and check every equality')
    parser.add_argument('--psd-margin', type=float, default=0.0, help='Require a positive scaled PSD margin in a cached solve')
    parser.add_argument('--face-manifest', type=Path, help='Face model and row transform for a cached solve')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('clarabel', 'scs', 'scs-indirect'), default='clarabel')
    parser.add_argument('--seconds', type=float, default=1780)
    parser.add_argument('--memory-mib', type=int, default=3900)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--eps', type=float, default=1e-9)
    parser.add_argument('--keep-equalities', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (25 <= args.seconds <= 1800 and 128 <= args.memory_mib < 4096
            and args.iterations > 0 and 0 < args.eps < 1):
        parser.error('guard limits: 25–1800 seconds and 128–4095 MiB')
    if not 0 <= args.psd_margin <= 1 or args.psd_margin and not args.cached:
        parser.error('--psd-margin must be in [0, 1] and requires --cached')
    return worker(args) if args.worker else supervise(args)


if __name__ == '__main__':
    sys.exit(main())
