"""Guarded SCS solve for the anchored S8-reduced complex ETF(4,9) moment relaxation."""

import argparse
import json
import math
import os
from numbers import Integral, Real
from pathlib import Path
import subprocess
import sys
from time import perf_counter, sleep

from search_sdp_cert import desvec


MAX_SECONDS = 1800
MAX_MEMORY_MIB = 2048
THREADS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
SCOPE = ('Full order-four anchored moment conditions for an arbitrary ETF(4,9), averaged '
         'over S8 on the eight nonanchor lines and over conjugation. For S8-invariant moments, '
         'the complete degree-two PSD test is equivalent to its restriction to the tail-S4 '
         'fixed subspace, split into eight characters. This scope concerns the relaxation.')


def write_json(path, value):
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if isinstance(item, bool) or item is None or isinstance(item, str):
            return item
        if isinstance(item, Integral):
            return int(item)
        if isinstance(item, Real):
            return float(item) if math.isfinite(item) else None
        raise TypeError(type(item))

    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def pack_model(model):
    return {
        **model,
        'scope': SCOPE,
        'representation': 'S8 orbits and conjugation fixed moments, first Gram row anchored at +mu',
        'normalization': 'moment id 0 is the constant and equals 1',
        'maximum_monomial_degree': 4,
        'svec': 'Lower triangle, column-major, off-diagonal scaled by sqrt(2)',
        'conic_signs': 'A x + s = b; rows first normalization, next exact equalities, then negative PSD svec maps',
        'base_equations': {
            name: [[monomial, coefficient] for monomial, coefficient in sorted(eq.items())]
            for name, eq in model['base_equations'].items()
        },
    }


def augment_equalities(model, overlay):
    """Add exact homogeneous rows in the model's existing moment order."""
    if not isinstance(overlay, dict) or not isinstance(overlay.get('rows'), list):
        raise ValueError('extra equalities must be a JSON object with a rows list')
    source_rows = overlay['rows']
    witnesses = overlay.get('witnesses', [None] * len(source_rows))
    if not isinstance(witnesses, list) or len(witnesses) != len(source_rows):
        raise ValueError('extra witnesses must match extra rows one-to-one')
    rows = []
    for row in source_rows:
        if not isinstance(row, dict):
            raise ValueError('each extra row must be a moment-id mapping')
        parsed = {}
        for index, coefficient in row.items():
            column = int(index)
            if str(column) != index or not 0 <= column < len(model['moments']):
                raise ValueError(f'invalid extra equality moment index {index!r}')
            if type(coefficient) is int:
                coefficient = [coefficient, 0]
            if (not isinstance(coefficient, list) or len(coefficient) != 2
                    or any(type(c) is not int for c in coefficient)):
                raise ValueError(f'invalid exact coefficient for moment index {index!r}')
            if any(coefficient):
                parsed[column] = tuple(coefficient)
        if not parsed:
            raise ValueError('extra equality row is identically zero')
        rows.append(parsed)
    return {
        **model,
        'equalities': [*model['equalities'], *rows],
        'equality_witnesses': [*model['equality_witnesses'], *witnesses],
        'extra_equality_count': len(rows),
        'extra_equality_scope': overlay.get('scope', 'Caller-provided homogeneous moment identities'),
    }


def conic_data(model):
    import numpy as np
    from scipy.sparse import coo_matrix

    n = len(model['moments'])
    if not model['moments'] or model['moments'][0] != ():
        raise ValueError('constant moment must be first')
    if tuple(model['coefficient_basis']) != ('1', 'sqrt(10)/5'):
        raise ValueError('unsupported exact coefficient basis')
    beta = math.sqrt(10) / 5
    rows, cols, values = [0], [0], [1.0]
    for row, equation in enumerate(model['equalities'], start=1):
        for col, (rational, radical) in equation.items():
            rows.append(row)
            cols.append(col)
            values.append(rational + beta * radical)
    n_equal = 1 + len(model['equalities'])
    offset = n_equal
    sizes = [len(block['basis']) for block in model['psd_blocks']]
    if not sizes or min(sizes) <= 0 or max(sizes) > 600:
        raise ValueError(f'unsupported PSD sizes {sizes}')
    for block, size in zip(model['psd_blocks'], sizes, strict=True):
        for i, j, entry in block['entries']:
            if not 0 <= j <= i < size:
                raise ValueError('PSD entry outside lower triangle')
            row = offset + j * size - j * (j - 1) // 2 + i - j
            factor = -1.0 if i == j else -math.sqrt(2)
            for col, coefficient in entry.items():
                rows.append(row)
                cols.append(col)
                values.append(factor * coefficient)
        offset += size * (size + 1) // 2
    if len(values) > 3_000_000:
        raise ValueError('too many sparse coefficients for the configured solver')
    matrix = coo_matrix((values, (rows, cols)), shape=(offset, n)).tocsc()
    b = np.zeros(offset)
    b[0] = 1
    return matrix, b, np.zeros(n), {'z': n_equal, 's': sizes}


def psd_blocks(vector, cone):
    offset = cone['z']
    for size in cone['s']:
        length = size * (size + 1) // 2
        yield desvec(vector[offset:offset + length], size)
        offset += length


def diagnostic_eigenvalues(vector, cone):
    import numpy as np

    return [float(np.linalg.eigvalsh(block)[0]) for block in psd_blocks(vector, cone)]


def diagnostics(A, b, c, cone, solution):
    import numpy as np

    x, y, s = (np.asarray(solution[k], dtype=float) for k in ('x', 'y', 's'))
    result = {'primal': None, 'dual': None, 'dual_infeasibility_ray': None}
    if np.isfinite(x).all() and np.isfinite(s).all():
        result['primal'] = {
            'linear_residual_inf': np.linalg.norm(A @ x + s - b, ord=np.inf),
            'equality_residual_inf': np.linalg.norm((A @ x - b)[:cone['z']], ord=np.inf),
            'slack_psd_min_eigenvalues': diagnostic_eigenvalues(s, cone),
            'moment_psd_min_eigenvalues': diagnostic_eigenvalues(b - A @ x, cone),
            'objective': c @ x,
        }
    if np.isfinite(y).all():
        by = b @ y
        result['dual'] = {
            'stationarity_inf': np.linalg.norm(A.T @ y + c, ord=np.inf),
            'psd_min_eigenvalues': diagnostic_eigenvalues(y, cone),
            'b_dot_y': by,
            'objective': -by,
        }
        if by < 0:
            ray = y / -by
            result['dual_infeasibility_ray'] = {
                'b_dot_y': b @ ray,
                'stationarity_inf': np.linalg.norm(A.T @ ray, ord=np.inf),
                'psd_min_eigenvalues': diagnostic_eigenvalues(ray, cone),
            }
    return result


def exact_primal_candidate(model, x, output):
    """Fit free moments to small rationals and solve remaining equalities over Q(beta)."""
    import numpy as np
    import sympy as sp
    from sympy.polys.matrices import DomainMatrix

    if not np.isfinite(x).all():
        return {'status': 'no_finite_primal'}
    beta = sp.sqrt(10) / 5
    field = sp.QQ.algebraic_field(beta)
    zero, one = field.zero, field.one
    beta_field = field.from_sympy(beta)
    n = len(model['moments'])
    rows = {0: {0: one, n: one}}
    for i, equality in enumerate(model['equalities'], start=1):
        rows[i] = {j: field(a) + field(b) * beta_field for j, (a, b) in equality.items()}
    rref, pivots = DomainMatrix(rows, (len(rows), n + 1), field).rref()
    if n in pivots:
        return {'status': 'exact_linear_system_inconsistent'}
    free = sorted(set(range(n)) - set(pivots))
    data = rref.to_dok()
    exact = [zero] * n
    for j in free:
        exact[j] = field.from_sympy(sp.Rational(float(x[j])).limit_denominator(10**7))
    for i, j in enumerate(pivots):
        exact[j] = data.get((i, n), zero) - sum(
            (data.get((i, k), zero) * exact[k] for k in free), zero)
    eq_passed = exact[0] == one and all(
        sum(((field(a) + field(b) * beta_field) * exact[j] for j, (a, b) in row.items()), zero) == zero
        for row in model['equalities'])
    if not eq_passed:
        raise ArithmeticError('exact equality reconstruction failed self-check')
    terms = []
    for item in exact:
        coeff = [0] * (2 - len(item.to_list())) + item.to_list()
        terms.append([str(coeff[1]), str(coeff[0])])
    exact_float = np.array([float(field.to_sympy(v).evalf(20)) for v in exact])
    residual = float(np.max(np.abs(exact_float - x)))
    minima = []
    for block in model['psd_blocks']:
        numeric = np.zeros((len(block['basis']), len(block['basis'])))
        for i, j, entry in block['entries']:
            numeric[i, j] = numeric[j, i] = sum(coef * exact_float[k] for k, coef in entry.items())
        minima.append(float(np.linalg.eigvalsh(numeric)[0]))
    write_json(output / 'anchored_exact_primal_candidate.json', {
        'schema': 'anchored-exact-primal-v1',
        'role': 'Reconstructed relaxation moment vector for independent verification',
        'coefficient_basis': list(model['coefficient_basis']), 'moments': terms,
        'exact_equalities_passed': eq_passed, 'psd_status': 'numeric_screen_only',
        'max_coordinate_change': residual, 'psd_min_eigenvalues_numeric': minima,
    })
    return {'status': 'exact_equality_candidate_saved', 'free_moments': len(free),
            'max_coordinate_change': residual, 'psd_min_eigenvalues_numeric': minima}


def worker(args):
    if sys.stdin.readline() != 'start\n':
        return 2
    start = perf_counter()
    import numpy as np
    import scs
    from general_4_9_model import build_model

    model = build_model(split=True)
    if args.extra_equalities is not None:
        overlay = json.loads(args.extra_equalities.read_text(encoding='utf-8-sig'))
        model = augment_equalities(model, overlay)
    A, b, c, cone = conic_data(model)
    out = args.output
    write_json(out / 'anchored_model.json', pack_model(model))
    np.savez_compressed(out / 'anchored_conic_data.npz', data=A.data, indices=A.indices,
                       indptr=A.indptr, shape=A.shape, b=b, c=c,
                       zero_cone=cone['z'], psd_cones=cone['s'])
    print(f'A shape {A.shape}, nnz={A.nnz}, cone={cone}', flush=True)
    remaining = args.seconds - (perf_counter() - start) - 3
    if remaining <= 0:
        raise TimeoutError('builder used available solver time')
    settings = {'eps_abs': args.eps, 'eps_rel': args.eps, 'eps_infeas': args.eps / 10,
                'max_iters': args.iterations, 'time_limit_secs': remaining,
                'verbose': args.verbose, 'normalize': True}
    if hasattr(scs, 'LinearSolver'):
        backend = scs.LinearSolver.CPU_INDIRECT if args.indirect else scs.LinearSolver.QDLDL
        solver = {'linear_solver': backend}
    else:
        solver = {'use_indirect': args.indirect}
    solution = scs.solve({'A': A, 'b': b, 'c': c}, cone, **settings, **solver)
    arrays = {k: np.asarray(solution[k]) for k in ('x', 'y', 's')}
    np.savez_compressed(out / 'solver_primal_dual.npz', **arrays, equality_dual=arrays['y'][:cone['z']],
                       **{f'dual_psd_{i}': block for i, block in enumerate(psd_blocks(arrays['y'], cone))},
                       **{f'primal_slack_psd_{i}': block for i, block in enumerate(psd_blocks(arrays['s'], cone))})
    diag = diagnostics(A, b, c, cone, arrays)
    info = solution['info']
    if info['status_val'] in (-2, -7) and diag['dual_infeasibility_ray'] is not None:
        ray = arrays['y'] / -(b @ arrays['y'])
        np.savez_compressed(out / 'solver_normalized_dual_ray.npz', y=ray,
                           equality_dual=ray[:cone['z']],
                           **{f'dual_psd_{i}': block for i, block in enumerate(psd_blocks(ray, cone))})
    summary = {
        'scope': SCOPE, 'status': info['status'], 'raw_scs_info': info, 'diagnostics': diag,
        'sizes': {'rows': A.shape[0], 'moments': A.shape[1], 'nnz': A.nnz, **cone},
        'extra_equality_count': model.get('extra_equality_count', 0),
        'scs_version': scs.__version__, 'backend': 'indirect' if args.indirect else 'direct',
        'settings': settings, 'threads': {name: os.environ.get(name) for name in THREADS},
        'wall_seconds_worker': perf_counter() - start,
    }
    write_json(out / 'solver_summary.json', summary)
    print(f'SCS returned {info["status"]}, iter={info["iter"]}', flush=True)
    if not args.no_recover and info['status_val'] in (1, 2):
        summary['reconstruction'] = exact_primal_candidate(model, arrays['x'], out)
    elif (info['status_val'] in (-2, -7) and diag['dual_infeasibility_ray'] is not None):
        summary['reconstruction'] = {'status': 'numeric_dual_ray_saved_exact_verification_pending'}
    else:
        summary['reconstruction'] = {'status': 'skipped'}
    summary['wall_seconds_worker'] = perf_counter() - start
    write_json(out / 'solver_summary.json', summary)
    return 0


def windows_job(child, limit):
    """Set a kernel job commit cap before allowing the worker to build the model."""
    import ctypes as ct
    from ctypes import wintypes as wt

    class BasicLimits(ct.Structure):
        _fields_ = [('per_process_time', ct.c_longlong), ('per_job_time', ct.c_longlong),
                    ('flags', wt.DWORD), ('min_ws', ct.c_size_t), ('max_ws', ct.c_size_t),
                    ('active_processes', wt.DWORD), ('affinity', ct.c_size_t),
                    ('priority', wt.DWORD), ('scheduling', wt.DWORD)]

    class ExtendedLimits(ct.Structure):
        _fields_ = [('basic', BasicLimits), ('io', ct.c_ulonglong * 6),
                    ('process_memory', ct.c_size_t), ('job_memory', ct.c_size_t),
                    ('peak_process_memory', ct.c_size_t), ('peak_job_memory', ct.c_size_t)]

    kernel = ct.WinDLL('kernel32', use_last_error=True)
    kernel.CreateJobObjectW.argtypes = (ct.c_void_p, wt.LPCWSTR)
    kernel.CreateJobObjectW.restype = wt.HANDLE
    kernel.SetInformationJobObject.argtypes = (wt.HANDLE, ct.c_int, ct.c_void_p, wt.DWORD)
    kernel.SetInformationJobObject.restype = wt.BOOL
    kernel.AssignProcessToJobObject.argtypes = (wt.HANDLE, wt.HANDLE)
    kernel.AssignProcessToJobObject.restype = wt.BOOL
    kernel.CloseHandle.argtypes = (wt.HANDLE,)
    kernel.CloseHandle.restype = wt.BOOL
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ct.WinError(ct.get_last_error())
    caps = ExtendedLimits()
    caps.basic.flags = 0x100 | 0x200 | 0x2000  # process/job memory cap and kill-on-close
    caps.process_memory = caps.job_memory = limit
    if (not kernel.SetInformationJobObject(job, 9, ct.byref(caps), ct.sizeof(caps))
            or not kernel.AssignProcessToJobObject(job, int(child._handle))):
        error = ct.WinError(ct.get_last_error())
        kernel.CloseHandle(job)
        raise error
    return lambda: kernel.CloseHandle(job)


def supervise(args):
    import psutil

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    existing = list(out.glob('solver_*')) + list(out.glob('anchored_*'))
    if existing:
        raise FileExistsError(f'Solver output exists in {out}; select a new output directory')
    command = [sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:], '--worker']
    env = dict(os.environ, PYTHONUNBUFFERED='1', **{name: '1' for name in THREADS})
    begin = perf_counter()
    limit = args.memory_mib * 1024**2
    peak = 0
    reason = None
    close_job = None
    job_error = None
    with (out / 'solver_run.log').open('w', encoding='utf-8') as log:
        child = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=log,
                                 stderr=subprocess.STDOUT, text=True)
        try:
            process = psutil.Process(child.pid)
            if os.name == 'nt':
                try:
                    close_job = windows_job(child, limit)
                except OSError as error:
                    job_error = str(error)
            child.stdin.write('start\n')
            child.stdin.close()
            while child.poll() is None:
                group = [process]
                try:
                    group.extend(process.children(recursive=True))
                    rss = sum(item.memory_info().rss for item in group if item.is_running())
                    peak = max(peak, rss)
                except psutil.NoSuchProcess:
                    continue
                elapsed = perf_counter() - begin
                if peak >= limit or elapsed >= args.seconds:
                    reason = 'rss_limit' if peak >= limit else 'wall_limit'
                    for item in reversed(group):
                        try:
                            item.kill()
                        except psutil.NoSuchProcess:
                            pass
                    break
                sleep(min(0.05, args.seconds - elapsed))
        finally:
            if child.poll() is None and reason is None:
                child.kill()
            returncode = child.wait()
            if close_job is not None:
                close_job()
    guard = {
        'command': command, 'exit_code': returncode, 'kill_reason': reason,
        'wall_seconds': perf_counter() - begin, 'peak_sampled_rss_bytes': peak,
        'wall_limit_seconds': args.seconds, 'memory_limit_bytes': limit,
        'windows_job_commit_limit_enforced': close_job is not None, 'windows_job_error': job_error,
        'rss_poll_max_seconds': 0.05,
    }
    write_json(out / 'solver_guard.json', guard)
    print(json.dumps(guard, ensure_ascii=True), flush=True)
    return 1 if reason is not None else returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--extra-equalities', type=Path, help='JSON exact rows and optional witnesses')
    parser.add_argument('--seconds', type=float, default=MAX_SECONDS)
    parser.add_argument('--memory-mib', type=int, default=MAX_MEMORY_MIB)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--iterations', type=int, default=50_000)
    parser.add_argument('--indirect', action='store_true')
    parser.add_argument('--no-recover', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (0 < args.seconds <= MAX_SECONDS and 16 <= args.memory_mib <= MAX_MEMORY_MIB
            and 0 < args.eps < 1 and args.iterations > 0):
        parser.error('Use 0 < seconds <= 1800, 16 <= memory-mib <= 2048, 0 < eps < 1 and positive iterations')
    return worker(args) if args.worker else supervise(args)


if __name__ == '__main__':
    sys.exit(main())
