"""Run a guarded numerical PSD-only SDP from a Round 7 CSC/svec archive.

The archive represents A x + s = b and objective q @ x. Each block uses
SCS lower-column order and sqrt(2) on off-diagonal entries. Saved x, y, s
are in the input coordinates, including when diagonal congruence is used.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
from math import isfinite
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, sleep

from solve_general_4_9 import THREADS, diagnostic_eigenvalues, windows_job, write_json
from solve_level3_4_9 import DEADLINE_ENV, clarabel_order


MAX_SECONDS = 900
MAX_MEMORY_MIB = 4096
SCS_ITERATIONS = 20_000
EPS = 1e-9
POLL_SECONDS = 0.05


def validate_congruence(scale, rows, sizes):
    import numpy as np

    if scale.shape != (rows,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('diagonal PSD congruence requires one positive finite scale per row')
    offset = 0
    for size in sizes:
        length = size * (size + 1) // 2
        block = scale[offset:offset + length]
        diagonal = np.arange(size)
        factors = np.sqrt(block[diagonal * size - diagonal * (diagonal - 1) // 2])
        col, row = np.triu_indices(size)
        if not np.allclose(block, factors[col] * factors[row], rtol=1e-10, atol=0):
            raise ValueError('row scaling is not a diagonal PSD congruence')
        offset += length


def load_conic(path, use_congruence):
    import numpy as np
    from scipy.sparse import csc_matrix

    def integers(archive, name):
        value = archive[name]
        if value.ndim != 1 or value.dtype.kind not in 'iu':
            raise ValueError(f'{name} must be an integer vector')
        return value

    def reals(archive, name):
        value = archive[name]
        if value.ndim != 1 or value.dtype.kind not in 'iuf':
            raise ValueError(f'{name} must be a real vector')
        value = value.astype(float, copy=False)
        if not np.isfinite(value).all():
            raise ValueError(f'{name} must contain finite values')
        return value

    with np.load(path, allow_pickle=False) as archive:
        shape = tuple(map(int, integers(archive, 'A_shape')))
        if len(shape) != 2 or min(shape) <= 0:
            raise ValueError('A_shape must specify positive row and column counts')
        sizes = list(map(int, integers(archive, 'block_sizes')))
        if not sizes or min(sizes) <= 0 or sum(s * (s + 1) // 2 for s in sizes) != shape[0]:
            raise ValueError('PSD row count must equal the sum of packed positive block sizes')
        b, q = (reals(archive, name) for name in ('b', 'q'))
        if (len(b), len(q)) != shape:
            raise ValueError('b and q dimensions do not match A_shape')
        data = reals(archive, 'A_data')
        indices, indptr = (integers(archive, name) for name in ('A_indices', 'A_indptr'))
        if (len(indptr) != shape[1] + 1 or indptr[0] != 0 or indptr[-1] != len(data)
                or len(indices) != len(data) or (indptr[1:] < indptr[:-1]).any()
                or (indices < 0).any() or (indices >= shape[0]).any()):
            raise ValueError('invalid CSC pointers or row indices')
        A = csc_matrix((data, indices, indptr), shape=shape)
        A.check_format(full_check=True)
        A.sum_duplicates()
        if not np.isfinite(A.data).all():
            raise ValueError('merged CSC entries must be finite')
        scale = None
        if use_congruence:
            scale = reals(archive, 'psd_congruence')
            validate_congruence(scale, shape[0], sizes)
        chart = {name: archive[name] for name in ('retained_chart_columns', 'full_chart_dimension')
                 if name in archive}
    return A, b, q, sizes, scale, chart


def backend_solve(backend, A, b, q, sizes, remaining, verbose, clarabel_linear_solver):
    if backend.startswith('scs'):
        import scs

        settings = dict(eps_abs=EPS, eps_rel=EPS, eps_infeas=EPS, max_iters=SCS_ITERATIONS,
                        time_limit_secs=remaining, verbose=verbose)
        selector = (scs.LinearSolver.CPU_INDIRECT if backend == 'scs-indirect'
                    else scs.LinearSolver.QDLDL)
        solver = scs.SCS({'A': A, 'b': b, 'c': q}, {'s': sizes},
                         linear_solver=selector, **settings)
        raw = solver.solve(warm_start=False)
        info = raw['info']
        status = {1: 'feasible', 2: 'feasible_inaccurate', -2: 'infeasible',
                  -7: 'infeasible_inaccurate'}.get(info['status_val'], 'unknown')
        return raw, info, status, {**settings, 'linear_solver': selector.name}

    import clarabel
    from scipy.sparse import csc_matrix

    settings = clarabel.DefaultSettings()
    settings.verbose = verbose
    settings.time_limit = remaining
    settings.max_threads = 1
    settings.max_iter = 200
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = EPS
    settings.tol_infeas_abs = settings.tol_infeas_rel = EPS
    settings.direct_solve_method = clarabel_linear_solver
    settings.chordal_decomposition_enable = False
    cones = [clarabel.PSDTriangleConeT(size) for size in sizes]
    solver = clarabel.DefaultSolver(csc_matrix((len(q), len(q))), q, A, b, cones, settings)
    solution = solver.solve()
    raw = {'x': solution.x, 'y': solution.z, 's': solution.s}
    solver_info = solver.get_info()
    info = {name: getattr(solver_info, name) for name in (
        'iterations', 'solve_time', 'cost_primal', 'cost_dual', 'res_primal', 'res_dual',
        'res_primal_inf', 'res_dual_inf', 'gap_abs', 'gap_rel', 'ktratio')}
    info['status'] = str(solution.status)
    info['linsolver'] = {name: getattr(solver_info.linsolver, name) for name in (
        'name', 'threads', 'direct', 'nnzA', 'nnzL')}
    status = {'Solved': 'feasible', 'AlmostSolved': 'feasible_inaccurate',
              'PrimalInfeasible': 'infeasible', 'AlmostPrimalInfeasible': 'infeasible_inaccurate'}.get(
                  info['status'], 'unknown')
    settings_dict = {name: getattr(settings, name) for name in (
        'verbose', 'time_limit', 'max_threads', 'max_iter', 'tol_gap_abs', 'tol_gap_rel',
        'tol_feas', 'tol_infeas_abs', 'tol_infeas_rel', 'direct_solve_method',
        'chordal_decomposition_enable')}
    return raw, info, status, settings_dict


def conic_diagnostics(A, b, q, cone, x, y, s, status):
    import numpy as np

    result = {'primal': None, 'dual': None, 'dual_infeasibility_ray': None}
    if np.isfinite(x).all() and np.isfinite(s).all():
        affine_slack = b - A @ x
        result['primal'] = {
            'linear_residual_inf': np.max(np.abs(s - affine_slack)),
            'slack_psd_min_eigenvalues': diagnostic_eigenvalues(s, cone),
            'affine_psd_min_eigenvalues': diagnostic_eigenvalues(affine_slack, cone),
            'objective': q @ x,
        }
    if np.isfinite(y).all():
        by = b @ y
        result['dual'] = {'stationarity_inf': np.max(np.abs(A.T @ y + q)),
                          'b_dot_y': by, 'objective': -by,
                          'psd_min_eigenvalues': diagnostic_eigenvalues(y, cone)}
        if status.startswith('infeasible') and by < 0:
            ray = y / -by
            result['dual_infeasibility_ray'] = {
                'stationarity_inf': np.max(np.abs(A.T @ ray)), 'b_dot_y': b @ ray,
                'psd_min_eigenvalues': diagnostic_eigenvalues(ray, cone),
            }
    return result


def worker(args):
    if sys.stdin.readline() != 'start\n':
        return 2
    enforced_address_space = None
    if os.name != 'nt':
        import resource

        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = args.memory_mib * 1024**2
        enforced_address_space = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (enforced_address_space, hard))
    print(f'Python worker PID {os.getpid()}, address-space cap {enforced_address_space}', flush=True)
    import numpy as np

    A, b, q, sizes, scale, chart = load_conic(args.input, args.psd_congruence)
    print(f'{args.backend}: A{A.shape} nnz={A.nnz}, PSD blocks {sizes}', flush=True)
    cone = {'z': 0, 's': sizes}
    order = clarabel_order(cone) if args.backend == 'clarabel' else np.arange(A.shape[0])
    if scale is None:
        backend_A, backend_b = A, b
    else:
        backend_A = A.copy()
        backend_A.data *= scale[backend_A.indices]
        backend_b = b * scale
        if not (np.isfinite(backend_A.data).all() and np.isfinite(backend_b).all()):
            raise ValueError('diagonal congruence caused overflow')
    if args.backend == 'clarabel':
        backend_A, backend_b = backend_A[order, :].tocsc(), backend_b[order]
    remaining = float(os.environ[DEADLINE_ENV]) - perf_counter() - min(10, args.seconds * 0.05)
    if remaining <= 0:
        raise TimeoutError('no remaining solver time after loading and converting the archive')
    raw, info, status, settings = backend_solve(args.backend, backend_A, backend_b, q, sizes,
                                                remaining, not args.quiet, args.clarabel_linear_solver)
    x, y_backend, s_backend = (np.asarray(raw[name], dtype=float) for name in ('x', 'y', 's'))
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    y, s = y_backend[inverse], s_backend[inverse]
    if scale is not None:
        y *= scale
        s /= scale
    vectors = args.output/'solver_primal_dual.npz'
    temporary = vectors.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, x=x, y=y, s=s, y_backend=y_backend, s_backend=s_backend,
                        backend_order=order, block_sizes=sizes, **chart)
    temporary.replace(vectors)
    print(f'{args.backend}: {info["status"]}; saved raw and input-order vectors', flush=True)
    with args.input.open('rb') as source:
        sha256 = hashlib.file_digest(source, 'sha256').hexdigest()
    summary = {'backend': args.backend, 'raw_info': info, 'settings': settings,
               'versions': {name: version(name) for name in ('numpy', 'scipy',
                            'clarabel' if args.backend == 'clarabel' else 'scs')},
               'input': str(args.input), 'input_sha256': sha256, 'numerical_status': status,
               'psd_blocks': sizes, 'A_shape': A.shape, 'A_nnz': A.nnz,
               'input_order': 'scs_lower_column', 'psd_congruence_applied': scale is not None,
               'diagnostics_complete': False}
    write_json(args.output/'solver_summary.json', summary)
    summary['diagnostics'] = conic_diagnostics(A, b, q, cone, x, y, s, status)
    summary['diagnostics_complete'] = True
    write_json(args.output/'solver_summary.json', summary)
    return 0


def supervise(args):
    import psutil

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob('solver_*')):
        raise FileExistsError(f'Solver output exists in {out}; select a new output directory')
    command = [sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:], '--worker']
    limit = args.memory_mib * 1024**2
    begin = perf_counter()
    deadline = begin + args.seconds
    env = dict(os.environ, PYTHONUNBUFFERED='1', **{name: '1' for name in THREADS})
    env[DEADLINE_ENV] = str(deadline)
    if os.name == 'nt' and sys.prefix != sys.base_prefix:
        # Run the interpreter directly while retaining the venv's package paths.
        # The venv redirector can create a child before assignment to the Job.
        command[0] = sys._base_executable
        env['__PYVENV_LAUNCHER__'] = sys.executable
    peak = 0
    reason = guard_error = close_job = None
    enforced = False
    with (out/'solver_run.log').open('x', encoding='utf-8') as log:
        child = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=log,
                                 stderr=subprocess.STDOUT, text=True)
        print(f'{args.backend} worker PID {child.pid}', flush=True)
        try:
            process = psutil.Process(child.pid)
            if os.name == 'nt':
                close_job = windows_job(child, limit)
                enforced = True
            else:
                import resource

                if not hasattr(resource, 'RLIMIT_AS'):
                    raise OSError('RLIMIT_AS is unavailable')
            child.stdin.write('start\n')
            child.stdin.close()
            while child.poll() is None:
                group = [process, *process.children(recursive=True)]
                rss = 0
                for item in group:
                    try:
                        rss += item.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak = max(peak, rss)
                remaining = deadline - perf_counter()
                if rss >= limit or remaining <= 0:
                    reason = 'rss_limit' if rss >= limit else 'wall_limit'
                    for item in reversed(group):
                        try:
                            item.kill()
                        except psutil.NoSuchProcess:
                            pass
                    break
                sleep(min(POLL_SECONDS, remaining))
        except psutil.NoSuchProcess:
            pass
        except (OSError, psutil.Error) as error:
            reason, guard_error = 'guard_error', str(error)
        finally:
            if child.poll() is None:
                child.kill()
            returncode = child.wait()
            if close_job is not None:
                close_job()
    if os.name != 'nt':
        with (out/'solver_run.log').open(encoding='utf-8', errors='replace') as log:
            prefix = f'Python worker PID {child.pid}, address-space cap '
            line = log.readline(256).strip()
            enforced = line.startswith(prefix) and line[len(prefix):].isdigit()
    guard = {'command': command, 'worker_pid': child.pid, 'exit_code': returncode,
             'kill_reason': reason, 'guard_error': guard_error, 'wall_seconds': perf_counter() - begin,
             'peak_sampled_rss_bytes': peak, 'wall_limit_seconds': args.seconds,
             'memory_limit_bytes': limit, 'rss_poll_max_seconds': POLL_SECONDS,
             'hard_memory_limit_enforced': enforced,
             'windows_job_commit_limit_enforced': close_job is not None}
    write_json(out/'solver_guard.json', guard)
    print(json.dumps(guard), flush=True)
    return 1 if reason is not None else returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--backend', choices=('clarabel', 'scs', 'scs-indirect'), default='clarabel')
    parser.add_argument('--seconds', type=float, default=MAX_SECONDS)
    parser.add_argument('--memory-mib', type=int, default=MAX_MEMORY_MIB)
    parser.add_argument('--psd-congruence', action='store_true', help='Apply the archive scaling vector')
    parser.add_argument('--clarabel-linear-solver', choices=('qdldl', 'faer'), default='qdldl')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose backend solver output')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (isfinite(args.seconds) and 0 < args.seconds <= MAX_SECONDS
            and 16 <= args.memory_mib <= MAX_MEMORY_MIB):
        parser.error('limits are 0 < seconds <= 900 and 16 <= memory-mib <= 4096')
    return worker(args) if args.worker else supervise(args)


if __name__ == '__main__':
    sys.exit(main())
