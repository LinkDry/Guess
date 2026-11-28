"""Stream the common PSD margin problem for a blockwise affine pencil.

Each block stores A, b, scale in SCS svec order. The unscaled slack is
b + A @ x. The barrier factors the scaled slack minus t * I and minimizes
-t - mu * logdet. Outputs x, y, and s use the original pencil coordinates.
"""
import argparse
from contextlib import ExitStack, contextmanager
import json
from math import fsum, isfinite, sqrt
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter, sleep
from zipfile import ZipFile


MAX_SECONDS = 1800
MAX_MEMORY_MIB = 4095
DEFAULT_MEMORY_MIB = 3584
POLL_SECONDS = 0.05
SQRT2 = sqrt(2.)
PRIMAL_MARGIN = 1e-9
DUAL_RATIO = 1e-4
QR_PANEL_BYTES = 16 * 1024**2
BATCH_BYTES = 32 * 1024**2
DEADLINE_ENV = 'ETF_SUPPORT5_WALL_DEADLINE'


def windows_job(child, limit):
    from solve_general_4_9 import windows_job as create_job

    return create_job(child, limit)


class PackedBlock:
    def __init__(self, size):
        import numpy as np

        self.size = size
        self.column, self.row = np.triu_indices(size)
        self.off_diagonal = self.row != self.column
        self.diagonal = np.flatnonzero(~self.off_diagonal)
        self.length = len(self.row)

    def unpack(self, packed):
        import numpy as np

        values = np.array(packed, dtype=float, copy=True)
        values[..., self.off_diagonal] /= SQRT2
        block = np.empty((*values.shape[:-1], self.size, self.size))
        block[..., self.row, self.column] = block[..., self.column, self.row] = values
        return block

    def pack(self, block):
        packed = block[..., self.row, self.column]
        packed[..., self.off_diagonal] *= SQRT2
        return packed


class Pencil:
    """Spool the dense maps to disk with bounded I/O, then map one block per pass."""

    def __init__(self, source, scratch):
        self.source = source
        self.scratch = scratch
        manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8-sig'))
        self.rank = manifest['free_dimension']
        self.sizes = manifest['block_sizes']
        if (type(self.rank) is not int or self.rank < 0 or not isinstance(self.sizes, list)
                or not self.sizes or any(type(n) is not int or n < 1 for n in self.sizes)):
            raise ValueError('manifest requires nonnegative free_dimension and positive block_sizes')
        start = 0 if (source/'block_0.npz').exists() else 1
        self.archives = [source/f'block_{k}.npz' for k in range(start, start + len(self.sizes))]
        self.blocks = [PackedBlock(n) for n in self.sizes]
        self.cleanup = ExitStack()
        self.constants = []

    def __enter__(self):
        import numpy as np
        from solve_level3_lmi_4_9 import validate_congruence

        self.scratch.mkdir()
        self.cleanup.callback(self.scratch.rmdir)
        try:
            for k, (path, block) in enumerate(zip(self.archives, self.blocks, strict=True)):
                with np.load(path, allow_pickle=False) as archive:
                    b, scale = (archive[name] for name in ('b', 'scale'))
                if (b.shape != (block.length,) or b.dtype.kind not in 'iuf'
                        or not np.isfinite(b).all()):
                    raise ValueError(f'{path.name}: b must be a finite packed real vector')
                if scale.dtype.kind not in 'iuf':
                    raise ValueError(f'{path.name}: scale must be a packed real vector')
                b, scale = b.astype(float, copy=False), scale.astype(float, copy=False)
                validate_congruence(scale, block.length, [block.size])
                self.constants.append((b, scale))
                # Never materialize a compressed archive's dense A in RAM.
                with ZipFile(path) as archive, archive.open('A.npy') as src:
                    cache = self.scratch/f'A_{k}.npy'
                    self.cleanup.callback(cache.unlink, missing_ok=True)
                    with cache.open('xb') as dst:
                        shutil.copyfileobj(src, dst, length=1024**2)
                with self.read_map(k) as a:
                    if a.shape != (block.length, self.rank) or a.dtype.kind not in 'iuf':
                        raise ValueError(f'{path.name}: A must be a real ({block.length}, {self.rank}) map')
        except BaseException:
            self.cleanup.close()
            raise
        return self

    def __exit__(self, *exc):
        self.cleanup.close()

    @contextmanager
    def read_map(self, k):
        import numpy as np

        a = np.load(self.scratch/f'A_{k}.npy', mmap_mode='r', allow_pickle=False)
        try:
            yield a
        finally:
            # np.memmap has no public close method. Each caller finishes reading here.
            a._mmap.close()


def condition_pencil(pencil, check=lambda: None):
    """TSQR on scaled [A, b]. x = R^-1 z gives orthonormal pencil directions."""
    import numpy as np
    from scipy.linalg import qr, solve_triangular

    width = pencil.rank + 1
    panel_rows = max(1, QR_PANEL_BYTES // (8 * width))
    upper = np.empty((0, width))
    for k, (b, scale) in enumerate(pencil.constants):
        with pencil.read_map(k) as a:
            for lo in range(0, len(b), panel_rows):
                check()
                sl = slice(lo, lo+panel_rows)
                panel = np.empty((len(b[sl]), width))
                panel[:, :-1] = a[sl]
                panel[:, -1] = b[sl]
                panel *= scale[sl, None]
                if not np.isfinite(panel).all():
                    raise ValueError(f'{pencil.archives[k].name}: nonfinite scaled pencil data')
                # 'r' avoids building the tall orthogonal factor.
                local, = qr(panel, mode='r', overwrite_a=True, check_finite=False)
                merged, = qr(np.vstack((upper, local[:width, :])), mode='r',
                             overwrite_a=True, check_finite=False)
                upper = merged[:width, :].copy()
    r = upper[:pencil.rank, :-1].copy()
    if pencil.rank:
        diagonal = abs(r.diagonal())
        if (r.shape != (pencil.rank, pencil.rank) or not np.isfinite(r).all()
                or diagonal.min() <= np.finfo(float).eps * width * diagonal.max()):
            raise ValueError('pencil directions are numerically rank deficient')
    center = solve_triangular(r, -upper[:pencil.rank, -1], check_finite=False)
    return center, r


def block_derivatives(block, a, scale, lower, r, batch, check=lambda: None):
    """Compute one block's gradient and Hessian of -logdet in (z, t)."""
    import numpy as np
    from scipy.linalg import solve_triangular
    from scipy.linalg.blas import dsyrk

    size, rank = block.size, a.shape[1]
    inverse = solve_triangular(lower, np.eye(size), lower=True, check_finite=False)
    # Each direction is stored just once in the packed Gram operand. The
    # n-by-n templates and whitening products are bounded by the batch size.
    whitened = np.empty((rank+1, block.length))
    batch = min(batch, max(1, BATCH_BYTES // (8 * 4 * size**2)))
    for lo in range(0, rank, batch):
        check()
        hi = min(rank, lo+batch)
        templates = block.unpack(a[:, lo:hi].T * scale)
        whitened[lo:hi] = block.pack((inverse @ templates) @ inverse.T)
        del templates
    whitened[-1] = block.pack(-(inverse @ inverse.T))
    # Apply R^-T before forming the Gram, avoiding squared conditioning from
    # forming the native-coordinate Hessian. Limit the temporary row panel.
    panel = max(1, QR_PANEL_BYTES // (8 * (rank+1)))
    for lo in range(0, block.length, panel):
        check()
        sl = slice(lo, lo+panel)
        if rank:
            whitened[:-1, sl] = solve_triangular(r, whitened[:-1, sl], trans='T',
                                                check_finite=False)
    gradient = -whitened[:, block.diagonal].sum(axis=1)
    triangle = dsyrk(1., whitened.T, trans=1)
    return gradient, triangle + np.triu(triangle, 1).T


def mapped_product(a, x):
    """Use float64 without asking BLAS to cast a whole file-backed map."""
    import numpy as np

    if a.dtype == np.dtype('float64'):
        return a @ x
    rows = max(1, QR_PANEL_BYTES // (8 * max(1, a.shape[1])))
    result = np.empty(len(a))
    for lo in range(0, len(a), rows):
        result[lo:lo+rows] = np.asarray(a[lo:lo+rows], dtype=float) @ x
    return result


def slacks(pencil, x, check):
    for k, (block, (b, scale)) in enumerate(zip(pencil.blocks, pencil.constants, strict=True)):
        check()
        with pencil.read_map(k) as a:
            yield block.unpack((b + mapped_product(a, x)) * scale)


def factor_slacks(matrices, margin):
    from scipy.linalg import cholesky

    factors = []
    for matrix in matrices:
        shifted = matrix.copy()
        shifted.flat[::len(matrix)+1] -= margin
        factors.append(cholesky(shifted, lower=True, check_finite=False, overwrite_a=True))
    return factors


def line_search(matrices, changes, factors, margin, dt, mu, decrement, check):
    """Test candidates using affine slack steps, without rereading the pencil."""
    import numpy as np

    for power in range(40):
        check()
        weight = 0.5**power
        try:
            trial = factor_slacks((v + weight*dv for v, dv in zip(matrices, changes, strict=True)),
                                  margin + weight*dt)
        except np.linalg.LinAlgError:
            continue
        log_change = fsum(np.log(new.diagonal()/old.diagonal()).sum()
                          for old, new in zip(factors, trial, strict=True))
        if -weight*dt - 2*mu*log_change <= -0.01*weight*decrement + 1e-16:
            return weight
    return None


def checkpoint(pencil, x, t, mu, output, started, stage_reason, total_steps):
    """Report primal/dual spectral and residual tests in both scalings."""
    import numpy as np
    from scipy.linalg import cho_solve, eigvalsh
    from solve_general_4_9 import write_json

    def spectrum(matrix):
        vals = eigvalsh(matrix, check_finite=True)
        tolerance = 16*np.finfo(float).eps*len(matrix)*max(abs(vals[0]), abs(vals[-1]))
        return float(vals[0]), float(tolerance)

    p_unscaled, p_scaled, d_unscaled, d_scaled = [], [], [], []
    s_parts, y_parts = [], []
    b_dot_y, trace, complementarity = [], [], []
    stationarity, stationarity_scale = np.zeros(pencil.rank), np.zeros(pencil.rank)
    for k, (block, (b, scale)) in enumerate(zip(pencil.blocks, pencil.constants, strict=True)):
        with pencil.read_map(k) as a:
            s = b + mapped_product(a, x)
            scaled_slack = block.unpack(scale * s)
            factor, = factor_slacks([scaled_slack], t)
            dual_scaled = mu * cho_solve((factor, True), np.eye(block.size), check_finite=False)
            y = scale * block.pack(dual_scaled)
            p_unscaled.append(spectrum(block.unpack(s)))
            p_scaled.append(spectrum(scaled_slack))
            d_unscaled.append(spectrum(block.unpack(y)))
            d_scaled.append(spectrum(dual_scaled))
            stride = max(1, QR_PANEL_BYTES // (8 * max(1, pencil.rank)))
            for lo in range(0, len(b), stride):
                sl = slice(lo, lo+stride)
                part = np.asarray(a[sl], dtype=float)
                stationarity += part.T @ y[sl]
                stationarity_scale += abs(part).T @ abs(y[sl])
            b_dot_y.append(float(b @ y))
            trace.append(float(np.trace(dual_scaled)))
            complementarity.append(float(s @ y))
            s_parts.append(s)
            y_parts.append(y)
    objective, dual_trace = fsum(b_dot_y), fsum(trace)
    residual = float(np.max(abs(stationarity), initial=0.))
    ratio = residual / -objective if objective < 0 else None
    relative_residual = float(np.max(np.divide(abs(stationarity), stationarity_scale,
                                               out=np.zeros_like(stationarity),
                                               where=stationarity_scale > 0), initial=0.))
    primal_pass = (all(v > tol for v, tol in p_unscaled)
                   and all(v > max(tol, PRIMAL_MARGIN) for v, tol in p_scaled))
    dual_psd = all(v >= -tol for v, tol in d_unscaled + d_scaled)
    dual_pass = (dual_psd and abs(dual_trace - 1) < DUAL_RATIO and objective < -PRIMAL_MARGIN
                 and ratio < DUAL_RATIO and relative_residual < DUAL_RATIO)
    status = 'primal_candidate' if primal_pass else 'dual_candidate' if dual_pass else 'inconclusive'
    summary = {
        'status': status, 'termination': stage_reason, 'newton_steps': total_steps,
        'mu': mu, 'primal_scaled_min_margin': float(t), 'dual_b_dot_y': objective,
        'dual_scaled_trace': dual_trace, 'dual_scaled_trace_residual': abs(dual_trace - 1),
        'dual_stationarity_inf_unscaled': residual,
        'dual_stationarity_inf_relative_to_negative_objective': ratio,
        'dual_stationarity_max_componentwise_relative': relative_residual,
        'primal_dual_inner_product': fsum(complementarity),
        'shifted_complementarity': fsum(complementarity) - t*dual_trace,
        'barrier_gap_at_stationarity': mu*sum(pencil.sizes),
        'dual_b_dot_y_minus_margin': objective - t,
        'wall_seconds': perf_counter()-started,
    }
    for name, values in [('primal_unscaled', p_unscaled), ('primal_scaled', p_scaled),
                         ('dual_unscaled', d_unscaled), ('dual_scaled', d_scaled)]:
        side, scaling = name.split('_')
        summary[f'{side}_block_min_eigenvalues_{scaling}'] = [v for v, _ in values]
        summary[f'{side}_block_psd_tolerances_{scaling}'] = [tol for _, tol in values]
    vector_path = output/'solver_primal_dual.npz'
    tmp = vector_path.with_suffix('.tmp.npz')
    np.savez(tmp, x=x, s=np.concatenate(s_parts), y=np.concatenate(y_parts),
             block_sizes=pencil.sizes, margin=t, mu=mu)
    tmp.replace(vector_path)
    write_json(output/'solver_summary.json', summary)
    return summary


class WallBudget(TimeoutError):
    pass


def find_margin(pencil, output, deadline, batch, warm_start=None):
    import numpy as np
    from scipy.linalg import cho_factor, cho_solve, eigvalsh, solve_triangular
    from solve_general_4_9 import write_json

    started = perf_counter()
    history = []

    def check():
        if perf_counter() >= deadline:
            raise WallBudget

    center, r = condition_pencil(pencil, check)
    x = center if warm_start is None else np.array(warm_start, dtype=float, copy=True)
    if x.shape != (pencil.rank,) or not np.isfinite(x).all():
        raise ValueError('warm start x must be a finite vector in the original pencil coordinates')
    minimum = min(eigvalsh(matrix, subset_by_index=[0, 0])[0]
                  for matrix in slacks(pencil, x, check))
    t = minimum/2 if minimum > PRIMAL_MARGIN else minimum-max(0.5, abs(minimum)*1e-6)
    steps = 0

    def save(mu, reason):
        summary = checkpoint(pencil, x, t, mu, output, started, reason, steps)
        history.append(summary)
        write_json(output/'barrier_trace.json', history)
        print(json.dumps(summary, allow_nan=False), flush=True)
        return summary

    if minimum > PRIMAL_MARGIN:
        summary = save(1e-2, 'initial_point')
        if summary['status'] == 'primal_candidate':
            return summary

    for mu in (10.**k for k in range(-2, -14, -1)):
        reason = 'stage_iteration_limit'
        try:
            for _ in range(35):
                matrices = list(slacks(pencil, x, check))
                factors = factor_slacks(matrices, t)
                grad = np.zeros(pencil.rank+1)
                grad[-1] = -1
                hessian = np.zeros((pencil.rank+1, pencil.rank+1))
                for k, (block, factor, (_, scale)) in enumerate(
                        zip(pencil.blocks, factors, pencil.constants, strict=True)):
                    check()
                    with pencil.read_map(k) as a:
                        g, h = block_derivatives(block, a, scale, factor, r, batch, check)
                    grad += mu*g
                    hessian += mu*h
                    del g, h
                try:
                    step = -cho_solve(cho_factor(hessian), grad)
                except np.linalg.LinAlgError:
                    reason = 'hessian_factorization_failed'
                    break
                decrement = -grad @ step
                if np.max(abs(grad)) <= 3e-11 or decrement <= 5e-18:
                    reason = 'barrier_stage_complete'
                    break
                dx = solve_triangular(r, step[:-1], check_finite=False)
                dt = step[-1]
                changes = []
                for k, (block, (_, scale)) in enumerate(zip(pencil.blocks, pencil.constants, strict=True)):
                    check()
                    with pencil.read_map(k) as a:
                        changes.append(block.unpack(scale * mapped_product(a, dx)))
                weight = line_search(matrices, changes, factors, t, dt, mu, decrement, check)
                if weight is None:
                    reason = 'line_search_failed'
                    break
                candidate_x, candidate_t = x + weight*dx, t + weight*dt
                # Recheck the candidate in the original chart near the PSD boundary.
                try:
                    factor_slacks(slacks(pencil, candidate_x, check), candidate_t)
                except np.linalg.LinAlgError:
                    reason = 'line_search_roundoff'
                    break
                x, t = candidate_x, candidate_t
                steps += 1
        except WallBudget:
            return save(mu, 'wall_budget')
        if mu == 1e-13 and reason == 'barrier_stage_complete':
            reason = 'barrier_schedule_complete'
        summary = save(mu, reason)
        if summary['status'] != 'inconclusive' or reason == 'hessian_factorization_failed':
            return summary
    return summary


def worker(args):
    # Wait until the supervisor has successfully installed the hard limit.
    if sys.stdin.readline().strip() != 'start' or DEADLINE_ENV not in os.environ:
        return 2
    if os.name != 'nt':
        import resource

        limit = args.memory_mib * 1024**2
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        limit = limit if hard == resource.RLIM_INFINITY else min(hard, limit)
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
        print(f'Hard address-space limit {limit}', flush=True)
    import numpy as np

    warm_start = None
    if args.warm_start is not None:
        with np.load(args.warm_start, allow_pickle=False) as archive:
            warm_start = archive['x']
    deadline = float(os.environ[DEADLINE_ENV]) - min(10, args.seconds * 0.05)
    with Pencil(args.input, args.output/'pencil_scratch') as pencil:
        print(f'Pencil rank {pencil.rank}, PSD orders {pencil.sizes}', flush=True)
        find_margin(pencil, args.output, deadline, args.batch_directions, warm_start)
    return 0


def supervise(args):
    import psutil
    from solve_general_4_9 import THREADS, write_json

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob('solver_*')) or (out/'pencil_scratch').exists() or (out/'barrier_trace.json').exists():
        raise FileExistsError(f'Solver output exists in {out}; select a new output directory')
    command = [sys.executable, '-B', str(Path(__file__).resolve()), *args.command_line, '--worker']
    started = perf_counter()
    deadline = started + args.seconds
    limit = args.memory_mib * 1024**2
    env = dict(os.environ, PYTHONUNBUFFERED='1', **{name: '1' for name in THREADS})
    env[DEADLINE_ENV] = str(deadline)
    if os.name == 'nt' and sys.prefix != sys.base_prefix:
        # Avoid the Windows venv redirector spawning outside the Job.
        command[0] = sys._base_executable
        env['__PYVENV_LAUNCHER__'] = sys.executable
    peak = 0
    reason = guard_error = close_job = None
    with (out/'solver_run.log').open('x', encoding='utf-8') as log:
        child = subprocess.Popen(command, env=env, stdin=subprocess.PIPE, stdout=log,
                                 stderr=subprocess.STDOUT, text=True)
        try:
            process = psutil.Process(child.pid)
            if os.name == 'nt':
                close_job = windows_job(child, limit)
            child.stdin.write('start\n')
            child.stdin.close()
            while child.poll() is None:
                remaining = deadline - perf_counter()
                if remaining <= 0:
                    reason = 'wall_limit'
                    break
                rss = 0
                for item in [process, *process.children(recursive=True)]:
                    try:
                        rss += item.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak = max(peak, rss)
                if rss >= limit:
                    reason = 'rss_limit'
                    break
                sleep(max(0., min(POLL_SECONDS, deadline-perf_counter())))
        except psutil.NoSuchProcess:
            pass
        except (OSError, psutil.Error) as error:
            reason, guard_error = 'guard_error', str(error)
        finally:
            if close_job is not None:
                close_job()  # kill-on-close also reaches any remaining Job descendants
            if child.poll() is None:
                child.kill()
            returncode = child.wait()
            child.stdin.close()
    if os.name == 'nt':
        enforced = close_job is not None
    else:
        with (out/'solver_run.log').open(encoding='utf-8', errors='replace') as log:
            enforced = log.readline(256).startswith('Hard address-space limit ')
    guard = {'command': command, 'worker_pid': child.pid, 'exit_code': returncode,
             'kill_reason': reason, 'guard_error': guard_error, 'wall_seconds': perf_counter()-started,
             'peak_sampled_rss_bytes': peak, 'wall_limit_seconds': args.seconds,
             'memory_limit_bytes': limit, 'rss_poll_max_seconds': POLL_SECONDS,
             'hard_memory_limit_enforced': enforced,
             'windows_job_commit_limit_enforced': close_job is not None}
    write_json(out/'solver_guard.json', guard)
    print(json.dumps(guard), flush=True)
    return 1 if reason is not None or not enforced else returncode


def parse_args(argv=None):
    command_line = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='Pencil directory')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seconds', type=float, default=MAX_SECONDS)
    parser.add_argument('--memory-mib', type=int, default=DEFAULT_MEMORY_MIB)
    parser.add_argument('--batch-directions', type=int, default=16)
    parser.add_argument('--warm-start', type=Path, help='NPZ containing x in input coordinates')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(command_line)
    if not (isfinite(args.seconds) and 0 < args.seconds <= MAX_SECONDS
            and 16 <= args.memory_mib <= MAX_MEMORY_MIB and args.batch_directions > 0):
        parser.error('limits are 0 < seconds <= 1800, 16 <= memory-mib <= 4095, and positive batch-directions')
    args.command_line = command_line
    return args


def main():
    args = parse_args()
    return worker(args) if args.worker else supervise(args)


if __name__ == '__main__':
    sys.exit(main())
