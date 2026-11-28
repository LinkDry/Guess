"""Guarded conic solver for targeted degree-six anchored ETF(4,9) moments."""

import argparse
from fractions import Fraction
from importlib import import_module
import json
import os
import subprocess
from pathlib import Path
import sys
from time import perf_counter, sleep

from solve_general_4_9 import (THREADS, augment_equalities, diagnostics, pack_model,
                              psd_blocks, windows_job, write_json)


BASIS = ('1', 'sqrt(10)/5')
CAP_SECONDS = 1800
CAP_MIB = 8064
DEADLINE_ENV = 'ETF_LEVEL3_WALL_DEADLINE'


def clarabel_order(cone):
    """Indices in SCS order taken in Clarabel's upper-column order."""
    import numpy as np

    offset = cone['z']
    parts = [np.arange(offset)]
    for size in cone['s']:
        row, col = np.tril_indices(size)
        parts.append(offset + col*size - col*(col-1)//2 + row-col)
        offset += size*(size+1)//2
    return np.concatenate(parts)


def fixed_overlay(model, candidate, quartic_only=False):
    """Convert a Round 4 candidate into exact affine constraints using y[()] = 1."""
    from math import lcm

    if tuple(candidate['coefficient_basis']) != BASIS:
        raise ValueError('fixed moments must use the anchored quadratic field')
    moment_index = {tuple(monomial): index for index, monomial in enumerate(model['moments'])}
    all_values = candidate['moments']
    source_order = candidate.get('moment_order')
    if source_order is None:
        raise ValueError('fixed candidate requires the source moment order')
    rows, witnesses = [], []
    for monomial, value in zip(source_order, all_values, strict=True):
        monomial = tuple(monomial)
        if not monomial or len(monomial) > 4 or (quartic_only and len(monomial) < 4):
            continue
        if monomial not in moment_index:
            raise ValueError(f'missing fixed moment {monomial}')
        index = moment_index[monomial]
        a, b = map(Fraction, value)
        scale = lcm(a.denominator, b.denominator)
        rows.append({str(index): scale, '0': [-int(a*scale), -int(b*scale)]})
        witnesses.append({'type': 'fixed_moment', 'monomial': monomial, 'value': value})
    return {'rows': rows, 'witnesses': witnesses,
            'scope': 'Conditional extension of the supplied quartic moment vector.'}


def cone_data(model):
    import numpy as np
    from scipy.sparse import coo_matrix

    if tuple(model['coefficient_basis']) != BASIS or tuple(model['moments'][0]) != ():
        raise ValueError('expected normalized anchored moment convention')
    equality = [{0: (1, 0)}, *model['equalities']]
    rhs = [(1, 0), *[(0, 0)]*len(model['equalities'])]
    rows, cols, values, b = [], [], [], []
    beta = np.sqrt(10) / 5
    row_scales = []
    for i, eq in enumerate(equality):
        # Also normalizes fixed rows with huge exact denominators.
        scale = max(abs(coef) for pair in eq.values() for coef in pair)
        row_scales.append(scale)
        b.append(rhs[i][0]/scale + beta*(rhs[i][1]/scale))
        for col, (a, c) in eq.items():
            rows.append(i)
            cols.append(col)
            values.append(a/scale + beta*(c/scale))
    sizes = []
    offset = len(equality)
    for block in model['psd_blocks']:
        size = len(block['basis'])
        if not 0 < size <= 2500:
            raise ValueError(f'PSD block size {size} outside the supported range')
        sizes.append(size)
        for i, j, entry in block['entries']:
            if not 0 <= j <= i < size:
                raise ValueError('PSD entry outside lower triangle')
            position = offset + j*size - j*(j-1)//2 + i-j
            factor = -1 if i == j else -np.sqrt(2)
            for col, coef in entry.items():
                rows.append(position)
                cols.append(col)
                values.append(factor * coef)
        offset += size*(size+1)//2
    matrix = coo_matrix((values, (rows, cols)), shape=(offset, len(model['moments']))).tocsc()
    return matrix, np.pad(b, (0, offset-len(b))), {'z': len(equality), 's': sizes}, equality, rhs, row_scales


def read_saved_model(folder):
    """Reload the exact saved model without constructing another orbit cache."""
    model = json.loads((folder/'solver_model.json').read_text(encoding='utf-8'))
    model['equalities'] = [{int(k): tuple(v) for k, v in row.items()} for row in model['equalities']]
    model['base_equations'] = {name: {tuple(m): tuple(c) for m, c in entries}
                               for name, entries in model['base_equations'].items()}
    for block in model['psd_blocks']:
        block['entries'] = [(i, j, {int(k): v for k, v in entry.items()})
                            for i, j, entry in block['entries']]
    return model


def diagonal_cone_scaling(matrix, model, cone):
    """Positive equality row scaling and invertible diagonal PSD congruences."""
    import numpy as np

    scales = np.ones(matrix.shape[0])
    equality_count = cone['z']
    norms = np.sqrt(np.asarray(matrix[:equality_count].power(2).sum(axis=1)).ravel())
    scales[:equality_count] = 1 / np.maximum(norms, 1e-30)
    position = equality_count
    for block, size in zip(model['psd_blocks'], cone['s'], strict=True):
        weights = np.array([sum(coef**2 for _, coef in polynomial) for polynomial in block['basis']])
        basis_scale = 1 / np.sqrt(weights)
        for j in range(size):
            width = size-j
            scales[position:position+width] = basis_scale[j] * basis_scale[j:]
            position += width
    return scales


def recover_dual(model, equality, rhs, ray, out, row_scales=None):
    """Check an exact residual bound on the box of normalized signed moments.

    For E x = d, M_j(x) PSD and |x_i|<=1, a contradiction follows from
    d·u < -||E.T u - sum_j M_j*(F_j F_j.T)||_1. All factors are rational.
    The residual uses the safe bound |a+b*beta|<=|a|+2|b|/3.
    """
    import numpy as np

    scales = row_scales if row_scales is not None else [1]*len(equality)
    cone = {'z': len(equality), 's': [len(block['basis']) for block in model['psd_blocks']]}
    decompositions = [np.linalg.eigh(block) for block in psd_blocks(ray, cone)]
    reports = []
    for digits in (8, 11, 14):
        denominator = 10**digits
        integer_dual = [int(round(float(y)*denominator)) for y in ray[:len(equality)]]
        dual = [Fraction(a, denominator*scale) for a, scale in zip(integer_dual, scales, strict=True)]
        left = [[Fraction(0), Fraction(0)] for _ in model['moments']]
        constant_a = constant_b = Fraction(0)
        for coeff, row, (a, b) in zip(dual, equality, rhs, strict=True):
            constant_a += coeff*a
            constant_b += coeff*b
            for position, (a, b) in row.items():
                left[position][0] += coeff*a
                left[position][1] += coeff*b
        factors = []
        for block, (eigenvalues, vectors) in zip(model['psd_blocks'], decompositions, strict=True):
            use = eigenvalues > 0
            numeric_factor = vectors[:, use] * np.sqrt(eigenvalues[use])
            factor = np.array([[round(float(v)*denominator) for v in row] for row in numeric_factor], dtype=object)
            product = factor @ factor.T
            for i, j, entry in block['entries']:
                coefficient = Fraction(product[i, j] * (1 if i == j else 2), denominator**2)
                if coefficient:
                    for position, value in entry.items():
                        left[position][0] -= coefficient*value
            factors.append(factor.tolist())
        bound = sum(abs(a) + Fraction(2, 3)*abs(b) for a, b in left)
        upper_constant = constant_a + Fraction(2, 3)*abs(constant_b)
        margin = -upper_constant - bound
        report = {'rounding_digits': digits, 'residual_l1_bound_exact': str(bound),
                  'constant_exact': [str(constant_a), str(constant_b)],
                  'margin_exact': str(margin), 'margin_numeric': float(margin),
                  'contradiction': margin > 0}
        reports.append(report)
        if report['contradiction']:
            write_json(out/'solver_dual_certificate_candidate.json', {
                'schema': 'anchored-moment-box-dual-v1', 'coefficient_basis': BASIS,
                'equality_dual': [str(value) for value in dual],
                'psd_factor_denominator': denominator, 'psd_factor_integer_rows': factors,
                'bound': report,
            })
            break
    return {**reports[-1], 'attempts': reports}


def run_solver(args, matrix, b, cone, remaining, *, initial_x=None):
    import numpy as np

    c = np.zeros(matrix.shape[1])
    if args.backend.startswith('scs'):
        import scs

        indirect = args.backend == 'scs-indirect'
        settings = dict(eps_abs=args.eps, eps_rel=args.eps, eps_infeas=args.eps/10,
                        max_iters=args.iterations, time_limit_secs=remaining, verbose=args.verbose)
        selector = {'linear_solver': scs.LinearSolver.CPU_INDIRECT if indirect else scs.LinearSolver.QDLDL}
        engine = scs.SCS({'A': matrix, 'b': b, 'c': c}, cone, **settings, **selector)
        raw = engine.solve(warm_start=initial_x is not None, x=initial_x)
        info = raw['info']
        result = 'infeasible' if info['status_val'] in (-2, -7) else 'feasible' if info['status_val'] in (1, 2) else 'unknown'
        return raw, info, result, settings
    import clarabel
    from scipy.sparse import csc_matrix

    permutation = clarabel_order(cone)
    cones = [clarabel.ZeroConeT(cone['z']), *[clarabel.PSDTriangleConeT(size) for size in cone['s']]]
    settings = clarabel.DefaultSettings()
    settings.verbose, settings.time_limit = args.verbose, remaining
    settings.max_threads, settings.max_iter = 1, args.iterations
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = args.eps
    settings.tol_infeas_abs = settings.tol_infeas_rel = args.eps / 10
    solution = clarabel.DefaultSolver(csc_matrix((len(c), len(c))), c, matrix[permutation, :].tocsc(),
                                      b[permutation], cones, settings).solve()
    inverse = np.argsort(permutation)
    raw = {'x': np.asarray(solution.x), 'y': np.asarray(solution.z)[inverse], 's': np.asarray(solution.s)[inverse]}
    status = str(solution.status)
    info = {'status': status, 'iterations': solution.iterations, 'solve_time': solution.solve_time,
            'res_pri': solution.r_prim, 'res_dual': solution.r_dual}
    result = ('infeasible' if status in ('PrimalInfeasible', 'AlmostPrimalInfeasible') else
              'feasible' if status in ('Solved', 'AlmostSolved') else 'unknown')
    return raw, info, result, {'time_limit': remaining, 'max_iter': args.iterations, 'eps': args.eps}


def worker(args):
    if sys.stdin.readline() != 'start\n':
        return 2
    import numpy as np
    from importlib.metadata import version

    started = perf_counter()
    if args.reuse_model is not None:
        model = read_saved_model(args.reuse_model)
    else:
        builder = getattr(import_module(args.builder), args.factory)
        options = json.loads(args.builder_options)
        if args.minors:
            from level3_4_9_minors import rank_equations

            options['extra_equations'] = rank_equations()
        model = builder(**options)
    if args.extra_equalities:
        model = augment_equalities(model, json.loads(args.extra_equalities.read_text(encoding='utf-8-sig')))
    many_rows = len(model['equalities'])
    if args.fixed_candidate:
        candidate = json.loads(args.fixed_candidate.read_text(encoding='utf-8-sig'))
        source = json.loads(args.fixed_model.read_text(encoding='utf-8-sig'))
        overlay = fixed_overlay(model, {**candidate, 'moment_order': source['moments']})
        model = augment_equalities(model, overlay)
    fixed = model.get('fixed_moment_rows', 0) + len(model['equalities']) - many_rows
    out = args.output
    matrix, b, cone, equality, rhs, row_scales = cone_data(model)
    saved_model = pack_model(model)
    saved_model['scope'] = model.get('scope', 'Targeted degree-six anchored moment relaxation.')
    saved_model['maximum_monomial_degree'] = 6
    saved_model['fixed_moment_rows'] = fixed
    saved_model['equality_row_scales'] = row_scales
    if args.reuse_model is None or args.extra_equalities or args.fixed_candidate:
        write_json(out/'solver_model.json', saved_model)
        np.savez_compressed(out/'solver_conic_data.npz', data=matrix.data, indices=matrix.indices,
                            indptr=matrix.indptr, shape=matrix.shape, b=b, zero_cone=cone['z'], psd_cones=cone['s'])
    scales = diagonal_cone_scaling(matrix, model, cone) if args.diagonal_scale else np.ones(len(b))
    operative_matrix = matrix.multiply(scales[:, None]).tocsc() if args.diagonal_scale else matrix
    if args.diagonal_scale:
        np.save(out/'solver_diagonal_cone_scales.npy', scales, allow_pickle=False)
    print(f'Assembled A{matrix.shape}, nnz={matrix.nnz}, {cone}, fixed={fixed}', flush=True)
    deadline = float(os.environ[DEADLINE_ENV])
    # Reserve time for diagnostic work and binary output after the backend returns.
    export_reserve = min(60, max(15, args.seconds * 0.10))
    remaining = deadline - perf_counter() - export_reserve
    if remaining <= 0:
        raise TimeoutError('builder exhausted solver time')
    initial_x = None
    if args.warm_start:
        if not args.backend.startswith('scs'):
            raise ValueError('warm-start vectors are supported only by SCS')
        with np.load(args.warm_start, allow_pickle=False) as previous:
            initial_x = np.ascontiguousarray(previous['x'], dtype=float)
        if initial_x.shape != (matrix.shape[1],) or not np.isfinite(initial_x).all():
            raise ValueError('warm-start x must be a finite vector in the same moment order')
    raw, info, status, settings = run_solver(args, operative_matrix, scales*b, cone, remaining,
                                            initial_x=initial_x)
    arrays = {name: np.asarray(raw[name]) for name in ('x','y','s')}
    native_vectors = {'y_backend': arrays['y'].copy(), 's_backend': arrays['s'].copy()} if args.diagonal_scale else {}
    arrays['y'] = scales * arrays['y']
    arrays['s'] = arrays['s'] / scales
    np.savez_compressed(out/'solver_primal_dual.npz', **arrays, **native_vectors,
                        equality_dual_scaled=arrays['y'][:cone['z']])
    diag = diagnostics(matrix, b, np.zeros(matrix.shape[1]), cone, arrays)
    result = {'scope': saved_model['scope'], 'status': status, 'raw_info': info,
              'builder': args.builder, 'factory': args.factory, 'builder_options': json.loads(args.builder_options),
              'settings': settings, 'backend': args.backend, 'diagonal_cone_scaling': args.diagonal_scale,
              'reused_model_dir': str(args.reuse_model) if args.reuse_model else None,
              'warm_start': str(args.warm_start) if args.warm_start else None, 'versions': {key: version(key) for key in ('scs','clarabel','numpy','scipy')},
              'rows': matrix.shape[0], 'moments': matrix.shape[1], 'nnz': matrix.nnz, 'cone': cone,
              'fixed_moment_rows': fixed, 'rank_minors': args.minors, 'diagnostics': diag, 'wall_seconds_worker': perf_counter()-started}
    write_json(out/'solver_summary.json', result)
    print(f'{args.backend} returned {info["status"]}; saved raw vectors.', flush=True)
    if status == 'infeasible' and diag['dual_infeasibility_ray'] is not None:
        ray = arrays['y'] / -(b @ arrays['y'])
        np.savez_compressed(out/'solver_normalized_ray.npz', y=ray)
        if not args.no_recover:
            result['dual_reconstruction'] = recover_dual(model, equality, rhs, ray, out, row_scales)
    result['wall_seconds_worker'] = perf_counter()-started
    write_json(out/'solver_summary.json', result)
    return 0


def supervise(args):
    import psutil

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    existing = list(out.glob('solver_*')) + list(out.glob('anchored_*'))
    if existing:
        raise FileExistsError(f'Solver output exists in {out}; select a new output directory')
    command = [sys.executable, '-B', str(Path(__file__).resolve()), *sys.argv[1:], '--worker']
    begin = perf_counter()
    env = dict(os.environ, PYTHONUNBUFFERED='1', **{name: '1' for name in THREADS})
    env[DEADLINE_ENV] = str(begin + args.seconds)
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
    parser.add_argument('--builder', default='level3_4_9_model')
    parser.add_argument('--factory', default='build_model')
    parser.add_argument('--builder-options', default='{}')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warm-start', type=Path, help='Primal x in an NPZ kept in the same moment order')
    parser.add_argument('--reuse-model', type=Path, help='Directory of a saved exact solver model')
    parser.add_argument('--diagonal-scale', action='store_true', help='Normalize equality rows and PSD basis coordinates')
    parser.add_argument('--extra-equalities', type=Path)
    parser.add_argument('--minors', action='store_true', help='Include targeted Gram and Schur rank equations')
    parser.add_argument('--fixed-candidate', type=Path)
    parser.add_argument('--fixed-model', type=Path)
    parser.add_argument('--seconds', type=float, default=CAP_SECONDS-10)
    parser.add_argument('--memory-mib', type=int, default=CAP_MIB)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--iterations', type=int, default=20000)
    parser.add_argument('--backend', choices=('clarabel','scs','scs-indirect'), default='scs')
    parser.add_argument('--no-recover', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (0 < args.seconds <= CAP_SECONDS and 16 <= args.memory_mib <= CAP_MIB
            and 0 < args.eps < 1 and args.iterations > 0):
        parser.error('Use 0 < seconds <= 1800, 16 <= memory-mib <= 8064, 0 < eps < 1 and positive iterations')
    if bool(args.fixed_candidate) != bool(args.fixed_model):
        parser.error('--fixed-candidate and --fixed-model must be supplied together')
    if not isinstance(json.loads(args.builder_options), dict):
        parser.error('--builder-options must be a JSON object')
    return worker(args) if args.worker else supervise(args)


if __name__ == '__main__':
    sys.exit(main())
