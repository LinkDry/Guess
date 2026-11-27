"""Unrestricted complex ETF Gram search in a first-row gauge."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

# Set before NumPy/SciPy import, including when Windows spawns a worker.
THREAD_VARIABLES = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                    'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
if __name__ in ('__main__', '__mp_main__'):
    for variable in THREAD_VARIABLES:
        os.environ[variable] = '1'

import numpy as np
import scipy
from scipy.optimize import minimize

from verify_etf import verify_numeric


@dataclass(frozen=True)
class GramModel:
    d: int
    n: int

    def __post_init__(self):
        if not 2 <= self.d < self.n <= self.d**2:
            raise ValueError('require 2 <= d < n <= d²')
        mu_squared = Fraction(self.n - self.d, self.d * (self.n - 1))
        mu = float(np.sqrt(float(mu_squared)))
        free_i, free_j = (indices + 1 for indices in np.triu_indices(self.n - 1, 1))
        template = np.eye(self.n, dtype=complex)
        template[0, 1:] = template[1:, 0] = mu
        for name, value in {
            'mu_squared': mu_squared, 'mu': mu, 'scale': self.n / self.d,
            'free_i': free_i, 'free_j': free_j, 'phase_count': len(free_i),
            'upper': np.triu_indices(self.n, 1), 'off_diagonal': ~np.eye(self.n, dtype=bool),
            'template': template,
        }.items():
            object.__setattr__(self, name, value)


DEFAULT_MODEL = GramModel(4, 9)
# Keep the round-2 analysis imports working with their original model.
D, N = DEFAULT_MODEL.d, DEFAULT_MODEL.n
LAMBDA, MU = DEFAULT_MODEL.scale, DEFAULT_MODEL.mu
FREE_I, FREE_J = DEFAULT_MODEL.free_i, DEFAULT_MODEL.free_j
PHASE_COUNT = DEFAULT_MODEL.phase_count
GATE = 1e-12
METRIC_NAMES = ('tightness_frobenius', 'welch_max_error', 'hermitian_max_error', 'diagonal_max_error')
REFINE_OPTIONS = {'ftol': 0.0, 'gtol': 1e-14, 'maxls': 50, 'maxcor': 20}


def _gram_with_free_entries(values, model):
    gram = np.broadcast_to(model.template, values.shape[:-1] + (model.n, model.n)).copy()
    gram[..., model.free_i, model.free_j] = values
    gram[..., model.free_j, model.free_i] = values.conj()
    return gram


def phase_gram(angles, *, model=DEFAULT_MODEL):
    return _gram_with_free_entries(model.mu * np.exp(1j * np.asarray(angles)), model)


def project_modulus(matrix, *, model=DEFAULT_MODEL):
    """Frobenius projection with diagonal one and anchored first row/column."""
    entries = (matrix[..., model.free_i, model.free_j] + matrix[..., model.free_j, model.free_i].conj()) / 2
    magnitudes = np.abs(entries)
    unit = np.divide(entries, magnitudes, out=np.ones_like(entries, dtype=complex), where=magnitudes > 0)
    return _gram_with_free_entries(model.mu * unit, model)


def project_spectrum(matrix, *, model=DEFAULT_MODEL):
    matrix = (matrix + matrix.conj().swapaxes(-1, -2)) / 2
    _, vectors = np.linalg.eigh(matrix)
    top = vectors[..., -model.d:]
    return model.scale * (top @ top.conj().swapaxes(-1, -2))


def raw_residual(gram, *, model=DEFAULT_MODEL):
    return np.linalg.norm(gram @ gram - model.scale * gram, axis=(-2, -1))


def metrics(gram, *, model=DEFAULT_MODEL):
    return {
        'tightness_frobenius': float(raw_residual(gram, model=model)),
        'welch_max_error': float(np.max(np.abs(np.abs(gram[model.off_diagonal]) - model.mu))),
        'hermitian_max_error': float(np.max(np.abs(gram - gram.conj().T))),
        'diagonal_max_error': float(np.max(np.abs(np.diag(gram) - 1))),
    }


def passes_double_gate(values):
    return values['tightness_frobenius'] < GATE and values['welch_max_error'] < GATE


def phase_objective(angles, *, model=DEFAULT_MODEL):
    gram = phase_gram(angles, model=model)
    residual = gram @ gram - model.scale * gram
    differential = gram @ residual + residual @ gram - model.scale * residual
    gradient = -2 * np.imag(differential[model.free_i, model.free_j].conj() * gram[model.free_i, model.free_j])
    return 0.5 * float(np.vdot(residual, residual).real), gradient


def drs_batch(seeds, *, steps, relaxation, on_hit=None, model=DEFAULT_MODEL):
    """Track the terminal modulus shadow and the minimum-residual shadow."""
    starts = np.stack([np.random.Generator(np.random.PCG64(seed)).uniform(-np.pi, np.pi, model.phase_count)
                       for seed in seeds])
    state = shadow = phase_gram(starts, model=model)
    starts = shadow.copy()
    best_grams = shadow.copy()
    best_scores = raw_residual(shadow, model=model)
    best_steps = np.zeros(len(seeds), dtype=np.int64)
    alerted = np.zeros(len(seeds), dtype=bool)

    def check_hits(scores, step):
        if on_hit is None:
            return
        for i in np.flatnonzero((scores < GATE) & ~alerted):
            alerted[i] = True
            on_hit(seeds[i], f'drs_step_{step}', shadow[i])

    check_hits(best_scores, 0)
    for step in range(1, steps + 1):
        state = state + relaxation * (project_spectrum(2 * shadow - state, model=model) - shadow)
        shadow = project_modulus(state, model=model)
        scores = raw_residual(shadow, model=model)
        improved = scores < best_scores
        best_scores[improved] = scores[improved]
        best_grams[improved] = shadow[improved]
        best_steps[improved] = step
        check_hits(scores, step)
    return {'initial_grams': starts, 'best_grams': best_grams,
            'best_steps': best_steps, 'last_grams': shadow}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def recover_exact(gram, folder, *, model=DEFAULT_MODEL):
    """Polish in high precision and try a bounded radical-basis recognition."""
    import mpmath as mp

    free_positions = list(zip(map(int, model.free_i), map(int, model.free_j), strict=True))
    upper_positions = list(zip(map(int, model.upper[0]), map(int, model.upper[1]), strict=True))
    labels = ('1', 'sqrt(2)', 'sqrt(3)', 'sqrt(5)', 'sqrt(6)', 'sqrt(10)', 'sqrt(15)', 'sqrt(30)')
    mu_expression = f'sqrt({model.mu_squared.numerator}/{model.mu_squared.denominator})'
    with mp.workdps(70):
        mu = mp.sqrt(mp.mpf(model.mu_squared.numerator) / model.mu_squared.denominator)
        scale = mp.mpf(model.n) / model.d
        initial = [mp.arg(mp.mpc(float(gram[i, j].real), float(gram[i, j].imag))) for i, j in free_positions]

        def make_gram(angles):
            trial = mp.eye(model.n)
            for j in range(1, model.n):
                trial[0, j] = trial[j, 0] = mu
            for (i, j), angle in zip(free_positions, angles, strict=True):
                trial[i, j] = mu * mp.exp(mp.j * angle)
                trial[j, i] = mp.conj(trial[i, j])
            return trial

        def equations(*angles):
            trial = make_gram(angles)
            residual = trial * trial - scale * trial
            return [part for i, j in upper_positions for part in (residual[i, j].real, residual[i, j].imag)]

        def jacobian(*angles):
            trial = make_gram(angles)
            out = mp.matrix(2 * len(upper_positions), model.phase_count)
            for col, (i, j) in enumerate(free_positions):
                change = mp.j * trial[i, j]
                for row, (a, b) in enumerate(upper_positions):
                    entry = (trial[a, i] * change if b == j else 0) + (change * trial[j, b] if a == i else 0)
                    entry += ((trial[a, j] * mp.conj(change) if b == i else 0)
                              + (mp.conj(change) * trial[i, b] if a == j else 0))
                    if (a, b) == (i, j):
                        entry -= scale * change
                    out[2 * row, col], out[2 * row + 1, col] = entry.real, entry.imag
            return out

        try:
            angles = mp.findroot(equations, initial, J=jacobian, tol=mp.mpf('1e-60'), maxsteps=16)
        except (ValueError, ZeroDivisionError) as error:
            return {'status': 'high_precision_polish_failed', 'message': str(error)}
        error = max(map(abs, equations(*angles)))
        if error > mp.mpf('1e-42'):
            return {'status': 'high_precision_residual_above_recognition_gate', 'max_component': mp.nstr(error, 12)}
        basis = [mp.sqrt(v) for v in (1, 2, 3, 5, 6, 10, 15, 30)]

        def recognize(value):
            tolerance = mp.mpf('1e-38')
            if abs(value) < tolerance:
                return '0'
            for width in (1, 2, 4, len(basis)):
                relation = mp.pslq(mp.matrix([value, *basis[:width]]), tol=tolerance,
                                   maxcoeff=256, maxsteps=500)
                if relation is None or not relation[0]:
                    continue
                denominator, *coefficients = relation
                numerator = '+'.join(f'({-c})*{label}' for c, label in zip(coefficients, labels) if c)
                return f'({numerator})/({denominator})'
            return None

        entries = [[1 if i == j else mu_expression for j in range(model.n)] for i in range(model.n)]
        for (i, j), angle in zip(free_positions, angles, strict=True):
            real, imag = recognize(mp.cos(angle)), recognize(mp.sin(angle))
            if real is None or imag is None:
                return {'status': 'phase_recognition_failed', 'index': [i, j],
                        'high_precision_max_component': mp.nstr(error, 12)}
            entries[i][j] = f'{mu_expression}*(({real})+I*({imag}))'
            entries[j][i] = f'{mu_expression}*(({real})-I*({imag}))'
    exact_path = folder / 'candidate_exact.json'
    _write_json(exact_path, {'format': 'etf-gram-v1', 'gram': entries})
    command = [sys.executable, '-B', str(Path(__file__).with_name('verify_etf.py')), str(exact_path),
               '--d', str(model.d), '--n', str(model.n), '--symbolic']
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {'status': 'exact_verifier_timeout', 'timeout_seconds': 120, 'exact_gram': exact_path.name}
    return {'status': 'exact_verified' if result.returncode == 0 else 'exact_verifier_rejected',
            'exact_gram': exact_path.name, 'verifier_exit': result.returncode,
            'stdout': result.stdout, 'stderr': result.stderr}


def record_hit(seed, stage, gram, out, *, model=DEFAULT_MODEL):
    """Save and report the first hit in a stage before attempting recovery."""
    folder = out / 'double_gate_hits' / f'seed_{seed}_{stage}'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'gram.npy'
    np.save(path, gram, allow_pickle=False)
    independent = verify_numeric(path, model.d, model.n)
    report = {'seed': seed, 'stage': stage, 'metrics': metrics(gram, model=model), 'independent_verification': independent}
    _write_json(folder / 'numeric_verification.json', report)
    print(f'NUMERIC TIGHTNESS CROSSING ({model.d},{model.n}) seed={seed} stage={stage} '
          f'independent_pass={independent["passed"]} file={path}', flush=True)
    try:
        exact = recover_exact(gram, folder, model=model) if independent['passed'] else {'status': 'independent_gate_failed'}
    except (ArithmeticError, ValueError) as error:
        exact = {'status': 'recovery_error', 'message': str(error)}
    _write_json(folder / 'recovery.json', exact)
    print(f'EXACT RECOVERY seed={seed} stage={stage} status={exact["status"]}', flush=True)
    return {'seed': seed, 'stage': stage, 'independent_pass': independent['passed'],
            'recovery_status': exact['status'],
            'double_gate_passed': passes_double_gate(report['metrics']), 'gram_file': path.relative_to(out).as_posix()}


def execute_batch(seeds, steps, relaxation, refine_maxiter, out, model=DEFAULT_MODEL):
    hits = []

    def alert(seed, stage, gram):
        hits.append(record_hit(seed, stage, gram, out, model=model))

    walk = drs_batch(seeds, steps=steps, relaxation=relaxation, on_hit=alert, model=model)
    finished = []
    for index, seed in enumerate(seeds):
        best_projected = walk['best_grams'][index]
        angles = np.angle(best_projected[model.free_i, model.free_j])
        refine_alerted = any(hit['seed'] == seed for hit in hits)

        def callback(current):
            nonlocal refine_alerted
            if not refine_alerted:
                gram = phase_gram(current, model=model)
                if raw_residual(gram, model=model) < GATE:
                    refine_alerted = True
                    alert(seed, 'lbfgs_iterate', gram)

        opt = minimize(partial(phase_objective, model=model), angles, jac=True, method='L-BFGS-B', callback=callback,
                       options={**REFINE_OPTIONS, 'maxiter': refine_maxiter})
        final_gram = phase_gram(opt.x, model=model)
        final_metrics = metrics(final_gram, model=model)
        if not refine_alerted and final_metrics['tightness_frobenius'] < GATE:
            alert(seed, 'lbfgs_final', final_gram)
        row = {
            'seed': seed, 'initial': metrics(walk['initial_grams'][index], model=model),
            'best_projected': {'step': int(walk['best_steps'][index]), **metrics(best_projected, model=model)},
            'last_projected': {'step': steps, **metrics(walk['last_grams'][index], model=model)},
            'final': {**final_metrics, 'double_gate_passed': passes_double_gate(final_metrics)},
            'lbfgs': {'iterations': int(opt.nit), 'function_evaluations': int(opt.nfev),
                      'success': bool(opt.success), 'status': int(opt.status), 'message': str(opt.message),
                      'gradient_inf_norm': float(np.max(np.abs(opt.jac)))},
        }
        finished.append((row, final_gram))
    return finished, hits


def distributions(rows):
    result = {}
    names = ('min', 'p01', 'p05', 'p25', 'median', 'p75', 'p95', 'p99', 'max')
    percentiles = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    for stage in ('initial', 'best_projected', 'last_projected', 'final'):
        result[stage] = {}
        for field in METRIC_NAMES:
            values = np.array([row[stage][field] for row in rows])
            summary = dict(zip(names, map(float, np.percentile(values, percentiles)), strict=True))
            result[stage][field] = {**summary, 'mean': float(values.mean()), 'std': float(values.std())}
    residuals = np.array([row['final']['tightness_frobenius'] for row in rows])
    edges = (0.0, 1e-12, 1e-8, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, float('inf'))
    counts, _ = np.histogram(residuals, bins=edges)
    result['final_tightness_histogram_left_closed'] = [
        {'low': low, 'high': high if np.isfinite(high) else None, 'count': int(count)}
        for low, high, count in zip(edges[:-1], edges[1:], counts, strict=True)
    ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--d', type=int, default=D)
    parser.add_argument('--n', type=int, default=N)
    parser.add_argument('--seeds', type=int, default=1000)
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--steps', type=int, default=1800)
    parser.add_argument('--relaxation', type=float, default=1.0)
    parser.add_argument('--refine-maxiter', type=int, default=750)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if min(args.seeds, args.steps, args.refine_maxiter, args.workers, args.batch_size) < 1 or args.seed_start < 0:
        parser.error('seeds, steps, refine-maxiter, workers and batch-size must be positive, seed-start nonnegative')
    if not 0 < args.relaxation < 2:
        parser.error('relaxation must be in (0, 2)')
    try:
        model = GramModel(args.d, args.n)
    except ValueError as error:
        parser.error(str(error))
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    stop_seed = args.seed_start + args.seeds
    batches = [list(range(first, min(first + args.batch_size, stop_seed)))
               for first in range(args.seed_start, stop_seed, args.batch_size)]
    returned, hits = [], []
    best_so_far = float('inf')
    print(f'Starting ({model.d},{model.n}), {args.seeds} seeds, {args.steps} DRS steps, {args.workers} processes', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute_batch, seeds, args.steps, args.relaxation, args.refine_maxiter, out, model)
                   for seeds in batches]
        for future in as_completed(futures):
            finished, alerts = future.result()
            returned.extend(finished)
            hits.extend(alerts)
            best_so_far = min(best_so_far, *(row['final']['tightness_frobenius'] for row, _ in finished))
            if (len(returned) // 100 != (len(returned) - len(finished)) // 100
                    or len(returned) == args.seeds):
                print(f'{len(returned)}/{args.seeds} final matrices, best residual {best_so_far:.17g}', flush=True)

    returned.sort(key=lambda item: item[0]['seed'])
    rows = [row for row, _ in returned]
    matrices = np.stack([matrix for _, matrix in returned])
    stack_path = out / 'all_final_grams.npy'
    np.save(stack_path, matrices, allow_pickle=False)
    stored_matrices = np.load(stack_path, allow_pickle=False)
    with TemporaryDirectory(prefix='verify_', dir=out) as work:
        verification_input = Path(work) / 'gram.npy'
        for i, (row, gram) in enumerate(zip(rows, stored_matrices, strict=True)):
            np.save(verification_input, gram, allow_pickle=False)
            row['matrix_index'] = i
            row['independent_verification'] = verify_numeric(verification_input, model.d, model.n)

    top = sorted(rows, key=lambda row: (row['final']['tightness_frobenius'], row['seed']))[:10]
    top10 = [{'seed': row['seed'], 'matrix_index': row['matrix_index'], 'final': row['final'],
              'best_projected': row['best_projected'], 'last_projected': row['last_projected'],
              'independent_pass': row['independent_verification']['passed']} for row in top]
    best_projected_row = min(rows, key=lambda row: (row['best_projected']['tightness_frobenius'], row['seed']))
    best_path = out / 'best_gram.npy'
    np.save(best_path, stored_matrices[top10[0]['matrix_index']], allow_pickle=False)
    best_verdict = verify_numeric(best_path, model.d, model.n)
    _write_json(out / 'best_gram.json', {'gram_file': best_path.name, **top10[0],
                                      'independent_verification': best_verdict})
    report = {
        'model': {'d': model.d, 'n': model.n, 'lambda': model.scale, 'mu': model.mu, 'symmetry': 'unrestricted',
                  'gauge': 'G[0,j]=G[j,0]=mu for j>0', 'free_real_phases': model.phase_count,
                  'phase_positions': list(zip(map(int, model.free_i), map(int, model.free_j), strict=True))},
        'settings': {'seed_start': args.seed_start, 'seed_count': args.seeds, 'drs_steps': args.steps,
                     'drs_relaxation': args.relaxation, 'refine_maxiter': args.refine_maxiter,
                     'lbfgs_options': REFINE_OPTIONS, 'rng': 'PCG64(seed), uniform [-pi,pi)',
                     'workers': args.workers, 'batch_size': args.batch_size, 'numeric_gate_strict': GATE},
        'reproduction': {'argv_from_repo_root': ['python', '-B', 'etf_search/src/search_unconstrained.py', *sys.argv[1:]],
                         'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                         'platform': platform.platform(), 'thread_limits': {v: os.environ.get(v) for v in THREAD_VARIABLES}},
        'definitions': {
            'last_projected': 'The modulus shadow after the last DRS step, before phase refinement.',
            'best_projected': 'The modulus shadow with minimum raw Gram residual among steps 0 through drs_steps.',
            'final': 'The Gram matrix at the L-BFGS returned phase vector, initialized at best_projected.',
            'raw_gram_residual': f'Frobenius norm of G@G-({model.n}/{model.d})*G on the original Gram scale.'},
        'runtime_seconds': perf_counter() - start,
        'all_final_grams_file': stack_path.name, 'all_final_grams_shape': list(matrices.shape),
        'best_gram_file': best_path.name, 'best_gram_metadata_file': 'best_gram.json',
        'best_projected_overall': {'seed': best_projected_row['seed'], **best_projected_row['best_projected']},
        'top10': top10, 'distributions': distributions(rows),
        'tightness_gate_hits': sorted(hits, key=lambda hit: (hit['seed'], hit['stage'])),
        'double_gate_hits': sorted((hit for hit in hits if hit['double_gate_passed']),
                                   key=lambda hit: (hit['seed'], hit['stage'])),
        'independent_verification': {'checked': len(rows), 'passed': sum(row['independent_verification']['passed'] for row in rows)},
        'runs': rows,
    }
    _write_json(out / 'summary.json', report)
    print(f'Saved {stack_path}, {best_path} and {out / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
