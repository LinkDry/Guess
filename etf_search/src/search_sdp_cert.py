"""Sparse degree-four moment relaxations for reflection-invariant ETF(4,9)."""
import argparse
from fractions import Fraction
from itertools import combinations, combinations_with_replacement
from importlib.metadata import version
import json
from math import isfinite, isqrt, sqrt
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter, sleep


def model(mode):
    import sympy as sp

    fixed = 1 if mode == 'fixed1' else 3
    sign = -1 if mode == 'fixed3_negative' else 1
    pairs = (9 - fixed) // 2
    mu = sp.sqrt(10) / 8
    scale = sp.Rational(9, 4)
    gram = sp.eye(9)
    transform = sp.eye(9)
    variables, groups, circles = [], [], []

    def phase(a, b):
        number = len(groups)
        real, imag = sp.symbols(f'x{2*number} x{2*number+1}', real=True)
        variables.extend((real, imag))
        circles.append(real**2 + imag**2 - 1)
        groups.append(tuple(sorted({(j-fixed)//2 for j in (a, b) if j >= fixed})))
        return mu*(real + sp.I*imag)

    def entry(a, b, z):
        gram[a, b], gram[b, a] = z, sp.conjugate(z)

    for j in range(1, 9):
        entry(0, j, mu)
    if fixed == 3:
        entry(1, 2, sign*mu)
    for a in range(fixed, 9, 2):
        transform[a:a+2, a:a+2] = sp.Matrix([[1, -sp.I], [1, sp.I]]) / sp.sqrt(2)
        entry(a, a+1, phase(a, a+1))
        for i in range(1, fixed):
            z = phase(i, a)
            entry(i, a, z)
            entry(i, a+1, sp.conjugate(z))
        for b in range(a+2, 9, 2):
            z, w = phase(a, b), phase(a, b+1)
            entry(a, b, z)
            entry(a+1, b+1, sp.conjugate(z))
            entry(a, b+1, w)
            entry(a+1, b, sp.conjugate(w))
    real_gram = (transform.H*gram*transform).applyfunc(sp.expand)
    assert real_gram == real_gram.T
    k, c, d = real_gram[:fixed, :fixed], real_gram[:fixed, fixed:], real_gram[fixed:, fixed:]
    k_inv = k.inv().applyfunc(sp.radsimp)
    schur = (d-c.T*k_inv*c).applyfunc(sp.expand)
    raw = {f'circle_{j}': p for j, p in enumerate(circles)}
    tight = real_gram**2 - scale*real_gram
    for a in range(9):
        for b in range(a, 9):
            raw[f'tight_{a}_{b}'] = tight[a, b]
    if fixed == 3:
        for a, b in combinations(range(2*pairs), 2):
            for i, j in combinations(range(2*pairs), 2):
                if (a, b) <= (i, j):
                    raw[f'schur_{a}{b}_{i}{j}'] = schur[a,i]*schur[b,j]-schur[a,j]*schur[b,i]
    eq = {}
    for name, expression in raw.items():
        p = sp.Poly(sp.expand(expression), *variables)
        if not p.is_zero:
            eq[name] = p
    return variables, groups, eq, schur


def sparse_terms(poly):
    return {tuple(i for i, deg in enumerate(exponents) for _ in range(deg)): coefficient
            for exponents, coefficient in poly.terms()}


def multiply(a, b):
    return tuple(sorted(a+b))


def coefficients(value):
    """Exact representation in the ordered basis 1, sqrt(2), sqrt(5), sqrt(10)."""
    import sympy as sp

    value = sp.expand(value)
    basis = [sp.sqrt(n) for n in (2, 5, 10)]
    irr = [value.coeff(radical) for radical in basis]
    const = value - sum(c*r for c, r in zip(irr, basis))
    if not all(c.is_Rational for c in [const, *irr]):
        raise ValueError(f'Unexpected coefficient in exact field: {value}')
    return tuple(Fraction(int(c.p), int(c.q)) for c in [const, *irr])


def pack(polynomials):
    return {name: [[list(mon), [str(x) for x in coeff]] for mon, coeff in terms.items()]
            for name, terms in polynomials.items()}


def build(mode, layout):
    x, groups, eq, _ = model(mode)
    polynomials = {name: {mon: coefficients(val) for mon, val in sparse_terms(p).items()}
                   for name, p in eq.items()}
    count = len(x)
    singletons = [(i,) for i in range(count)]
    bases = [[(), *singletons]]
    if layout != 'linear':
        num_pairs = 4 if mode == 'fixed1' else 3
        bags = []
        if layout == 'dense':
            bags.append(list(range(count)))
        else:
            # All variables incident to a pair form a bag. Add pair-induced bags.
            bags.extend([2*i+j for i, support in enumerate(groups) if p in support for j in range(2)]
                        for p in range(num_pairs))
            bags.extend([2*i+j for i, support in enumerate(groups) if set(support) <= {p, q} for j in range(2)]
                        for p, q in combinations(range(num_pairs), 2))
        for bag in sorted({tuple(sorted(v)) for v in bags}):
            bases.append([(), *((i,) for i in bag), *combinations_with_replacement(bag, 2)])
        # Add mixed cliques of at most eight variables when needed for a single constraint.
        if layout == 'sparse':
            existing = [set(v) for v in bags]
            support_sets = {tuple(sorted({i for mon in terms for i in mon})) for terms in polynomials.values()}
            for support in sorted(support_sets):
                if len(support) <= 8 and not any(set(support) <= bag for bag in existing):
                    bases.append([(), *((i,) for i in support), *combinations_with_replacement(support, 2)])
                    existing.append(set(support))
    return [str(v) for v in x], polynomials, bases


def prepare(polynomials, bases):
    import numpy as np
    from scipy.sparse import coo_matrix

    moments = {(): 0}
    cone_rows = []
    for basis in bases:
        for j, left in enumerate(basis):
            for i in range(j, len(basis)):
                mon = multiply(left, basis[i])
                column = moments.setdefault(mon, len(moments))
                cone_rows.append((column, -1.0 if i == j else -sqrt(2)))

    equalities = [(None, ())]
    rows, cols, vals = [0], [0], [1.0]
    radical = np.sqrt([1, 2, 5, 10])
    by_degree = [[m for m in moments if len(m) <= degree] for degree in range(4)]
    is_dense = len(bases) == 2 and len(bases[-1]) > 300
    for name, terms in polynomials.items():
        deg = max(map(len, terms))
        float_terms = {mon: float(np.dot(np.asarray(coef, dtype=float), radical)) for mon, coef in terms.items()}
        multipliers = by_degree[4-deg] if deg > 0 and len(bases) > 1 else [()]
        for multiplier in multipliers:
            shifted = [multiply(mon, multiplier) for mon in terms]
            if multiplier and not is_dense and any(mon not in moments for mon in shifted):
                continue
            row = len(equalities)
            equalities.append((name, multiplier))
            for mon, value in zip(shifted, float_terms.values()):
                col = moments.setdefault(mon, len(moments))
                rows.append(row)
                cols.append(col)
                vals.append(value)
    offset = len(equalities)
    for i, (col, coeff) in enumerate(cone_rows, offset):
        rows.append(i)
        cols.append(col)
        vals.append(coeff)
    matrix = coo_matrix((vals, (rows, cols)), shape=(offset+len(cone_rows), len(moments))).tocsc()
    b = np.zeros(matrix.shape[0]); b[0] = 1
    return matrix, b, equalities, moments


def desvec(values, size):
    import numpy as np

    result = np.zeros((size, size))
    at = 0
    for j in range(size):
        result[j, j] = values[at]
        count = size - 1 - j
        result[j+1:, j] = result[j, j+1:] = values[at+1:at+1+count] / sqrt(2)
        at += count + 1
    return result


def exact_certificate(polys, bases, eq_rows, equality_dual, blocks, digits=11):
    """Prove strict negativity on the phase box using rounded exact PSD factors."""
    import numpy as np

    denom = 10**digits
    residual = {}
    root_coeff = {}
    residual[()] = [Fraction(-1), Fraction(0), Fraction(0), Fraction(0)]
    raw_multiplier = -equality_dual / (-equality_dual[0])
    for (name, mon), number in zip(eq_rows[1:], raw_multiplier[1:]):
        weight = Fraction(round(float(number)*denom), denom)
        if not weight:
            continue
        root_coeff.setdefault(name, []).append([list(mon), str(weight)])
        for source, coef in polys[name].items():
            r = residual.setdefault(multiply(source, mon), [Fraction(0)]*4)
            for i in range(4):
                r[i] -= weight*coef[i]
    saved_blocks = []
    for basis, block in zip(bases, blocks):
        eig, vec = np.linalg.eigh(block / (-equality_dual[0]))
        active = eig > 1e-15
        factor = np.rint(vec[:, active] * np.sqrt(eig[active])[None, :] * denom).astype(np.int64)
        if not factor.size:
            continue
        if np.max(np.abs(factor)) * sqrt(factor.shape[1]) > 2e9:
            gram = factor.astype(object) @ factor.astype(object).T
        else:
            gram = factor @ factor.T
        for i, m in enumerate(basis):
            for j in range(i+1):
                value = int(gram[i, j]) * (1 if i == j else 2)
                if value:
                    r = residual.setdefault(multiply(m, basis[j]), [Fraction(0)]*4)
                    r[0] -= Fraction(value, denom**2)
        saved_blocks.append({'basis': [list(m) for m in basis], 'factor_integer_rows': factor.tolist()})
    interval_denominator = 10**30
    radical_floors = [interval_denominator, *[isqrt(r*interval_denominator**2) for r in (2, 5, 10)]]
    bound = Fraction(0)
    for coefficient in residual.values():
        left = sum(c*floor for c, floor in zip(coefficient, radical_floors))
        lower = left + sum(min(c, 0) for c in coefficient[1:])
        upper = left + sum(max(c, 0) for c in coefficient[1:])
        bound += max(abs(lower), abs(upper))
    bound /= interval_denominator
    return {'schema': 'search-sdp-certificate-v1', 'meaning': '-1 = sum h_i*m_i + sum |L_i^T v_i/d|^2 + residual',
            'coefficient_basis': ['1', 'sqrt(2)', 'sqrt(5)', 'sqrt(10)'], 'factor_denominator': denom,
            'multipliers': root_coeff, 'psd_blocks': saved_blocks,
            'radical_interval_denominator': interval_denominator,
            'residual_upper_bound_on_unit_box': str(bound), 'exact_proof': bound < 1}, bound


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if type(value).__module__ == 'numpy':
        return json_safe(value.item())
    if isinstance(value, float) and not isfinite(value):
        return None
    return value


def write_json(path, payload):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def worker(args):
    import numpy as np
    import scs

    start = perf_counter()
    variables, polys, bases = build(args.branch, args.layout)
    matrix, b, eq_rows, moments = prepare(polys, bases)
    n_eq = len(eq_rows)
    sizes = list(map(len, bases))
    out = args.output
    write_json(out/'model.json', {'schema': 'search-sdp-model-v1', 'branch': args.branch, 'layout': args.layout,
               'variables': variables, 'coefficient_basis': ['1', 'sqrt(2)', 'sqrt(5)', 'sqrt(10)'],
               'monomials': 'Sorted repeated zero-based variable indices', 'equations': pack(polys),
               'equalities': eq_rows, 'moment_order': list(moments), 'bases': bases,
               'svec': 'Lower triangle, column major, off-diagonal multiplied by sqrt(2)'})
    print(f'{args.branch} {args.layout}: A{matrix.shape} nnz={matrix.nnz} PSD={sizes}', flush=True)
    # Peak-memory guard is in the supervising process. This cap also blocks unsupported layouts.
    if max(sizes) > 600 or matrix.nnz > 3_000_000:
        raise ValueError('Moment relaxation exceeds configured matrix limits')
    data = {'A': matrix, 'b': b, 'c': np.zeros(matrix.shape[1])}
    settings = {'eps_abs': args.eps, 'eps_rel': args.eps, 'eps_infeas': args.eps/10,
                'max_iters': args.iterations, 'time_limit_secs': max(0.1, args.seconds-(perf_counter()-start)-2),
                'verbose': args.verbose}
    backend = ({'linear_solver': scs.LinearSolver.CPU_INDIRECT}
               if hasattr(scs, 'LinearSolver') else {'use_indirect': True})
    solution = scs.solve(data, {'z': n_eq, 's': sizes}, **settings, **backend)
    y = solution['y']
    blocks = []
    offset = n_eq
    for size in sizes:
        length = size*(size+1)//2
        blocks.append(desvec(y[offset:offset+length], size))
        offset += length
    np.savez_compressed(out/'raw_solver.npz', y=y, equality_dual=y[:n_eq], x=solution['x'], s=solution['s'],
                       **{f'psd_dual_{j}': block for j, block in enumerate(blocks)})
    info = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in solution['info'].items()}
    result = {'branch': args.branch, 'layout': args.layout, 'settings': settings, 'solver': 'SCS indirect',
              'solver_info': info, 'matrix_shape': matrix.shape, 'matrix_nonzeros': matrix.nnz,
              'versions': {name: version(name) for name in ('scs', 'numpy', 'scipy', 'sympy', 'psutil')},
              'phase_coordinate_count': len(variables), 'equality_row_count': n_eq,
              'psd_block_sizes': sizes, 'exact_certificate': False}
    if info['status_val'] in (1, 2) and np.isfinite(solution['x']).all():
        point = solution['x']
        result['equality_max_error'] = float(np.max(np.abs(matrix[:n_eq]@point-b[:n_eq])))
        offset = n_eq
        result['moment_min_eigenvalues'] = []
        slacks = b-matrix@point
        for size in sizes:
            length = size*(size+1)//2
            result['moment_min_eigenvalues'].append(float(np.linalg.eigvalsh(desvec(slacks[offset:offset+length], size))[0]))
            offset += length
    if info['status_val'] in (-2, -7) and np.isfinite(y).all() and y[0] < 0:
        result['dual_max_error'] = float(np.max(np.abs(matrix.T@y)))
        certificate, bound = exact_certificate(polys, bases, eq_rows, y[:n_eq], blocks)
        write_json(out/'certificate.json', certificate)
        result['exact_certificate'] = certificate['exact_proof']
        result['certificate_residual_bound'] = float(bound)
    result['worker_wall_seconds'] = perf_counter()-start
    write_json(out/'summary.json', result)
    print(info['status'], result['exact_certificate'], flush=True)


def recover(output):
    """Rebuild a certificate from persisted solver output without rerunning SCS."""
    import numpy as np

    start = perf_counter()
    metadata = json.loads((output/'model.json').read_text(encoding='utf-8'))
    polys = {name: {tuple(mon): tuple(map(Fraction, coef)) for mon, coef in terms}
             for name, terms in metadata['equations'].items()}
    bases = [[tuple(mon) for mon in basis] for basis in metadata['bases']]
    eq_rows = [(name, tuple(mon)) for name, mon in metadata['equalities']]
    with np.load(output/'raw_solver.npz', allow_pickle=False) as dual:
        equality_dual = dual['equality_dual']
        blocks = [dual[f'psd_dual_{index}'] for index in range(len(bases))]
    if not np.isfinite(equality_dual).all() or equality_dual[0] >= 0:
        raise ValueError('Expected a finite infeasibility dual with negative normalization')
    certificate, bound = exact_certificate(polys, bases, eq_rows, equality_dual, blocks)
    write_json(output/'certificate.json', certificate)
    summary_path = output/'summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        summary.update(exact_certificate=certificate['exact_proof'], certificate_residual_bound=float(bound),
                       recovery_wall_seconds=perf_counter()-start)
        write_json(summary_path, summary)
    print(f'Exact proof: {certificate["exact_proof"]}, residual bound {float(bound):.12g}', flush=True)
    return 0 if certificate['exact_proof'] else 1


def supervise(args):
    import psutil

    args.output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker']
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    begin = perf_counter()
    peak = 0
    killed = None
    with (args.output/'solver.log').open('w', encoding='utf-8') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                group = [process, *process.children(recursive=True)]
                rss = sum(item.memory_info().rss for item in group)
                peak = max(peak, rss)
            except psutil.NoSuchProcess:
                continue
            if peak >= args.memory_mib*1024**2 or perf_counter()-begin >= args.seconds:
                killed = 'rss_limit' if peak >= args.memory_mib*1024**2 else 'time_limit'
                for item in reversed(group):
                    try:
                        item.kill()
                    except psutil.NoSuchProcess:
                        pass
                break
            sleep(0.1)
        returncode = child.wait()
    report = {'exit_code': returncode, 'kill_reason': killed, 'peak_sampled_rss_bytes': peak,
              'wall_seconds': perf_counter()-begin, 'memory_limit_mib': args.memory_mib,
              'time_limit_seconds': args.seconds}
    write_json(args.output/'guard.json', report)
    print(json.dumps(report), flush=True)
    return returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recover', action='store_true', help='Rebuild a certificate from saved model and dual')
    parser.add_argument('--branch', choices=('fixed1', 'fixed3_positive', 'fixed3_negative'))
    parser.add_argument('--layout', choices=('linear', 'sparse', 'dense'), default='sparse')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=900)
    parser.add_argument('--memory-mib', type=int, default=2048)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--iterations', type=int, default=20000)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.recover:
        return recover(args.output)
    if not args.branch:
        parser.error('--branch is required for a solve')
    if not 0 < args.seconds <= 900 or not 0 < args.memory_mib <= 2048 or args.iterations < 1 or args.eps <= 0:
        parser.error('Use at most 900 s and 2048 MiB and positive tolerance and iteration budget')
    return worker(args) if args.worker else supervise(args)


if __name__ == '__main__':
    sys.exit(main())
