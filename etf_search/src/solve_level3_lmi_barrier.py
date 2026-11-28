"""Solve the common PSD margin using a compact primal log-det Newton system."""
import argparse
from fractions import Fraction
import gzip
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky, qr, solve_triangular

from solve_level3_lmi_4_9 import load_conic


def blocks_of(packed, sizes, directions):
    offset = 0
    blocks = []
    for size in sizes:
        length = size*(size+1)//2
        column, row = np.triu_indices(size)
        values = packed[offset:offset+length].T.copy()
        values[:, row != column] /= np.sqrt(2)
        block = np.empty((len(values), size, size))
        block[:, row, column] = block[:, column, row] = values
        blocks.append(block if directions else block[0])
        offset += length
    return blocks


def pack_blocks(blocks):
    result = []
    for block in blocks:
        column, row = np.triu_indices(len(block))
        values = block[row, column].copy()
        values[row != column] *= np.sqrt(2)
        result.append(values)
    return np.concatenate(result)


def find_margin(A, rhs, sizes, scale, seconds, output):
    """Improve t along a feasible barrier path for D M(x) D - tI > 0."""
    start = perf_counter()
    history = []
    packed_directions = -A.multiply(scale[:, None]).toarray()
    orthogonal, triangle = qr(packed_directions, mode='economic', check_finite=False)
    rhs = rhs*scale
    center = -orthogonal.T @ rhs
    constants = blocks_of((rhs+orthogonal @ center)[:, None], sizes, False)
    shift = pack_blocks([np.eye(n) for n in sizes])
    stencils = blocks_of(np.column_stack((orthogonal, -shift)), sizes, True)
    variables = len(center)+1
    point = np.zeros(variables)
    point[-1] = min(np.linalg.eigvalsh(block)[0] for block in constants)-0.5
    target = np.zeros(variables)
    target[-1] = 1
    order = sum(sizes)
    block_arrays = [np.ascontiguousarray(block.reshape(variables, -1)) for block in stencils]

    def factorizations(value):
        return [cholesky(matrix + (value @ block).reshape(matrix.shape), lower=True, check_finite=False)
                for matrix, block in zip(constants, block_arrays, strict=True)]

    def objective(value, factors, mu):
        return -value[-1] - 2*mu*sum(np.log(factor.diagonal()).sum() for factor in factors)

    def save(mu, factors, reason):
        x = solve_triangular(triangle, point[:-1]+center, check_finite=False)
        dual = pack_blocks([mu*cho_solve((factor, True), np.eye(len(factor)), check_finite=False)
                            for factor in factors])*scale
        primal = rhs/scale - A @ x
        np.savez_compressed(output/'solver_primal_dual.npz', x=x, y=dual, s=primal)
        summary = {'mu': mu, 'primal_scaled_min_margin': float(point[-1]),
                   'dual_b_dot_y': float((rhs/scale) @ dual),
                   'dual_stationarity_inf_unscaled': float(max(abs(A.T @ dual))),
                   'barrier_gap_at_stationarity': mu*order, 'termination': reason,
                   'wall_seconds': perf_counter()-start}
        (output/'solver_summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
        history.append(summary)
        (output/'barrier_trace.json').write_text(json.dumps(history, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(summary), flush=True)
        return summary

    for mu in (10.**k for k in range(-2, -14, -1)):
        for _ in range(35):
            factors = factorizations(point)
            if perf_counter()-start >= seconds:
                save(mu, factors, 'wall_budget')
                return
            gradient = -target.copy()
            hessian = np.zeros((variables, variables))
            for factor, stencil in zip(factors, stencils, strict=True):
                inverse_factor = solve_triangular(factor, np.eye(len(factor)), lower=True, check_finite=False)
                whitened = inverse_factor @ stencil @ inverse_factor.T
                gradient -= mu*np.trace(whitened, axis1=1, axis2=2)
                flattened = whitened.reshape((variables, -1))
                hessian += mu*(flattened @ flattened.T)
            try:
                step = -cho_solve(cho_factor(hessian, check_finite=False), gradient, check_finite=False)
            except np.linalg.LinAlgError:
                save(mu, factors, 'hessian_factorization_failed')
                return
            decrement = -gradient @ step
            if np.max(np.abs(gradient)) <= 3e-11 or decrement <= 5e-18:
                break
            old = objective(point, factors, mu)
            for power in range(40):
                weight = 0.5**power
                candidate = point + weight*step
                try:
                    value = objective(candidate, factorizations(candidate), mu)
                except np.linalg.LinAlgError:
                    continue
                if value <= old - 0.01*weight*decrement + 1e-16:
                    point[:] = candidate
                    break
            else:
                break
        summary = save(mu, factorizations(point), 'barrier_stage_complete')
        if summary['primal_scaled_min_margin'] > 1e-9:
            return
        if (summary['dual_b_dot_y'] < -1e-9 and
                summary['dual_stationarity_inf_unscaled'] < -1e-4*summary['dual_b_dot_y']):
            return


def export_exact_primal(chart_path, retained, solution, output, denominator=10**14):
    """Round only the free coordinates and apply the saved affine map exactly."""
    with gzip.open(chart_path, 'rt', encoding='utf-8') as stream:
        chart = json.load(stream)
    if chart['schema'] != 'anchored-affine-chart-v1':
        raise ValueError('unexpected affine chart schema')
    width = len(chart['free_columns'])
    retained = list(map(int, retained))
    if len(set(retained)) != len(retained) or any(not 0 <= i < width for i in retained):
        raise ValueError('invalid retained chart coordinates')
    slots = [Fraction(1)] + [Fraction(0)]*width
    for i, value in zip(retained, solution, strict=True):
        slots[i+1] = Fraction(round(float(value)*denominator), denominator)
    coordinates = []
    for (scale, terms), parity in zip(chart['rows'], chart['parity'], strict=True):
        exact = sum((numerator*slots[slot] for slot, numerator in terms), Fraction(0))/scale
        coordinates.append(['0', str(exact)] if parity else [str(exact), '0'])
    candidate = {'schema': 'anchored-exact-primal-v1', 'coefficient_basis': ['1', 'sqrt(10)/5'],
                 'moments': coordinates, 'source_chart_free_rounding_denominator': denominator}
    output.write_text(json.dumps(candidate, separators=(',', ':'))+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seconds', type=float, default=240)
    parser.add_argument('--chart', type=Path, help='Export a rational lift using this affine chart')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    A, rhs, _, sizes, scale, archive = load_conic(args.input, use_congruence=True)
    find_margin(A, rhs, sizes, scale, args.seconds, args.output)
    if args.chart:
        retained = archive.get('retained_chart_columns', np.arange(A.shape[1]))
        with np.load(args.output/'solver_primal_dual.npz', allow_pickle=False) as raw:
            export_exact_primal(args.chart, retained, raw['x'], args.output/'exact_original_primal.json')


if __name__ == '__main__':
    main()
