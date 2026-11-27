"""Independently verify a search SDP certificate using integer arithmetic."""
import argparse
from fractions import Fraction
import json
from math import isqrt, lcm
from pathlib import Path


def verify(model, certificate):
    expected_basis = ['1', 'sqrt(2)', 'sqrt(5)', 'sqrt(10)']
    if model['schema'] != 'search-sdp-model-v1' or certificate['schema'] != 'search-sdp-certificate-v1':
        raise ValueError('Unsupported schema')
    if model['coefficient_basis'] != expected_basis or certificate['coefficient_basis'] != expected_basis:
        raise ValueError('Unsupported field coordinates')
    phase_count = len(model['variables'])

    def monomial(indices):
        if any(type(i) is not int or not 0 <= i < phase_count for i in indices):
            raise ValueError('Invalid monomial variable index')
        return tuple(sorted(indices))

    denominator = certificate['factor_denominator']
    if type(denominator) is not int or denominator <= 0:
        raise ValueError('Invalid PSD factor denominator')
    equations = {}
    for key, terms in certificate['multipliers'].items():
        eq_terms = [(monomial(m), tuple(map(Fraction, values))) for m, values in model['equations'][key]]
        if any(len(values) != 4 for _, values in eq_terms):
            raise ValueError('Invalid coefficient dimension')
        multipliers = [(monomial(m), Fraction(weight)) for m, weight in terms]
        equations[key] = eq_terms, multipliers
    common = denominator**2
    for terms, multipliers in equations.values():
        coefficient_denominator = lcm(*(v.denominator for _, values in terms for v in values))
        multiplier_denominator = lcm(*(w.denominator for _, w in multipliers))
        common = lcm(common, coefficient_denominator*multiplier_denominator)
    residual = {(): [-common, 0, 0, 0]}
    for terms, multipliers in equations.values():
        for index, weight in multipliers:
            for term, values in terms:
                row = residual.setdefault(tuple(sorted(index+term)), [0, 0, 0, 0])
                for i, value in enumerate(values):
                    product = weight*value
                    row[i] -= product.numerator*(common//product.denominator)
    psd_scale = common//denominator**2
    for block in certificate['psd_blocks']:
        basis = [monomial(m) for m in block['basis']]
        factor = block['factor_integer_rows']
        if not basis or len(basis) != len(factor) or not factor[0]:
            raise ValueError('Invalid PSD factor dimension')
        width = len(factor[0])
        if any(len(row) != width or any(type(v) is not int for v in row) for row in factor):
            raise ValueError('The PSD factor must be a rectangular integer matrix')
        for i in range(len(basis)):
            for j in range(i+1):
                dot = sum(a*b for a, b in zip(factor[i], factor[j]))
                if dot:
                    row = residual.setdefault(tuple(sorted(basis[i]+basis[j])), [0, 0, 0, 0])
                    row[0] -= psd_scale * dot * (1 if i == j else 2)
    interval_denominator = certificate['radical_interval_denominator']
    if type(interval_denominator) is not int or interval_denominator <= 0:
        raise ValueError('Invalid radical interval denominator')
    lower = [interval_denominator] + [isqrt(d*interval_denominator**2) for d in (2, 5, 10)]
    upper = [interval_denominator] + [v+1 for v in lower[1:]]
    total = 0
    for row in residual.values():
        endpoints = [(c*lo, c*hi) for c, lo, hi in zip(row, lower, upper)]
        total += max(abs(sum(min(a,b) for a,b in endpoints)), abs(sum(max(a,b) for a,b in endpoints)))
    bound = Fraction(total, common*interval_denominator)
    if bound != Fraction(certificate['residual_upper_bound_on_unit_box']):
        raise ValueError('Saved bound differs from exact reconstruction')
    return bound


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('certificate', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    model = json.loads(args.model.read_text(encoding='utf-8'))
    certificate = json.loads(args.certificate.read_text(encoding='utf-8'))
    bound = verify(model, certificate)
    result = {'exact_proof': bound < 1, 'residual_upper_bound': str(bound), 'residual_upper_bound_approx': float(bound)}
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    print(text, end='')
    return 0 if result['exact_proof'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
