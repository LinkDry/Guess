"""Evaluate all promoted rank equalities whose averaged degree is at most four."""
import argparse
from collections import defaultdict
from fractions import Fraction
from itertools import combinations_with_replacement
import json
from pathlib import Path
from time import perf_counter

from general_4_9_model import NUMBER_OF_VARIABLES, SignedMoments
from level3_4_9_minors import rank_equations


def check_quartic(model, witness):
    index = SignedMoments()
    moments = {tuple(monomial): tuple(map(Fraction, coordinate))
               for monomial, coordinate in zip(model['moments'], witness['moments'], strict=True)}
    counts = defaultdict(lambda: {'total': 0, 'quartic_readable': 0, 'identically_zero': 0, 'violated': 0})
    failures = []
    multipliers = [tuple(combinations_with_replacement(range(NUMBER_OF_VARIABLES), degree)) for degree in range(3)]
    for name, equation in rank_equations().items():
        family = name.split('_', 1)[0]
        cap = 1 if family == 'gram5' else 2
        for degree in range(cap + 1):
            summary = counts[family, degree]
            for multiplier in multipliers[degree]:
                summary['total'] += 1
                if sum(i % 2 for i in multiplier) % 2 != int(name.endswith('_imag')):
                    summary['quartic_readable'] += 1
                    summary['identically_zero'] += 1
                    continue
                row = defaultdict(lambda: [0, 0])
                for monomial, (a, b) in equation.items():
                    code = index.orbit(tuple(sorted(monomial + multiplier)))
                    if code:
                        left = row[abs(code)]
                        sign = 1 if code > 0 else -1
                        left[0] += sign * a
                        left[1] += sign * b
                terms = [(index.representatives[i-1], a, b) for i, (a, b) in row.items() if a or b]
                if any(monomial not in moments for monomial, _, _ in terms):
                    continue
                summary['quartic_readable'] += 1
                if not terms:
                    summary['identically_zero'] += 1
                real = imaginary_coefficient = Fraction(0)
                for monomial, a, b in terms:
                    x, y = moments[monomial]
                    real += a*x + Fraction(2, 5)*b*y
                    imaginary_coefficient += a*y + b*x
                if real or imaginary_coefficient:
                    summary['violated'] += 1
                    failures.append({'equation': name, 'multiplier': list(multiplier),
                                     'residual_in_1_beta': [str(real), str(imaginary_coefficient)]})
    return {'coefficient_basis': ['1', 'sqrt(10)/5'],
            'counts_by_family_and_multiplier_degree': [dict(family=k[0], multiplier_degree=k[1], **v)
                                                       for k, v in sorted(counts.items())],
            'violations': failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'candidate', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    begin = perf_counter()
    doc = check_quartic(json.loads(args.model.read_text(encoding='utf-8')),
                        json.loads(args.candidate.read_text(encoding='utf-8')))
    doc['wall_seconds'] = perf_counter() - begin
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(doc, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({key: value for key, value in doc.items() if key != 'violations'}, indent=2))


if __name__ == '__main__':
    main()
