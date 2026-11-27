"""Low-degree consistency identities from changing the anchored ETF line."""
import argparse
from collections import Counter
from itertools import product
import json
from math import gcd
from pathlib import Path

from general_4_9_model import VARIABLES, SignedMoments, edge


ANCHOR = 8


def reanchor(word, vertex):
    """Swap vertex and anchor, ungauge the edges, and cancel inverse pairs."""
    def swap(index):
        if index == ANCHOR:
            return vertex
        return ANCHOR if index == vertex else index

    counts = Counter()
    for left, right in word:
        left, right = swap(left), swap(right)
        for a, b in ((vertex, left), (left, right), (right, vertex)):
            if ANCHOR not in (a, b):
                counts[min(a, b), max(a, b)] += 1 if a < b else -1
    return tuple(directed for (a, b), count in sorted(counts.items())
                 for directed in [(a, b) if count > 0 else (b, a)] * abs(count))


def real_orbit(word, indexing, moment_ids):
    """Expand the real part of a product of directed unit complex edges."""
    result = Counter()
    for kind in product(('x', 'y'), repeat=len(word)):
        imaginary = kind.count('y')
        if imaginary % 2:
            continue
        sign = (-1) ** (imaginary // 2)
        indices = []
        for variable_kind, (a, b) in zip(kind, word):
            index, orientation = edge(variable_kind, a, b)
            indices.append(index)
            sign *= orientation
        orbit = indexing.orbit(tuple(sorted(indices)))
        if orbit:
            ident = indexing.representatives[abs(orbit) - 1]
            result[moment_ids[ident]] += sign if orbit > 0 else -sign
    return result


def build_reanchor_rows(moments):
    """Enumerate orientations of each unlabeled undirected monomial shape."""
    indexing = SignedMoments()
    moment_ids = {tuple(mon): i for i, mon in enumerate(moments)}
    witnesses = {}
    for monomial in moments:
        if not monomial or any(index % 2 for index in monomial):
            continue
        edge_list = [VARIABLES[i][1:] for i in monomial]
        used = {vertex for ab in edge_list for vertex in ab}
        anchors = sorted(used | ({min(set(range(8)) - used)} if len(used) < 8 else set()))
        for orientations in product((1, -1), repeat=len(edge_list)):
            word = tuple(ab[::direction] for ab, direction in zip(edge_list, orientations))
            lhs = real_orbit(word, indexing, moment_ids)
            for anchor in anchors:
                transformed = reanchor(word, anchor)
                if len(transformed) > 4:
                    continue
                residual = lhs.copy()
                residual.subtract(real_orbit(transformed, indexing, moment_ids))
                row = sorted((i, c) for i, c in residual.items() if c)
                if not row:
                    continue
                divisor = gcd(*(coefficient for _, coefficient in row))
                if row[0][1] < 0:
                    divisor = -divisor
                normalized = tuple((i, c // divisor) for i, c in row)
                witnesses.setdefault(normalized, {'word': word, 'new_anchor': anchor, 'divisor': divisor})
    return {'rows': [dict(row) for row in witnesses], 'witnesses': list(witnesses.values()),
            'scope': 'real moments of degree <=4 on both sides of the anchor change'}



def audit_overlay(model, overlay, paths):
    """Check redundancy over Q(sqrt(10)) and evaluate the saved moment witnesses."""
    from fractions import Fraction
    from sympy import QQ, sqrt
    from sympy.polys.matrices import DomainMatrix

    field = QQ.algebraic_field(sqrt(10))
    beta = field.from_sympy(sqrt(10) / 5)
    count = len(model['moments'])

    def rank(rows):
        converted = []
        for row in rows:
            values = [field.zero] * count
            for index, (a, b) in row.items():
                values[int(index)] = field.convert(a) + field.convert(b) * beta
            converted.append(values)
        return DomainMatrix(converted, (len(converted), count), field).rank()

    extra = [{i: (a, 0) for i, a in row.items()} for row in overlay['rows']]
    base_rank = rank(model['equalities'])
    extended_rank = rank([*model['equalities'], *extra])
    checks = {}
    for path in paths:
        witness = json.loads(path.read_text(encoding='utf-8'))
        values = [tuple(map(Fraction, row)) for row in witness['moments']]
        checks[path.parent.name] = all(
            sum(a * values[int(i)][j] for i, a in row.items()) == 0
            for row in overlay['rows'] for j in range(2)
        )
    return {'base_homogeneous_rank': base_rank, 'rank_with_reanchoring': extended_rank,
            'rows_checked': len(extra), 'all_reanchoring_rows_redundant': base_rank == extended_rank,
            'exact_witnesses_pass_reanchoring': checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--rank-output', type=Path)
    parser.add_argument('--candidate', type=Path, action='append', default=[])
    args = parser.parse_args()
    model = json.loads(args.model.read_text(encoding='utf-8'))
    data = build_reanchor_rows(model['moments'])
    args.output.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(f'{len(data["rows"])} re-anchoring equalities')
    if args.rank_output:
        audit = audit_overlay(model, data, args.candidate)
        args.rank_output.write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
