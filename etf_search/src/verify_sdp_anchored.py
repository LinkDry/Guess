"""Independent equations and signed averaging for the anchored ETF(4,9)."""

from functools import cache
from itertools import combinations, permutations

import sympy as sp


EDGES = tuple((i, j) for i, j in combinations(range(8), 2))
VARIABLES = tuple((kind, i, j) for i, j in EDGES for kind in ("x", "y"))
EDGE_INDEX = {edge: 2*i for i, edge in enumerate(EDGES)}
BRANCH = "anchored_general_4_9"


@cache
def canonical_anchored_equations():
    """Use the bottom block of the first-line dephased signature matrix."""
    symbols = sp.symbols("x:56", real=True)
    entries = sp.zeros(8)
    for edge, index in EDGE_INDEX.items():
        i, j = edge
        entries[i, j] = symbols[index] + sp.I * symbols[index+1]
        entries[j, i] = sp.conjugate(entries[i, j])
    beta = sp.sqrt(10) / 5
    # The full signature matrix has off-anchor entries 1 and bottom block S.
    # Its equation implies S1=beta*1 and S^2-beta*S-8I+J=0.
    row = sum(entries[0, j] for j in range(8)) - beta
    tight = (entries**2 - beta*entries - 8*sp.eye(8) + sp.ones(8))[0, 1]
    return symbols, {
        "circle": entries[0, 1]*entries[1, 0] - 1,
        "row_real": sp.re(row), "row_imag": sp.im(row),
        "tight_real": sp.re(tight), "tight_imag": sp.im(tight),
    }


@cache
def canonical_rank_minor(kind, rows, columns):
    """Reconstruct a rank constraint from the original first-line gauged Gram."""
    symbols = sp.symbols("x:56", real=True)
    mu = sp.sqrt(10) / 8
    if kind == "schur":
        dimension, order = 8, 4
    elif kind == "gram":
        dimension, order = 9, 5
    else:
        raise ValueError("rank minor must be a Schur or Gram minor")
    for indices in (rows, columns):
        if len(indices) != order or tuple(sorted(set(indices))) != indices or any(
            type(i) is not int or not 0 <= i < dimension for i in indices
        ):
            raise ValueError("invalid rank minor indices")

    def gram_entry(i, j):
        if i == j:
            return sp.S.One
        if 8 in (i, j):
            return mu
        position = EDGE_INDEX[min(i, j), max(i, j)]
        orientation = 1 if i < j else -1
        return mu * (symbols[position] + orientation * sp.I * symbols[position+1])

    # Compute in an exact polynomial ring over Q(sqrt(10), i). Laplace
    # expansion reuses complementary minors and keeps intermediate terms sparse.
    from sympy.polys.rings import ring

    expressions = [[gram_entry(i, j) - (mu**2 if kind == "schur" else 0) for j in columns] for i in rows]
    active = tuple(x for x in symbols if any(entry.has(x) for row in expressions for entry in row))
    domain = sp.QQ.algebraic_field(sp.sqrt(10), sp.I)
    polynomial_ring, *_ = ring(active, domain)
    matrix = [[polynomial_ring.from_expr(value) for value in row] for row in expressions]

    @cache
    def determinant(remaining):
        if not remaining:
            return polynomial_ring.one
        i = order - len(remaining)
        total = polynomial_ring.zero
        for position, j in enumerate(remaining):
            if matrix[i][j]:
                product = matrix[i][j] * determinant(remaining[:position] + remaining[position+1:])
                total += product if position % 2 == 0 else -product
        return total

    real, imag = [], []
    for degrees, coefficient in determinant(tuple(range(order))).items():
        monomial = sp.prod(x**degree for x, degree in zip(active, degrees) if degree)
        c, d = domain.to_sympy(coefficient).as_real_imag()
        real.append(c * monomial)
        imag.append(d * monomial)
    return symbols, sp.Add(*real), sp.Add(*imag)


@cache
def named_rank_equation(name):
    """Expand the scaled determinant used by the degree-six manifest."""
    import re

    match = re.fullmatch(r"(schur4|gram5)_([0-8]+)_([0-8]+)_(real|imag)", name)
    if not match:
        return None
    stem, rows, columns, part = match.groups()
    kind = "schur" if stem == "schur4" else "gram"
    indices = tuple(map(int, rows)), tuple(map(int, columns))
    _, real, imag = canonical_rank_minor(kind, *indices)
    return (32 ** len(rows)) * (real if part == "real" else imag)


def anchored_variable_order(variables):
    expected_names = [f"x{i}" for i in range(len(VARIABLES))]
    return variables == expected_names or variables == [list(edge) for edge in VARIABLES]


@cache
def actions(support_size):
    edges = tuple(combinations(range(support_size), 2))
    result = []
    for p in permutations(range(support_size)):
        action = []
        for left, right in edges:
            a, b = p[left], p[right]
            index = EDGE_INDEX[(a, b) if a < b else (b, a)]
            action.extend((index+1, (index+2) if a < b else -(index+2)))
        result.append(tuple(action))
    return tuple(result)


class SignedAverage:
    """Map each monomial to its signed orbit average under S8 and conjugation.

    The average of a surviving representative remains a polynomial bounded
    by 1 on the unit box because each signed relabeling preserves the box.
    """

    def __init__(self):
        self._orbits = {(): ((), 1)}

    def representative(self, monomial):
        if sum(v % 2 for v in monomial) % 2:
            return (), 0
        edges = [VARIABLES[v] for v in monomial]
        used = sorted({v for _, i, j in edges for v in (i, j)})
        relabel = {v: i for i, v in enumerate(used)}
        compact = tuple(sorted(EDGE_INDEX[relabel[i], relabel[j]] + (kind == "y") for kind, i, j in edges))
        known = self._orbits.get(compact)
        if known is not None:
            return known

        local_index = {edge: 2*i for i, edge in enumerate(combinations(range(len(used)), 2))}
        sources = [local_index[relabel[i], relabel[j]] + (kind == "y") for kind, i, j in edges]
        seen = {}
        vanishes = False
        for action in actions(len(used)):
            image = [action[i] for i in sources]
            target = tuple(sorted(abs(i) - 1 for i in image))
            sign = (-1)**sum(i < 0 for i in image)
            if seen.setdefault(target, sign) != sign:
                vanishes = True
        canonical = min(seen)
        for target, sign in seen.items():
            self._orbits[target] = (canonical, 0 if vanishes else sign*seen[canonical])
        return self._orbits[compact]


def compressed_psd_bases(quadratic_monomials):
    """Orbit character projections for tail S4 and two head transpositions."""
    from itertools import product

    # Enumerate the entire subgroup. Retain one nonzero projection per orbit.
    head = [(0, 1, 2, 3), (1, 0, 2, 3), (0, 1, 3, 2), (1, 0, 3, 2)]
    subgroup = []
    for switches, first_four in enumerate(head):
        for tail in permutations((4, 5, 6, 7)):
            permutation = first_four + tail
            mapping = []
            for kind, i, j in VARIABLES:
                a, b = permutation[i], permutation[j]
                edge_index = EDGE_INDEX[min(a, b), max(a, b)]
                sign = -1 if kind == "y" and a > b else 1
                mapping.append((edge_index + (kind == "y"), sign))
            subgroup.append((switches, mapping))
    characters = list(product((1, -1), repeat=3))
    blocks = {character: [] for character in characters}
    visited = set()
    for term in quadratic_monomials:
        if term in visited:
            continue
        orbit = []
        for switches, mapping in subgroup:
            mapped = [mapping[i] for i in term]
            target = tuple(sorted(i for i, _ in mapped))
            sign = (-1)**sum(s < 0 for _, s in mapped)
            orbit.append((switches, target, sign))
            visited.add(target)
        conjugation = (-1)**sum(i % 2 for i in term)
        for first, second, parity in characters:
            if parity != conjugation:
                continue
            projection = {}
            for switches, target, sign in orbit:
                if switches & 1:
                    sign *= first
                if switches & 2:
                    sign *= second
                if projection.setdefault(target, sign) != sign:
                    break
            else:
                blocks[first, second, parity].append(projection)
    return blocks


def reanchored_equality(witness, lookup):
    """Expand a real phase word before and after exchanging the anchor."""
    from collections import Counter
    from itertools import product

    anchor, divisor = witness["new_anchor"], witness["divisor"]
    if type(anchor) is not int or not 0 <= anchor < 8 or type(divisor) is not int or not divisor:
        raise ValueError("invalid reanchor witness")
    word = witness["word"]
    if not isinstance(word, list) or not 1 <= len(word) <= 4 or any(
        not isinstance(edge, list) or len(edge) != 2 or edge[0] == edge[1]
        or any(type(v) is not int or not 0 <= v < 8 for v in edge) for edge in word
    ):
        raise ValueError("invalid reanchor word")

    # On a dephased ETF the directed phase from the old anchor (8) is 1.
    # The new gauge sends z_ij to z_{anchor,s(i)} z_{s(i),s(j)} z_{s(j),anchor}.
    permutation = list(range(9))
    permutation[8], permutation[anchor] = anchor, 8
    exponents = Counter()
    for left, right in word:
        path = [anchor, permutation[left], permutation[right], anchor]
        for a, b in zip(path, path[1:]):
            if a != b and 8 not in (a, b):
                exponents[min(a, b), max(a, b)] += 1 if a < b else -1
    transformed = []
    for (i, j), exponent in exponents.items():
        transformed.extend([(i, j) if exponent > 0 else (j, i)] * abs(exponent))
    if len(transformed) > 4:
        raise ValueError("reanchored word exceeds degree four")

    coefficients = Counter()
    for sign, directed in ((1, word), (-1, transformed)):
        for choices in product((0, 1), repeat=len(directed)):
            imag_count = sum(choices)
            if imag_count % 2:
                continue
            weight = sign * (-1)**(imag_count//2)
            power = []
            for (a, b), imag in zip(directed, choices):
                power.append(EDGE_INDEX[min(a, b), max(a, b)] + imag)
                if imag and a > b:
                    weight = -weight
            index, orientation = lookup(tuple(sorted(power)))
            if orientation:
                coefficients[index] += orientation * weight
    if any(c % divisor for c in coefficients.values()):
        raise ValueError("reanchor witness divisor does not yield integer coefficients")
    return tuple((index, c//divisor, 0) for index, c in sorted(coefficients.items()) if c)


def audit_anchored_primal(candidate, model, exact_psd, field):
    """Independently test normalization, the full equality span and PSD."""
    from collections import defaultdict
    from itertools import combinations_with_replacement
    from math import gcd, lcm
    from sympy.polys.matrices import DomainMatrix
    from verify_sdp_factor_cert import monomial, rational

    coordinate_basis = ["1", "sqrt(10)/5"]
    if candidate.get("coefficient_basis") != coordinate_basis or model.get("coefficient_basis") != coordinate_basis:
        raise ValueError("unexpected anchored moment coefficient basis")
    if not anchored_variable_order(model["variables"]):
        raise ValueError("unexpected anchored ETF variable order")
    representatives = [monomial(row, len(VARIABLES)) for row in model["moments"]]
    raw_values = candidate["moments"]
    if len(raw_values) != len(representatives):
        raise ValueError("moment coordinate vector has the wrong length")
    if any(not isinstance(row, list) or len(row) != 2 for row in raw_values):
        raise ValueError("each moment needs two rational coordinates")
    values = [tuple(map(rational, row)) for row in raw_values]
    if not representatives or representatives[0] != () or values[0] != (1, 0):
        raise ValueError("constant moment must equal 1")
    signed = SignedAverage()
    representative_map = {}
    for i, power in enumerate(representatives):
        canonical, sign = signed.representative(power)
        if len(power) > 4 or not sign or canonical in representative_map:
            raise ValueError("invalid or duplicate signed orbit representative")
        representative_map[canonical] = (i, sign)

    @cache
    def lookup(power):
        canonical, sign = signed.representative(power)
        if sign == 0:
            return None, 0
        if canonical not in representative_map:
            raise ValueError(f"missing moment orbit for {power}")
        index, source_sign = representative_map[canonical]
        return index, sign * source_sign

    beta = field.scalar([0, "1/5"])
    domain = field.domain

    def as_number(a, b):
        return domain.convert(sp.QQ(a)) + domain.convert(sp.QQ(b)) * beta

    denominator = lcm(*(c.denominator for coordinates in values for c in coordinates))
    scaled = [tuple(int(c * denominator) for c in coordinates) for coordinates in values]

    symbols, equations = canonical_anchored_equations()
    degrees = [tuple(combinations_with_replacement(range(len(VARIABLES)), d)) for d in range(4)]
    equation_rows = set()
    for expression in equations.values():
        terms = []
        for exponents, coefficient in sp.Poly(expression, *symbols).terms():
            if not coefficient:
                continue
            power = tuple(i for i, degree in enumerate(exponents) for _ in range(degree))
            a, b = field.components(domain.from_sympy(coefficient))
            terms.append((power, (int(a), int(5*b))))
        max_degree = max(len(power) for power, _ in terms)
        for degree in range(5 - max_degree):
            for multiplier in degrees[degree]:
                row = defaultdict(lambda: [0, 0])
                for source, (a, b) in terms:
                    index, sign = lookup(tuple(sorted(source + multiplier)))
                    if sign:
                        row[index][0] += sign*a
                        row[index][1] += sign*b
                nonzero = sorted((index, a, b) for index, (a, b) in row.items() if a or b)
                if nonzero:
                    divisor = gcd(*(c for _, a, b in nonzero for c in (a, b)))
                    if (nonzero[0][1], nonzero[0][2]) < (0, 0):
                        divisor = -divisor
                    equation_rows.add(tuple((index, a//divisor, b//divisor) for index, a, b in nonzero))
    base_rows = len(equation_rows)
    overlay_rows = set()
    if model.get("extra_equality_count", 0):
        equations_and_witnesses = list(zip(model["equalities"], model["equality_witnesses"], strict=True))
        for declared, witness in equations_and_witnesses:
            if not isinstance(witness, dict):
                continue
            reconstructed = reanchored_equality(witness, lookup)
            serialized = tuple(sorted((int(index), *pair) for index, pair in declared.items()))
            if not reconstructed or serialized != reconstructed:
                raise ValueError("reanchor equality does not match its independently expanded witness")
            overlay_rows.add(reconstructed)
        if len(overlay_rows) != model["extra_equality_count"]:
            raise ValueError("unexpected number of verified reanchor witnesses")
    combined = equation_rows | overlay_rows
    failures = 0
    matrix_rows = []
    for source in sorted(combined):
        a_sum = b_sum = 0
        row = [domain.zero] * len(values)
        for index, a, b in source:
            c, d = scaled[index]
            a_sum += 5*a*c + 2*b*d
            b_sum += a*d + b*c
            row[index] = as_number(a, b)
        if a_sum or b_sum:
            failures += 1
        matrix_rows.append(row)
    normalization = [domain.zero] * len(values)
    normalization[0] = domain.one
    shape = len(matrix_rows), len(values)
    rank = DomainMatrix(matrix_rows, shape, domain).rank()
    rank_with_norm = DomainMatrix([*matrix_rows, normalization], (shape[0]+1, shape[1]), domain).rank()

    # For any negative eigenspace of the full invariant moment form, a
    # coordinate monomial with nonzero spectral projection can be relabeled
    # into the first four vertices. That projected vector is fixed by tail
    # S4. The two transpositions and conjugation then give orthogonal blocks.
    blocks = []
    psd_bases = compressed_psd_bases([*degrees[0], *degrees[1], *degrees[2]])
    for character, basis in psd_bases.items():
        matrix = [[domain.zero for _ in basis] for _ in basis]
        for i, first in enumerate(basis):
            for j, second in enumerate(basis[:i+1]):
                a = b = 0
                for f, fc in first.items():
                    for g, gc in second.items():
                        index, sign = lookup(tuple(sorted(f + g)))
                        if sign:
                            c, d = scaled[index]
                            weight = fc * gc * sign
                            a += weight * c
                            b += weight * d
                matrix[i][j] = matrix[j][i] = as_number(a, b)
        blocks.append({"character": list(character), "size": len(basis), **exact_psd(matrix, field)})

    # Ensure all degree-four monomials have a moment, including those not
    # referenced explicitly in the equation or compressed PSD checks.
    for degree in degrees:
        for power in degree:
            lookup(power)
    for power in combinations_with_replacement(range(len(VARIABLES)), 4):
        lookup(power)
    holds = failures == 0 and all(block["holds"] for block in blocks)
    scope = "complete S8 and conjugation invariant order-two anchored ETF(4,9) moment relaxation"
    if overlay_rows:
        scope += " with checked reanchor identities"
    return {"mode": "exact_anchored_primal", "passed": holds,
            "relaxation_feasibility_verified": holds, "scope": scope,
            "moment_coordinates": len(values), "normalization_holds": True,
            "equalities": {"distinct_rows": len(combined), "base_rows": base_rows,
                           "reanchor_witnesses_checked": len(overlay_rows), "rank": rank,
                           "rank_with_normalization": rank_with_norm,
                           "degrees_of_freedom": len(values) - rank_with_norm, "failed_rows": failures},
            "psd": {"holds": all(block["holds"] for block in blocks), "blocks": blocks}}
