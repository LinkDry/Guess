"""Exact verification of rounded SOS factors and biquadratic box bounds."""

from collections import defaultdict
from fractions import Fraction
from itertools import combinations
from math import isqrt, lcm
import re

import numpy as np
import sympy as sp


COEFFICIENT_BASIS = ["1", "sqrt(2)", "sqrt(5)", "sqrt(10)"]
ROOT_DENOMINATOR = 10**30
RATIONAL = re.compile(r"[+-]?[0-9]+(?:/[1-9][0-9]*)?")


def rational(value):
    if type(value) is int or (isinstance(value, str) and RATIONAL.fullmatch(value)):
        return Fraction(value)
    raise ValueError(f"invalid exact rational coefficient {value!r}")


def monomial(indices, size):
    if not isinstance(indices, list) or any(type(i) is not int or not 0 <= i < size for i in indices):
        raise ValueError("invalid monomial variable index")
    if sorted(indices) != indices:
        raise ValueError("monomial indices must be in sorted order")
    return tuple(indices)


def polynomial(terms, size, parse_coefficient):
    if not isinstance(terms, list):
        raise ValueError("polynomial must contain a list of terms")
    result = {}
    for term in terms:
        if not isinstance(term, list) or len(term) != 2:
            raise ValueError("invalid polynomial term")
        key = monomial(term[0], size)
        if key in result:
            raise ValueError("duplicate polynomial monomial")
        result[key] = parse_coefficient(term[1])
    return result


def coefficient(value):
    if not isinstance(value, list) or len(value) != len(COEFFICIENT_BASIS):
        raise ValueError("expected four rational biquadratic coordinates")
    return tuple(map(rational, value))


def integer_polynomial(poly, denominator):
    return {mon: val.numerator * (denominator // val.denominator) for mon, val in poly.items() if val}


def add_psd_block(residual, block, size, scale):
    basis = [monomial(indices, size) for indices in block["basis"]]
    rows = block["factor_integer_rows"]
    if not basis or len(set(basis)) != len(basis) or not isinstance(rows, list) or len(rows) != len(basis):
        raise ValueError("factor must have one row per distinct basis monomial")
    if not isinstance(rows[0], list) or not rows[0]:
        raise ValueError("factor must have at least one column")
    width = len(rows[0])
    if any(not isinstance(row, list) or len(row) != width or any(type(x) is not int for x in row) for row in rows):
        raise ValueError("factor must be a rectangular integer matrix")

    # The object dtype uses Python integers, so both multiplication and addition are exact.
    factor = np.array(rows, dtype=object)
    gram = factor @ factor.T
    for i, m_i in enumerate(basis):
        for j, m_j in enumerate(basis[:i+1]):
            entry = gram[i, j]
            if entry:
                power = tuple(sorted(m_i + m_j))
                residual[power][0] -= scale * entry * (1 if i == j else 2)
    return {"basis_size": len(basis), "factor_columns": width}


def canonical_equations(variables, branch):
    """Derive necessary equations directly in the real pair coordinates.

    Each exchanged pair is lifted to sqrt(2)*Re(z), sqrt(2)*Im(z).
    After gauging its cross entry with fixed line zero to the positive real
    Welch value, the pair cross blocks of the real Gram follow from their
    two independent complex entries. Every encoded phase belongs to a unit
    circle. ETF tightness requires R^2=(9/4)R and rank R=4. In the fixed3
    cases the invertible 3-by-3 fixed block gives a rank-one Schur complement.
    """
    fixed = 1 if branch == "fixed1" else 3
    sign = -1 if branch == "fixed3_negative" else 1
    phase_count = 16 if fixed == 1 else 15
    if variables != [f"x{i}" for i in range(2*phase_count)]:
        raise ValueError(f"unexpected phase variable order for {branch}")
    symbols = sp.symbols(f"x:{2*phase_count}", real=True)
    mu = sp.sqrt(10)/8
    gram = sp.eye(9)
    equations = {}
    next_phase = 0

    def phase():
        nonlocal next_phase
        x, y = symbols[2*next_phase:2*next_phase+2]
        equations[f"circle_{next_phase}"] = x*x + y*y - 1
        next_phase += 1
        return mu*x, mu*y

    def put(i, j, entry):
        gram[i, j] = gram[j, i] = entry

    for i, j in combinations(range(fixed), 2):
        put(i, j, sign*mu if (i, j) == (1, 2) else mu)
    for a in range(fixed, 9, 2):
        real, imag = phase()
        gram[a, a], gram[a+1, a+1] = 1+real, 1-real
        put(a, a+1, -imag)
        put(0, a, sp.sqrt(2)*mu)
        for i in range(1, fixed):
            real, imag = phase()
            put(i, a, sp.sqrt(2)*real)
            put(i, a+1, sp.sqrt(2)*imag)
        for b in range(a+2, 9, 2):
            r_z, i_z = phase()
            r_w, i_w = phase()
            put(a, b, r_z+r_w)
            put(a, b+1, i_z-i_w)
            put(a+1, b, -i_z-i_w)
            put(a+1, b+1, r_z-r_w)
    if next_phase != phase_count:
        raise RuntimeError("incorrect canonical phase count")

    tight = gram*gram - sp.Rational(9, 4)*gram
    for i in range(9):
        for j in range(i, 9):
            equations[f"tight_{i}_{j}"] = tight[i, j]
    if fixed == 3:
        # The fixed triangle can be switched to a constant-sign triangle.
        switch = sp.diag(1, sign, sign)
        cross = switch * gram[:3, 3:]
        t = sign * mu
        inv_a = sp.radsimp(1/(1-t))
        inv_b = sp.radsimp(-t/((1-t)*(1+2*t)))
        inverse = inv_a * sp.eye(3) + inv_b * sp.ones(3)
        schur = (gram[3:, 3:] - cross.T*inverse*cross).applyfunc(sp.expand)
        pairs = list(combinations(range(6), 2))
        for (a, b), (i, j) in combinations_with_diagonal(pairs):
            equations[f"schur_{a}{b}_{i}{j}"] = schur[a, i]*schur[b, j] - schur[a, j]*schur[b, i]
    return symbols, equations


def combinations_with_diagonal(items):
    for i, left in enumerate(items):
        for right in items[i:]:
            yield left, right


def check_provenance(variables, branch, used_equations):
    if branch not in {"fixed1", "fixed3_negative", "fixed3_positive"}:
        return {"status": "unchecked", "equations_checked": 0}
    symbols, canonical = canonical_equations(variables, branch)
    radicals = [sp.sqrt(r) for r in (2, 5, 10)]
    field = sp.QQ.algebraic_field(sum(radicals[:2]))
    roots = [field.one] + [field.from_sympy(root) for root in radicals]
    missing, mismatched = [], []
    # Canonical coefficients repeat heavily, including in the Schur minors.
    coordinate_cache = {}
    expression_cache = {}
    for name, supplied in sorted(used_equations.items()):
        if name not in canonical:
            missing.append(name)
            continue
        expected = {}
        for exponent, value in sp.Poly(canonical[name].expand(), *symbols).terms():
            if value:
                power = tuple(i for i, degree in enumerate(exponent) for _ in range(degree))
                if value not in expression_cache:
                    expression_cache[value] = field.from_sympy(value)
                expected[power] = expression_cache[value]
        converted = {}
        for power, components in supplied.items():
            if any(components):
                if components not in coordinate_cache:
                    coordinate_cache[components] = sum(
                        (field.convert(sp.QQ(c.numerator, c.denominator))*root for c, root in zip(components, roots)),
                        field.zero,
                    )
                converted[power] = coordinate_cache[components]
        if converted != expected:
            mismatched.append(name)
    scope = ("ETF(4,9), antiunitary cycle type 2^4 1" if branch == "fixed1"
             else f"ETF(4,9), antiunitary cycle type 2^3 1^3, {'negative' if branch.endswith('negative') else 'positive'} triangle")
    return {"status": "mismatch" if missing or mismatched else "matched", "scope": scope,
            "source": "independent_real_gram_and_schur_equations", "equations_checked": len(used_equations),
            "unknown_equations": missing, "mismatched_equations": mismatched}


def verify_factor_certificate(cert, model):
    if model.get("schema") != "search-sdp-model-v1":
        raise ValueError("expected search-sdp-model-v1")
    if cert.get("coefficient_basis") != COEFFICIENT_BASIS or model.get("coefficient_basis") != COEFFICIENT_BASIS:
        raise ValueError("unexpected biquadratic coefficient basis")
    variables = model["variables"]
    if (not isinstance(variables, list) or not variables or
        any(not isinstance(v, str) or not v for v in variables) or len(set(variables)) != len(variables)):
        raise ValueError("variables must be distinct nonempty strings")
    size = len(variables)
    factor_denominator = cert["factor_denominator"]
    if type(factor_denominator) is not int or factor_denominator < 1:
        raise ValueError("factor_denominator must be a positive integer")
    if not isinstance(cert["multipliers"], dict) or not isinstance(model["equations"], dict):
        raise ValueError("expected ID-indexed multipliers and equations")
    if not cert["multipliers"].keys() <= model["equations"].keys():
        raise ValueError("a multiplier refers to an unknown equation")
    multipliers = {name: polynomial(rows, size, rational) for name, rows in cert["multipliers"].items()}
    multipliers = {name: poly for name, poly in multipliers.items() if any(poly.values())}
    equations = {name: polynomial(model["equations"][name], size, coefficient) for name in multipliers}
    multiplier_denominator = lcm(*(v.denominator for poly in multipliers.values() for v in poly.values()))
    equation_denominator = lcm(*(v.denominator for poly in equations.values() for row in poly.values() for v in row))
    scale = lcm(multiplier_denominator * equation_denominator, factor_denominator**2)
    multiplier_scale = scale // (multiplier_denominator * equation_denominator)
    residual = defaultdict(lambda: [0, 0, 0, 0])
    residual[()][0] = -scale
    for name, poly in equations.items():
        weights = integer_polynomial(multipliers[name], multiplier_denominator)
        for source_power, coordinates in poly.items():
            components = [(k, c.numerator * (equation_denominator // c.denominator) * multiplier_scale)
                          for k, c in enumerate(coordinates) if c]
            for multiplier_power, weight in weights.items():
                target = residual[tuple(sorted(source_power + multiplier_power))]
                for k, c in components:
                    target[k] -= c * weight

    blocks = cert["psd_blocks"]
    if not isinstance(blocks, list) or any(not isinstance(block, dict) for block in blocks):
        raise ValueError("PSD blocks must be an array of factor objects")
    psd = [add_psd_block(residual, block, size, scale // factor_denominator**2) for block in blocks]
    # Each root lies between floor/d and (floor+1)/d by integer square root.
    # Bound each residual coefficient after combining all four exact coordinates.
    root_floors = [isqrt(r * ROOT_DENOMINATOR**2) for r in (2, 5, 10)]
    bound_numerator = 0
    nonzero_terms = 0
    maximum_degree = 0
    for power, (constant, *irrational) in residual.items():
        if not constant and not any(irrational):
            continue
        nonzero_terms += 1
        maximum_degree = max(maximum_degree, len(power))
        lower = constant * ROOT_DENOMINATOR
        upper = lower
        for c, floor in zip(irrational, root_floors):
            lower += c * floor + min(0, c)
            upper += c * floor + max(0, c)
        bound_numerator += max(abs(lower), abs(upper))
    bound = Fraction(bound_numerator, scale * ROOT_DENOMINATOR)
    # Every monomial has magnitude <=1 on the box. At a common zero of the
    # equations, -1 = sum_of_squares + residual is impossible if bound < 1.
    refuted = bound < 1
    provenance = check_provenance(variables, model.get("branch"), equations)
    report = {"mode": "exact_rational_intervals", "branch": model.get("branch"),
              "passed": refuted and provenance["status"] != "mismatch",
              "polynomial_system_refuted_on_unit_box": refuted,
              "etf_nonexistence_verified": refuted and provenance["status"] == "matched",
              "constraint_provenance": provenance, "psd_by_exact_factorization": True,
              "psd_blocks": psd, "factor_denominator": factor_denominator,
              "residual_upper_bound_on_unit_box": str(bound), "residual_bound_approx": float(bound),
              "residual_terms": nonzero_terms, "residual_maximum_degree": maximum_degree,
              "radical_interval_denominator": ROOT_DENOMINATOR}
    if "residual_upper_bound_on_unit_box" in cert:
        try:
            report["producer_bound_matches"] = rational(cert["residual_upper_bound_on_unit_box"]) == bound
        except ValueError:
            report["producer_bound_matches"] = False
    return report
