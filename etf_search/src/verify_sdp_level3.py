"""Independent exact checks for targeted degree-six anchored moment artifacts."""

from collections import defaultdict
from fractions import Fraction
from functools import cache, lru_cache
from itertools import combinations_with_replacement
from math import gcd, isqrt, lcm

import numpy as np
import sympy as sp

from verify_sdp_anchored import (
    VARIABLES, SignedAverage, anchored_variable_order, canonical_anchored_equations,
    compressed_psd_bases, named_rank_equation, reanchored_equality,
)
from verify_sdp_factor_cert import monomial, rational


BASIS = ["1", "sqrt(10)/5"]
ROOT_SCALE = 10**30


def pair(value):
    if type(value) is int:
        return Fraction(value), Fraction(0)
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("expected two rational coefficients")
    return tuple(map(rational, value))


def multiply_coefficients(left, right):
    a, b = left
    c, d = right
    return a*c + Fraction(2, 5)*b*d, a*d + b*c


def row_terms(row, size):
    if not isinstance(row, dict):
        raise ValueError("moment row must map indices to coefficients")
    result = {}
    for key, value in row.items():
        index = int(key)
        if str(index) != key or not 0 <= index < size:
            raise ValueError("invalid moment row index")
        coefficients = pair(value)
        if any(coefficients):
            result[index] = coefficients
    return result


def beta_interval(coefficients):
    a, b = coefficients
    root = isqrt(10 * ROOT_SCALE**2)
    endpoints = [a + b * Fraction(root + delta, 5 * ROOT_SCALE) for delta in (0, 1)]
    return min(endpoints), max(endpoints)


@cache
def equation_terms(name):
    symbols, base = canonical_anchored_equations()
    value = base.get(name)
    if value is None:
        value = named_rank_equation(name)
    if value is None or value == 0:
        raise ValueError(f"no canonical anchored equation for {name!r}")
    terms = {}
    for exponents, coefficient in sp.Poly(value.expand(), *symbols).terms():
        if not coefficient:
            continue
        a, b = coefficient.subs(sp.sqrt(10), 0), coefficient.coeff(sp.sqrt(10))
        if not a.is_Rational or not b.is_Rational or a + b*sp.sqrt(10) != coefficient:
            raise ValueError("unexpected coefficient outside Q(beta)")
        power = tuple(i for i, exponent in enumerate(exponents) for _ in range(exponent))
        terms[power] = Fraction(a), Fraction(5*b)
    return terms


class MomentManifest:
    def __init__(self, model):
        if model.get("coefficient_basis") != BASIS or not anchored_variable_order(model["variables"]):
            raise ValueError("unexpected anchored variable order or coefficient basis")
        self.model = model
        self.average = SignedAverage()
        self.representatives = [monomial(word, len(VARIABLES)) for word in model["moments"]]
        if not self.representatives or self.representatives[0] != ():
            raise ValueError("moment zero must be the constant")
        self.indices = {}
        for index, word in enumerate(self.representatives):
            representative, sign = self.average.representative(word)
            if len(word) > 6 or not sign or representative in self.indices:
                raise ValueError("invalid or duplicate signed orbit representative")
            self.indices[representative] = index, sign

    @lru_cache(maxsize=300_000)
    def lookup(self, word):
        representative, sign = self.average.representative(word)
        if not sign:
            return None, 0
        if representative not in self.indices:
            raise ValueError(f"missing moment for monomial {word}")
        index, orientation = self.indices[representative]
        return index, orientation * sign

    def row(self, witness):
        if isinstance(witness, dict):
            if witness.get("type") == "fixed_moment":
                word = monomial(witness["monomial"], len(VARIABLES))
                index, orientation = self.lookup(word)
                if index is None or not word:
                    raise ValueError("fixed moment witness must select a nonconstant surviving orbit")
                a, b = pair(witness["value"])
                scale = lcm(a.denominator, b.denominator)
                return {index: (scale * orientation, 0), 0: (-scale*a, -scale*b)}, "fixed_moment"
            return {i: (a, b) for i, a, b in reanchored_equality(witness, self.lookup)}, "reanchor"
        name, power, divisor = witness
        power = monomial(power, len(VARIABLES))
        equation = equation_terms(name)
        if type(divisor) is not int or divisor == 0 or len(power) + max(map(len, equation)) > 6:
            raise ValueError("invalid equality scaling or truncation degree")
        accum = defaultdict(lambda: [0, 0])
        for term, (a, b) in equation.items():
            index, sign = self.lookup(tuple(sorted(term + power)))
            if sign:
                accum[index][0] += sign*a/divisor
                accum[index][1] += sign*b/divisor
        kind = "rank" if name.startswith(("gram5_", "schur4_")) else "base"
        return {i: tuple(c) for i, c in accum.items() if any(c)}, kind

    def check_rows(self):
        rows = [{0: (1, 0)}]
        kinds = defaultdict(int)
        rank_names = set()
        for supplied, witness in zip(self.model["equalities"], self.model["equality_witnesses"], strict=True):
            reconstructed, kind = self.row(witness)
            reconstructed = {i: v for i, v in reconstructed.items() if any(v)}
            if not reconstructed or row_terms(supplied, len(self.representatives)) != reconstructed:
                raise ValueError(f"equality row {len(rows)} disagrees with its independent expansion")
            rows.append(reconstructed)
            kinds[kind] += 1
            if kind == "rank":
                rank_names.add(witness[0])
        return rows, dict(kinds), sorted(rank_names)

    def full_product(self, left, right):
        totals = defaultdict(int)
        for i, ci in left.items():
            for j, cj in right.items():
                index, sign = self.lookup(tuple(sorted(i+j)))
                if sign:
                    totals[index] += sign*ci*cj
        return {i: c for i, c in totals.items() if c}

    def character_product(self, left, right):
        # This shortcut applies to the independently constructed orbit bases.
        # All terms of the left orbit have the same paired contribution when
        # left and right transform under the same real character.
        word, orientation = next(iter(left.items()))
        return self.full_product({word: orientation * len(left)}, right)


def declared_polynomials(block):
    result = []
    for polynomial in block["basis"]:
        terms = {}
        for word, coefficient in polynomial:
            power = monomial(word, len(VARIABLES))
            if len(power) > 3 or power in terms or type(coefficient) is not int or not coefficient:
                raise ValueError("invalid integer polynomial in PSD basis")
            terms[power] = coefficient
        if not terms:
            raise ValueError("empty polynomial in PSD basis")
        result.append(terms)
    return result


def checked_face_rows(manifest):
    model = manifest.model
    if "face_reduction" not in model:
        return (*manifest.check_rows(), None)
    from verify_sdp_face import check_exact_face

    audit = check_exact_face(model)
    if not audit["passed"]:
        raise ValueError("facial equivalence to original relaxation was not verified")
    count = model["face_reduction"]["input_equality_count"]
    original = {**model, "equalities": model["equalities"][:count],
                "equality_witnesses": model["equality_witnesses"][:count]}
    rows, kinds, names = MomentManifest(original).check_rows()
    kinds["face_kernel"] = len(model["equalities"]) - count
    rows.extend(row_terms(row, len(manifest.representatives)) for row in model["equalities"][count:])
    return rows, kinds, names, audit


def verify_moment_dual(candidate, model):
    if candidate["coefficient_basis"] != BASIS:
        raise ValueError("unexpected dual coefficient basis")
    manifest = MomentManifest(model)
    equality_rows, kinds, rank_names, face = checked_face_rows(manifest)
    equality_dual = [rational(v) for v in candidate["equality_dual"]]
    denominator = candidate["psd_factor_denominator"]
    if type(denominator) is not int or denominator <= 0 or len(equality_dual) != len(equality_rows):
        raise ValueError("invalid dual length or factor denominator")
    residual = defaultdict(lambda: [Fraction(0), Fraction(0)])
    for multiplier, row in zip(equality_dual, equality_rows):
        for index, (a, b) in row.items():
            residual[index][0] += multiplier*a
            residual[index][1] += multiplier*b
    sizes = []
    for block, rows in zip(model["psd_blocks"], candidate["psd_factor_integer_rows"], strict=True):
        basis = declared_polynomials(block)
        if len(rows) != len(basis) or not rows or any(
            len(row) != len(rows[0]) or any(type(v) is not int for v in row) for row in rows
        ):
            raise ValueError("PSD factor must be a rectangular integer matrix")
        factor = np.array(rows, dtype=object)
        gram = factor @ factor.T
        for i, left in enumerate(basis):
            for j, right in enumerate(basis[:i+1]):
                if gram[i,j]:
                    weight = Fraction(gram[i,j] * (1 if i == j else 2), denominator**2)
                    product = (manifest.character_product(left, right) if face is not None
                               else manifest.full_product(left, right))
                    for index, coefficient in product.items():
                        residual[index][0] -= weight*coefficient
        sizes.append(len(basis))
    # Equality row zero is m_empty=1. Both sides of this inequality are
    # evaluated with exact rational enclosures for the real root beta.
    bound = sum(max(map(abs, beta_interval(coefficients))) for coefficients in residual.values())
    margin = -equality_dual[0] - bound
    conditional = kinds.get("fixed_moment", 0) != 0
    return {"passed": margin > 0, "mode": "exact_anchored_moment_dual", "conditional": conditional,
            "relaxation_refuted_on_unit_box": margin > 0, "etf_nonexistence_verified": margin > 0 and not conditional,
            "equality_rows_verified": kinds, "rank_equations_verified": rank_names,
            "psd_factor_block_sizes": sizes, "residual_upper_bound": str(bound), "margin": str(margin),
            **({"structural_face_equivalence_verified": True} if face is not None else {})}


@cache
def selected_monomials(stage):
    if stage not in ("support3", "support4", "support5", "support6"):
        raise ValueError("unknown cubic support restriction")
    cutoff = int(stage[-1])
    monomials = [()]
    for degree in range(1, 4):
        for power in combinations_with_replacement(range(len(VARIABLES)), degree):
            if degree < 3 or len({v for i in power for v in VARIABLES[i][1:]}) <= cutoff:
                monomials.append(power)
    return tuple(monomials)


def canonical_psd_bases(stage):
    return compressed_psd_bases(selected_monomials(stage), fixed_vertices=max(4, int(stage[-1])))


def validate_polynomial_manifest(model):
    from verify_sdp_factor_cert import polynomial

    supplied = model.get("base_equations", {})
    for name, terms in supplied.items():
        parsed = polynomial(terms, len(VARIABLES), pair)
        if {key: c for key, c in parsed.items() if any(c)} != equation_terms(name):
            raise ValueError(f"declared polynomial {name} does not match the canonical equation")
    return len(supplied)


def verify_degree6_primal(candidate, model, exact_psd, field):
    if candidate["coefficient_basis"] != BASIS:
        raise ValueError("unexpected primal coefficient basis")
    manifest = MomentManifest(model)
    values = [pair(p) for p in candidate["moments"]]
    if len(values) != len(manifest.representatives) or values[0] != (1, 0):
        raise ValueError("constant moment must equal 1 and moment count must match")
    rows, kinds, rank_names, face = checked_face_rows(manifest)
    polynomial_count = validate_polynomial_manifest(model)
    failures = []
    for number, row in enumerate(rows[1:], 1):
        a = b = 0
        for index, coefficient in row.items():
            c, d = multiply_coefficients(coefficient, values[index])
            a += c
            b += d
        if a or b:
            failures.append(number)

    stage = model["stage"]
    monomials = selected_monomials(stage)
    bases = canonical_psd_bases(stage)
    beta = field.scalar([0, "1/5"])
    domain = field.domain
    scale = lcm(*(v.denominator for p in values for v in p))
    scaled = [tuple(int(v*scale) for v in p) for p in values]
    blocks = []
    for character, basis in bases.items():
        matrix = [[domain.zero] * len(basis) for _ in basis]
        for i, left in enumerate(basis):
            for j, right in enumerate(basis[:i+1]):
                a = b = 0
                for index, weight in manifest.character_product(left, right).items():
                    c, d = scaled[index]
                    a += c * weight
                    b += d * weight
                matrix[i][j] = matrix[j][i] = domain.convert(a) + domain.convert(b)*beta
        blocks.append({"character": list(character), "size": len(basis), **exact_psd(matrix, field)})
    passed = not failures and all(block["holds"] for block in blocks)
    return {"passed": passed, "mode": "exact_anchored_degree6_primal", "stage": stage,
            "cubics": len(monomials) - (1 + 56 + 1596), "psd": blocks,
            "relaxation_feasibility_verified": passed, "equality_rows_verified": kinds,
            "rank_equations_verified": rank_names, "failed_row_numbers": failures,
            "declared_polynomials_verified": polynomial_count,
            **({"structural_face_equivalence_verified": True} if face is not None else {}),
            "scope": "Targeted degree six with selected cubic support and manifest equality witnesses"}


def audit_model_numerically(model, coordinates=None, validate_psd_maps=True):
    """Check every manifest equation and optionally replay the primal vector."""
    manifest = MomentManifest(model)
    rows, kinds, rank_names = manifest.check_rows()
    polynomial_count = validate_polynomial_manifest(model)
    stage = model["stage"]
    monomials = selected_monomials(stage)
    report = {"passed": True, "mode": "exact_degree6_model_provenance", "stage": stage,
              "moment_coordinates": len(manifest.representatives),
              "declared_polynomials_verified": polynomial_count, "equality_rows_verified": kinds,
              "rank_equations_verified": rank_names, "cubic_monomials": len(monomials)-1653}
    if coordinates is not None:
        if coordinates.shape != (len(manifest.representatives),) or not np.isfinite(coordinates).all():
            raise ValueError("invalid numeric primal moment vector")
        beta = np.sqrt(10.0)/5
        equality_errors = defaultdict(float)
        scaled_errors = defaultdict(float)
        for row, witness in zip(rows[1:], model["equality_witnesses"], strict=True):
            name = witness.get("type", "reanchor") if isinstance(witness, dict) else witness[0].split("_")[0]
            value = sum((float(a) + float(b)*beta)*coordinates[i] for i, (a, b) in row.items())
            error = float(abs(value))
            scale = max(abs(coefficient) for pair in row.values() for coefficient in pair)
            equality_errors[name] = max(equality_errors[name], error)
            scaled_errors[name] = max(scaled_errors[name], error/float(scale))
        report.update(mode="numeric_degree6_primal_diagnostic", normalization_error=float(abs(coordinates[0]-1)),
                      equality_errors=dict(equality_errors), row_scaled_equality_errors=dict(scaled_errors))
    if not validate_psd_maps:
        return report

    canonical_bases = canonical_psd_bases(stage)
    blocks = []
    characters = set()
    for supplied in model["psd_blocks"]:
        character = tuple(supplied["character"])
        if character not in canonical_bases or character in characters:
            raise ValueError("invalid or duplicate PSD block character")
        characters.add(character)
        basis = declared_polynomials(supplied)
        expected = canonical_bases[character]
        signed_basis = {frozenset(poly.items()) for poly in expected}
        signed_basis.update(frozenset((key, -value) for key, value in poly.items()) for poly in expected)
        if (len(expected) != len(basis) or any(frozenset(poly.items()) not in signed_basis for poly in basis)
                or len({frozenset(poly) for poly in basis}) != len(basis)):
            raise ValueError("PSD basis does not span the independently generated character space")
        declared = {}
        for i, j, entry in supplied["entries"]:
            if type(i) is not int or type(j) is not int or not 0 <= j <= i < len(basis) or (i, j) in declared:
                raise ValueError("invalid PSD entry")
            parsed = row_terms(entry, len(manifest.representatives))
            if any(b or a.denominator != 1 for a, b in parsed.values()):
                raise ValueError("PSD moment coefficients must be integers")
            declared[i, j] = {index: int(a) for index, (a, _) in parsed.items()}
        matrix = np.zeros((len(basis), len(basis))) if coordinates is not None else None
        for i, left in enumerate(basis):
            for j, right in enumerate(basis[:i+1]):
                independent = manifest.character_product(left, right)
                if independent != declared.get((i, j), {}):
                    raise ValueError(f"PSD moment map disagrees with its basis at {character}, {i}, {j}")
                if matrix is not None:
                    matrix[i, j] = matrix[j, i] = sum(w*coordinates[index] for index, w in independent.items())
        block = {"character": list(character), "size": len(basis), "entries_verified": len(declared)}
        if matrix is not None:
            block["min_eigenvalue"] = float(np.linalg.eigvalsh(matrix)[0])
        blocks.append(block)
    if set(canonical_bases) != characters:
        raise ValueError("missing PSD character block")
    report["psd_blocks"] = blocks
    if coordinates is not None:
        tolerance = 1e-8
        equality_error = max(equality_errors.values(), default=0.0)
        minimum_eigenvalue = min(block["min_eigenvalue"] for block in blocks)
        passed = equality_error <= tolerance and report["normalization_error"] <= tolerance and minimum_eigenvalue >= -tolerance
        report.update(passed=passed, numeric_feasibility_passed=passed, numeric_tolerance=tolerance,
                      relaxation_feasibility_verified=False)
    return report
