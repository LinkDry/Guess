"""Check -1 = sum(v.T Q v) + sum(weight*p**2) + sum(m_i*f_i).

The etf-sos-v1 JSON format uses an ordered list of real variables and sparse
polynomials given as lists of [exponent_vector, coefficient] pairs. Constraints
and multipliers are keyed by constraint ID. Each SOS block has a monomial
'basis' and symmetric 'Q', or a 'polynomial' and nonnegative 'weight'.

Exact scalars are JSON integers or rational strings. When 'radicand' is a
nonsquare integer D > 1, [a,b] represents a+b*sqrt(D) in its real embedding.
Numeric mode also accepts decimal coefficients. Its eigenvalues and tolerance
checks are diagnostic. Exact input is verified by field arithmetic throughout.

Exit codes: 0 for acceptance in the requested mode, 1 for failed checks, 2 for
invalid input. A matched external constraint file establishes file agreement.
--fixed1 additionally checks the independently derived ETF(4,9) branch equations.

For search-sdp-certificate-v1, --model supplies the equation file (default
model.json next to the certificate). The verifier uses exact integer factors
and rational root intervals to bound the residual on [-1,1]^n. Recognized
ETF branches also get a coefficient-by-coefficient constraint provenance check.
An anchored-exact-primal-v1 candidate uses --model (default anchored_model.json
next to the candidate) for moment order. The audit derives equations and PSD
blocks independently and checks any included reanchoring witnesses.
For anchored-moment-box-dual-v1 and degree-six primal candidates, the verifier
checks canonical equality witnesses and signed orbit products. Use
--audit-level3-model to audit a manifest without a certificate, optionally
with --primal-npz to report numerical feasibility diagnostics.
"""

import argparse
from decimal import Decimal
from hashlib import sha256
from math import isqrt
import json
from pathlib import Path
import re

import mpmath as mp
import sympy as sp


QQ = sp.QQ
RATIONAL = re.compile(r"[+-]?[0-9]+(?:/[1-9][0-9]*)?")


class RealField:
    def __init__(self, radicand=None, numeric=False):
        if radicand is not None and (
            type(radicand) is not int or radicand <= 1 or isqrt(radicand)**2 == radicand
        ):
            raise ValueError("radicand must be a nonsquare integer greater than one")
        self.radicand = radicand
        self.domain = QQ if radicand is None else QQ.algebraic_field(sp.sqrt(radicand))
        self.numeric = numeric
        self.zero = self.domain.zero
        self.one = self.domain.one

    def rational(self, value):
        if type(value) is int:
            return QQ(value)
        if isinstance(value, str) and RATIONAL.fullmatch(value):
            return QQ(value)
        if self.numeric and isinstance(value, (Decimal, str)):
            value = Decimal(value)
            if value.is_finite():
                return QQ(*value.as_integer_ratio())
        raise ValueError(f"invalid scalar {value!r}")

    def scalar(self, value):
        if isinstance(value, list) and len(value) == 2 and self.radicand is not None:
            a, b = map(self.rational, value)
            return self.domain.convert(a) + self.domain.convert(b) * self.domain.unit
        return self.domain.convert(self.rational(value))

    def components(self, value):
        if self.radicand is None:
            return value, QQ.zero
        terms = value.to_list()
        return (terms[-1] if terms else QQ.zero, terms[0] if len(terms) == 2 else QQ.zero)

    def sign(self, value):
        a, b = self.components(value)
        sa = int(a > 0) - int(a < 0)
        sb = int(b > 0) - int(b < 0)
        if not sa or not sb or sa == sb:
            return sa or sb
        delta = a*a - self.radicand*b*b
        return sa if delta > 0 else sb if delta < 0 else 0

    def approximate(self, value):
        a, b = self.components(value)
        a_numeric = mp.mpf(int(a.numerator)) / int(a.denominator)
        if not b:
            return a_numeric
        return a_numeric + mp.mpf(int(b.numerator)) / int(b.denominator) * mp.sqrt(self.radicand)

    def display(self, value):
        return str(self.domain.to_sympy(value))


def read_object(path):
    if path.suffix == ".gz":
        import gzip

        with gzip.open(path, "rb") as stream:
            data = stream.read()
    else:
        data = path.read_bytes()

    def strict_pairs(pairs):
        result = dict(pairs)
        if len(result) != len(pairs):
            raise ValueError("duplicate JSON object key")
        return result

    def reject_constant(constant):
        raise ValueError(f"nonfinite JSON scalar {constant}")

    obj = json.loads(data, parse_float=Decimal, parse_constant=reject_constant, object_pairs_hook=strict_pairs)
    if not isinstance(obj, dict):
        raise ValueError("expected JSON object")
    return obj, sha256(data).hexdigest()


def polynomials(document, field, size, key):
    raw = document[key]
    if not isinstance(raw, dict):
        raise ValueError(f"{key} must be an object mapping string IDs to polynomials")
    return {name: parse_poly(value, field, size) for name, value in raw.items()}


def monomial(value, size):
    if not isinstance(value, list) or len(value) != size or any(type(e) is not int or e < 0 for e in value):
        raise ValueError("invalid monomial exponent vector")
    return tuple(value)


def add_term(target, mon, coefficient, zero):
    total = target.get(mon, zero) + coefficient
    if total:
        target[mon] = total
    else:
        target.pop(mon, None)


def parse_poly(entries, field, size):
    if not isinstance(entries, list):
        raise ValueError("polynomial must be an array of [exponents, coefficient] terms")
    result = {}
    for term in entries:
        if not isinstance(term, list) or len(term) != 2:
            raise ValueError("invalid polynomial term")
        add_term(result, monomial(term[0], size), field.scalar(term[1]), field.zero)
    return result


def add_product(target, left, right, field, scale=None):
    if scale is None:
        scale = field.one
    for a, c in left.items():
        for b, d in right.items():
            power = tuple(x+y for x, y in zip(a, b))
            add_term(target, power, scale*c*d, field.zero)


def exact_psd(matrix, field):
    """Schur elimination by congruence with an explicit zero-pivot check."""
    if any(matrix[i][j] != matrix[j][i] for i in range(len(matrix)) for j in range(i)):
        return {"holds": False, "reason": "asymmetric"}
    schur = [row.copy() for row in matrix]
    rank = 0
    for i, row in enumerate(schur):
        pivot = row[i]
        sign = field.sign(pivot)
        if sign < 0:
            return {"holds": False, "reason": "negative Schur pivot", "pivot_index": i}
        if sign == 0:
            if any(row[j] for j in range(i+1, len(schur))):
                return {"holds": False, "reason": "nonzero Schur row at zero pivot", "pivot_index": i}
            continue
        rank += 1
        for j in range(i+1, len(schur)):
            if not row[j]:
                continue
            ratio = row[j] / pivot
            for k in range(j, len(schur)):
                schur[j][k] -= ratio * row[k]
    return {"holds": True, "rank": rank}


def numeric_psd(matrix, field, tolerance, digits):
    dimension = len(matrix)
    values = [[field.approximate(c) for c in row] for row in matrix]
    symmetry_error = max(abs(values[i][j] - values[j][i]) for i in range(dimension) for j in range(dimension))
    sym = mp.matrix([[(values[i][j] + values[j][i])/2 for j in range(dimension)] for i in range(dimension)])
    minimum = mp.eigsy(sym, eigvals_only=True)[0]
    # Gershgorin gives a rigorous PSD shift for the exact symmetrized input.
    # Shifting by t*I changes each basis square coefficient by t.
    half = field.scalar("1/2")
    shift = field.zero
    for i, row in enumerate(matrix):
        bound = -row[i]
        for j, entry in enumerate(row):
            if i != j:
                entry = half*(entry + matrix[j][i])
                bound += entry if field.sign(entry) >= 0 else -entry
        if field.sign(bound - shift) > 0:
            shift = bound
    return {"holds": bool(symmetry_error <= tolerance and minimum >= -tolerance),
            "min_eigenvalue_estimate": mp.nstr(minimum, digits),
            "max_symmetry_error_estimate": mp.nstr(symmetry_error, digits),
            "certified_gershgorin_psd_shift": field.display(shift)}, shift


def canonical_fixed1(variables, field):
    """Generate the twenty necessary equations from the real pair lifting.

    The four lifted pairs are ((a,x_j),(0,y_j)), with a=sqrt(2)*mu. Their
    wedge vectors have Gram (w_j.T w_k + a^2*y_j.T y_k). The equalities of
    opposite complex inner-product moduli force off-diagonals to zero.
    The unit-pair diagonal wedge norms are 1-mu^2. Tightness on the three
    orthogonal coordinates gives XX.T+YY.T=9/4 I. Since W has three rows
    and four columns, its kernel contains a unit h.
    """
    names = [f"{c}{i}{j}" for c in "xy" for i in range(3) for j in range(4)] + [f"h{j}" for j in range(4)]
    if set(variables) != set(names):
        raise ValueError("fixed1 requires variables x00..x23, y00..y23, h0..h3")
    symbols = tuple(sp.Symbol(name) for name in variables)
    by_name = dict(zip(variables, symbols))
    X, Y = [sp.Matrix(3, 4, lambda i, j: by_name[f"{c}{i}{j}"]) for c in "xy"]
    h = sp.Matrix([by_name[f"h{j}"] for j in range(4)])
    W = sp.Matrix.hstack(*(X[:, j].cross(Y[:, j]) for j in range(4)))
    mu_squared = (sp.Rational(9, 4) - 1) / 8
    E = W.T*W + 2*mu_squared*(Y.T*Y) - (1-mu_squared)*sp.eye(4)
    T = X*X.T + Y*Y.T - sp.Rational(9, 4)*sp.eye(3)
    equations = {"h_unit": h.dot(h)-1}
    for name, matrix in (("bivector", E), ("tight", T)):
        for i in range(matrix.rows):
            for j in range(i, matrix.cols):
                equations[f"{name}_{i}{j}"] = matrix[i, j]
    equations.update({f"kernel_{i}": exp for i, exp in enumerate(W*h)})
    return {name: {mon: field.domain.from_sympy(c) for mon, c in sp.Poly(exp, *symbols).terms()}
            for name, exp in equations.items()}


def verify(document, trusted=None, fixed1=False, digits=50, tolerance="1e-9"):
    if document.get("format") != "etf-sos-v1" or document.get("mode") not in ("exact", "numeric"):
        raise ValueError('expected etf-sos-v1 with mode "exact" or "numeric"')
    mode = document["mode"]
    numeric = mode == "numeric"
    variables = document["variables"]
    if (not isinstance(variables, list) or not variables or
        any(not isinstance(v, str) or not v for v in variables) or len(set(variables)) != len(variables)):
        raise ValueError("variables must be a nonempty list of distinct names")
    size = len(variables)
    field = RealField(document.get("radicand"), numeric)
    constraints = polynomials(document, field, size, "constraints")
    multipliers = polynomials(document, field, size, "multipliers")
    if not constraints.keys() >= multipliers.keys():
        raise ValueError("a multiplier references an unknown constraint")
    provenance = {"status": "unchecked"}
    if trusted is not None:
        if trusted.get("format") != "etf-sos-constraints-v1":
            raise ValueError("invalid trusted constraint format")
        if trusted.get("radicand") != document.get("radicand") or trusted.get("variables") != variables:
            matches = False
        else:
            expected = polynomials(trusted, RealField(trusted.get("radicand")), size, "constraints")
            matches = constraints == expected
        provenance = {"status": "matched" if matches else "mismatch", "source": "trusted_file"}
    if fixed1:
        matches = constraints == canonical_fixed1(variables, field)
        provenance = {"status": "matched" if matches and provenance["status"] != "mismatch" else "mismatch",
                      "source": "independent_fixed1_equations", "scope": "ETF(4,9), antiunitary cycle type 2^4 1"}

    difference = {(0,)*size: field.one}
    for name, multiplier in multipliers.items():
        add_product(difference, multiplier, constraints[name], field)
    blocks = document["sos"]
    if not isinstance(blocks, list):
        raise ValueError("sos must be an array of blocks")
    block_results = []
    correction = {}
    with mp.workdps(digits):
        threshold = mp.mpf(tolerance)
        if not mp.isfinite(threshold) or threshold <= 0:
            raise ValueError("tolerance must be positive and finite")
        for block in blocks:
            if set(block) == {"weight", "polynomial"}:
                weight = field.scalar(block["weight"])
                poly = parse_poly(block["polynomial"], field, size)
                add_product(difference, poly, poly, field, weight)
                if numeric:
                    details, shift = numeric_psd([[weight]], field, threshold, digits)
                    add_product(correction, poly, poly, field, shift)
                else:
                    details = exact_psd([[weight]], field)
            elif set(block) == {"basis", "Q"}:
                basis = [monomial(m, size) for m in block["basis"]]
                rows = block["Q"]
                if not basis or len(set(basis)) != len(basis) or len(rows) != len(basis) or any(
                    not isinstance(row, list) or len(row) != len(basis) for row in rows
                ):
                    raise ValueError("Q must be nonempty and square with a matching distinct monomial basis")
                Q = [[field.scalar(c) for c in row] for row in rows]
                for i, exponent in enumerate(basis):
                    for j, other in enumerate(basis):
                        power = tuple(x+y for x, y in zip(exponent, other))
                        add_term(difference, power, Q[i][j], field.zero)
                if numeric:
                    details, shift = numeric_psd(Q, field, threshold, digits)
                    for exponent in basis:
                        add_term(correction, tuple(2*e for e in exponent), shift, field.zero)
                else:
                    details = exact_psd(Q, field)
            else:
                raise ValueError("invalid SOS block fields")
            block_results.append(details)

        identity = {"holds": not difference, "nonzero_residual_terms": len(difference)}
        if difference:
            mon = min(difference)
            identity["first_residual_term"] = [list(mon), field.display(difference[mon])]
        psd = {"holds": all(block["holds"] for block in block_results), "blocks": block_results}
        provenance_ok = provenance["status"] != "mismatch"
        refuted = not numeric and identity["holds"] and psd["holds"]
        report = {"mode": mode, "identity": identity, "psd": psd, "constraint_provenance": provenance,
                  "polynomial_system_refuted": refuted, "etf_nonexistence_verified": bool(refuted and fixed1 and provenance_ok)}
        if numeric:
            norm = max((abs(field.approximate(c)) for c in difference.values()), default=mp.mpf(0))
            correction_l1_bound = field.zero
            for c in correction.values():
                correction_l1_bound += c if field.sign(c) >= 0 else -c
            report.update(numerically_consistent=bool(norm <= threshold and psd["holds"]),
                          working_digits=digits, numerical_tolerance=tolerance,
                          max_identity_coefficient_defect_estimate=mp.nstr(norm, digits),
                          psd_repair_identity_coefficient_l1_bound=field.display(correction_l1_bound))
        report["passed"] = bool(provenance_ok and (report["numerically_consistent"] if numeric else refuted))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("certificate", type=Path)
    parser.add_argument("--trusted-constraints", type=Path)
    parser.add_argument("--model", type=Path, help="equations and moment order for the certificate")
    parser.add_argument("--negative-direction", type=Path, help="verify a PSD counterexample for a primal candidate")
    parser.add_argument("--audit-face", action="store_true", help="audit a structural facial reduction manifest")
    parser.add_argument("--audit-level3-model", action="store_true", help="audit the manifest supplied as the first argument")
    parser.add_argument("--primal-npz", type=Path, help="also diagnose the numeric x vector in a solver NPZ")
    parser.add_argument("--fixed1", action="store_true", help="verify necessary equations for the single-fixed-line branch")
    parser.add_argument("--digits", type=int, default=50)
    parser.add_argument("--tolerance", default="1e-9")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.digits < 20:
            raise ValueError("working precision must be at least 20 digits")
        document, digest = read_object(args.certificate)
        if args.negative_direction:
            from verify_sdp_face import verify_negative_direction

            model, model_hash = read_object(args.model or args.certificate.with_name("solver_face_model.json"))
            witness, witness_hash = read_object(args.negative_direction)
            report = verify_negative_direction(document, model, witness)
            report.update(face_model_sha256=model_hash, integer_test_vector_sha256=witness_hash)
        elif args.audit_face:
            from verify_sdp_face import check_exact_face

            source, source_hash = read_object(args.model) if args.model else (None, None)
            if source is not None and document.get("source_sha256") != source_hash:
                raise ValueError("facial source manifest SHA-256 disagrees")
            report = check_exact_face(document, source)
            if source_hash is not None:
                report["source_model_sha256"] = source_hash
        elif args.audit_level3_model:
            from verify_sdp_level3 import audit_model_numerically

            coordinates = None
            if args.primal_npz:
                import numpy as np
                from io import BytesIO

                primal_bytes = args.primal_npz.read_bytes()
                with np.load(BytesIO(primal_bytes), allow_pickle=False) as archive:
                    coordinates = archive["x"]
            report = audit_model_numerically(document, coordinates)
            if coordinates is not None:
                report["primal_sha256"] = sha256(primal_bytes).hexdigest()
        elif document.get("schema") == "search-sdp-certificate-v1":
            from verify_sdp_factor_cert import verify_factor_certificate

            model_path = args.model if args.model is not None else args.certificate.with_name("model.json")
            model, model_hash = read_object(model_path)
            report = verify_factor_certificate(document, model)
            report["model_sha256"] = model_hash
        elif document.get("schema") in {"anchored-exact-primal-v1", "anchored-moment-box-dual-v1"}:
            from verify_sdp_anchored import audit_anchored_primal
            from verify_sdp_level3 import verify_degree6_primal, verify_moment_dual

            model_path = args.model or args.certificate.with_name("solver_model.json")
            if args.model is None and not model_path.exists():
                model_path = args.certificate.with_name("solver_face_model.json")
            if args.model is None and not model_path.exists():
                model_path = args.certificate.with_name("anchored_model.json")
            model, model_hash = read_object(model_path)
            if document["schema"] == "anchored-moment-box-dual-v1":
                report = verify_moment_dual(document, model)
            elif model.get("maximum_monomial_degree") == 6:
                report = verify_degree6_primal(document, model, exact_psd, RealField(10))
            else:
                report = audit_anchored_primal(document, model, exact_psd, RealField(10))
            report["model_sha256"] = model_hash
        else:
            trusted, trusted_hash = read_object(args.trusted_constraints) if args.trusted_constraints else (None, None)
            report = verify(document, trusted, args.fixed1, args.digits, args.tolerance)
            if trusted_hash is not None:
                report["trusted_constraints_sha256"] = trusted_hash
        report["certificate_sha256"] = digest
        code = 0 if report["passed"] else 1
    except (ArithmeticError, KeyError, OSError, TypeError, ValueError) as exc:
        report = {"passed": False, "error": str(exc)}
        code = 2
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
