"""Independent diagnostics and exact postsolve checks for the Round 7 face LMI.

The CSC pencil and solution arrays are floating-point proposals. Every exact
branch uses the checked Round 6 manifest and reconstructs the relevant map over
Q(sqrt(10)). A bounded dual certificate refutes feasible moments in the stated
box, which contains all moments obtained from an anchored ETF.
"""

import argparse
from fractions import Fraction
import gzip
import hashlib
import json
from math import lcm, sqrt
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, csr_matrix, vstack

from verify_sdp_cert import RealField, exact_psd
from verify_sdp_factor_cert import rational
from verify_sdp_face import check_exact_face
from verify_sdp_level3 import BASIS, MomentManifest, beta_interval, pair, row_terms


BETA = sqrt(10) / 5
BETA_SQUARED = Fraction(2, 5)


def triangle(size, packing):
    if packing == "lower":
        return ((i, j) for j in range(size) for i in range(j, size))
    if packing == "upper":
        return ((i, j) for j in range(size) for i in range(j + 1))
    raise ValueError("unknown PSD packing")


def unpack_svec(vector, size, packing):
    if np.shape(vector) != (size * (size + 1) // 2,):
        raise ValueError("wrong packed block length")
    matrix = np.zeros((size, size))
    for (i, j), value in zip(triangle(size, packing), vector, strict=True):
        matrix[i, j] = matrix[j, i] = value if i == j else value / sqrt(2)
    return matrix


def unpack_blocks(vector, sizes, packing):
    offset = 0
    blocks = []
    for size in sizes:
        end = offset + size * (size + 1) // 2
        blocks.append(unpack_svec(vector[offset:end], size, packing))
        offset = end
    if len(vector) != offset:
        raise ValueError("wrong packed cone length")
    return blocks


def finite_vector(solution, names, size):
    for name in names:
        if name not in solution:
            continue
        value = np.asarray(solution[name], dtype=float)
        if value.shape != (size,):
            raise ValueError(f"wrong shape for {name}")
        return value if np.isfinite(value).all() else None
    return None


def maxabs(vector):
    return float(np.max(np.abs(vector), initial=0))


def min_eigenvalues(vector, sizes, packing):
    return [float(np.linalg.eigvalsh(block)[0]) for block in unpack_blocks(vector, sizes, packing)]


def conic_diagnostics(matrix, b, q, sizes, solution, packing):
    m, n = matrix.shape
    x = finite_vector(solution, ("x", "t"), n)
    z = finite_vector(solution, ("z", "y"), m)
    slack = finite_vector(solution, ("s",), m)
    result = {"mode": "numeric_reduced_lmi", "exact_proof": False, "packing": packing,
              "shape": [m, n], "nnz": matrix.nnz, "primal": None, "dual": None}
    if x is not None:
        affine = b - matrix @ x
        result["primal"] = {"objective": float(q @ x), "min_eigenvalues": min_eigenvalues(affine, sizes, packing)}
        if slack is not None:
            result["primal"].update(slack_residual_inf=maxabs(slack-affine),
                                    slack_min_eigenvalues=min_eigenvalues(slack, sizes, packing))
        if z is not None:
            result["primal"]["affine_complementarity"] = float(affine @ z)
    if z is not None:
        stationarity = matrix.T @ z
        pairing = float(b @ z)
        result["dual"] = {"min_eigenvalues": min_eigenvalues(z, sizes, packing),
                          "objective_stationarity_inf": maxabs(stationarity + q),
                          "ray_stationarity_inf": maxabs(stationarity), "ray_b_dot_z": pairing}
        if pairing < 0:
            result["dual"]["ray_stationarity_inf_at_b_dot_z_minus_one"] = maxabs(stationarity)/(-pairing)
    return result


def load_conic(path):
    with np.load(path, allow_pickle=False) as raw:
        shape = tuple(raw["A_shape"].tolist())
        sizes = raw["block_sizes"].tolist()
        if (len(shape) != 2 or any(type(i) is not int or i < 0 for i in shape)
                or not isinstance(sizes, list) or not sizes
                or any(type(i) is not int or i <= 0 for i in sizes)
                or shape[0] != sum(i * (i + 1) // 2 for i in sizes)):
            raise ValueError("invalid reduced LMI dimensions")
        matrix = csc_matrix((raw["A_data"], raw["A_indices"], raw["A_indptr"]), shape=shape)
        matrix.check_format(full_check=True)
        b, q = raw["b"], raw["q"]
        kept = raw["retained_chart_columns"] if "retained_chart_columns" in raw else np.arange(shape[1])
        full_dimension = raw["full_chart_dimension"].item() if "full_chart_dimension" in raw else shape[1]
        congruence = raw["psd_congruence"] if "psd_congruence" in raw else None
        if (type(full_dimension) is not int or full_dimension < shape[1]
                or kept.shape != (shape[1],) or kept.dtype.kind not in "iu"
                or len(set(kept)) != len(kept) or np.any(kept < 0) or np.any(kept >= full_dimension)):
            raise ValueError("invalid retained chart columns")
        if congruence is not None and (congruence.shape != (shape[0],) or not np.isrealobj(congruence)
                                       or not np.isfinite(congruence).all() or np.any(congruence <= 0)):
            raise ValueError("invalid PSD congruence weights")
        if (b.shape != (shape[0],) or q.shape != (shape[1],)
                or any(not np.isrealobj(v) or not np.isfinite(v).all() for v in (matrix.data, b, q))):
            raise ValueError("invalid reduced LMI coefficients")
    return matrix, b, q, sizes, kept.tolist(), full_dimension, congruence


def read_json(path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8-sig") as source:
            return json.load(source)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rational_scaled_row(row, parity):
    """Strip a common beta, if present, after substituting m_i=beta**p_i u_i."""
    terms = {}
    tags = set()
    for i, (a, b) in row.items():
        p = parity[i]
        if a:
            tags.add(p)
            terms[i] = Fraction(a)
        if b:
            tags.add(1-p)
            terms[i] = Fraction(b)*BETA_SQUARED if p else Fraction(b)
    if len(tags) != 1:
        raise ValueError("selected equality has zero or mixed parity")
    return terms


class AffineChart:
    """Exact affine chart in rationally scaled moment coordinates."""

    def __init__(self, document, parity):
        self.parity = tuple(document["parity"])
        if (self.parity != tuple(parity) or any(type(p) is not int or p not in (0, 1) for p in self.parity)
                or not self.parity or self.parity[0] != 0):
            raise ValueError("chart parity disagrees with moment order")
        self.pivot = tuple(document["pivot_columns"])
        self.free = tuple(document["free_columns"])
        self.selected = tuple(document["selected_rows"])
        size = len(self.parity)
        if (any(type(i) is not int for i in (*self.pivot, *self.free))
                or sorted(self.pivot + self.free) != list(range(size))):
            raise ValueError("pivot and free columns must partition the moment order")
        if (any(type(i) is not int or i < 0 for i in self.selected)
                or len(set(self.selected)) != len(self.selected)
                or len(self.selected) != len(self.pivot) or 0 not in self.selected):
            raise ValueError("invalid selected equality rows")
        self.width = len(self.free) + 1
        self.rows = []
        for denominator, terms in document["rows"]:
            if type(denominator) is not int or denominator <= 0:
                raise ValueError("invalid chart row denominator")
            row = {}
            for slot, numerator in terms:
                if (type(slot) is not int or not 0 <= slot < self.width or slot in row
                        or type(numerator) is not int or not numerator):
                    raise ValueError("invalid chart row term")
                row[slot] = numerator
            self.rows.append((denominator, row))
        if len(self.rows) != size:
            raise ValueError("wrong chart moment count")
        for slot, i in enumerate(self.free, 1):
            denominator, row = self.rows[i]
            if row != {slot: denominator}:
                raise ValueError("chart changes a free coordinate")
        self.common_denom = lcm(*(denom for denom, _ in self.rows))
        self.integers = [{k: n * (self.common_denom // denom) for k, n in row.items()}
                         for denom, row in self.rows]

    def verify_coverage(self, equalities):
        """Check that every solution of the selected equations lies in the chart."""
        from flint import nmod_mat

        rows = []
        for index in self.selected:
            if not 0 <= index < len(equalities):
                raise ValueError("selected equality index outside the manifest")
            equation = equalities[index]
            if index == 0 and equation != {0: (1, 0)}:
                raise ValueError("unexpected normalization row")
            row = rational_scaled_row(equation, self.parity)
            denominator = lcm(*(v.denominator for v in row.values()))
            rows.append({i: int(coef * denominator) for i, coef in row.items()})
            identity = [0] * self.width
            identity[0] = -self.common_denom * denominator if index == 0 else 0
            for column, coefficient in rows[-1].items():
                for slot, value in self.integers[column].items():
                    identity[slot] += coefficient * value
            if any(identity):
                raise ValueError(f"chart substitution fails for selected equality {index}")
        rank = len(self.pivot)
        # A full rank reduction certifies full rank over Q. A deficient
        # reduction only means this prime does not provide a certificate.
        for prime in (65537, 1000000007, 1000000009):
            entries = [row.get(column, 0) % prime for row in rows for column in self.pivot]
            if nmod_mat(rank, rank, entries, prime).rank() == rank:
                return rank
        raise ValueError("selected pivot minor has no full rank modular certificate")

    def lift(self, free):
        if len(free) + 1 != self.width:
            raise ValueError("wrong free coordinate count")
        slots = [Fraction(1), *map(Fraction, free)]
        denominator = lcm(*(v.denominator for v in slots))
        slots = [int(v * denominator) for v in slots]
        result = []
        for parity, (scale, row) in zip(self.parity, self.rows, strict=True):
            value = Fraction(sum(numerator * slots[slot] for slot, numerator in row.items()),
                             scale * denominator)
            result.append((0, value) if parity else (value, 0))
        return result

    def linear_form(self, coefficients):
        """Evaluate a rational linear combination of physical moments on the chart."""
        nonzero = [(i, Fraction(value)) for i, value in enumerate(coefficients) if value]
        if len(coefficients) != len(self.parity):
            raise ValueError("wrong moment coefficient count")
        denominator = lcm(*(value.denominator for _, value in nonzero))
        parts = [[0] * self.width for _ in range(2)]
        for i, value in nonzero:
            coefficient = int(value * denominator)
            for slot, numerator in self.integers[i].items():
                parts[self.parity[i]][slot] += coefficient * numerator
        denominator *= self.common_denom
        return [(Fraction(a, denominator), Fraction(b, denominator)) for a, b in zip(*parts, strict=True)]


def moment_bounds(words):
    """A signed orbit average stays in the unit box, with squares nonnegative."""
    return [(1, 1) if not word else (0, 1) if all(word.count(i) % 2 == 0 for i in set(word))
            else (-1, 1) for word in words]


def bounded_upper(chart, objective, bounds, weights=None):
    """Bound an exact chart affine form over the moment box with optional LP multipliers.

    Each signed multiplier w subtracts w*m_i from the residual and adds
    max(w*lo_i,w*hi_i). Independently check the remaining affine form using
    just the free-coordinate box. The bound holds for any rational weights.
    """
    if len(objective) != chart.width or len(bounds) != len(chart.parity):
        raise ValueError("objective or moment bounds do not match chart")
    coefficients = [0] * len(bounds)
    offset = 0
    for i, weight in (weights or {}).items():
        if type(i) is not int or not 0 <= i < len(bounds):
            raise ValueError("invalid bound multiplier index")
        weight = Fraction(weight)
        coefficients[i] = weight
        offset += max(weight * endpoint for endpoint in bounds[i])
    subtraction = chart.linear_form(coefficients)
    residual = [(a-c, b-d) for (a, b), (c, d) in zip(objective, subtraction, strict=True)]
    upper = offset + beta_interval(residual[0])[1]
    for (a, b), i in zip(residual[1:], chart.free, strict=True):
        # 1/beta=(5/2)*beta, so scalar t_i bounds need no floating point.
        if chart.parity[i]:
            a, b = b, a / BETA_SQUARED
        upper += max(beta_interval((end*a, end*b))[1] for end in bounds[i])
    return upper


def propose_box_weights(chart, objective, bounds, denominator):
    """Use a floating LP for suggestions, then recover rational coefficients."""
    columns, values, pointers, constants = [], [], [0], []
    for parity, (scale, row) in zip(chart.parity, chart.rows, strict=True):
        factor = BETA if parity else 1.0
        constants.append(float(Fraction(row.get(0, 0), scale)) * factor)
        for slot, value in row.items():
            if slot:
                columns.append(slot - 1)
                values.append(float(Fraction(value, scale)) * factor)
        pointers.append(len(columns))
    if chart.width == 1:
        return {}
    affine = csr_matrix((values, columns, pointers), shape=(len(bounds), chart.width-1))
    constants = np.array(constants)
    lo, hi = np.asarray(bounds, dtype=float).T
    cost = np.array([-(float(a) + float(b)*BETA) for a, b in objective[1:]])
    result = linprog(cost, A_ub=vstack((affine, -affine)), b_ub=np.r_[hi-constants, constants-lo],
                     bounds=(None, None), method="highs")
    if not result.success:
        return {}
    upper, lower = np.split(result.ineqlin.marginals, 2)
    weights = [Fraction(round(float(v)*denominator), denominator) for v in lower-upper]
    return {i: w for i, w in enumerate(weights) if w}


def assemble_affine_objective(entries, factors, count):
    """Compute coefficients of sum_b trace((G_b G_b^T) M_b(m)) in m."""
    coefficients = [0] * count
    for block, factor in zip(entries, factors, strict=True):
        gram = np.asarray(factor, dtype=object) @ np.asarray(factor, dtype=object).T
        for i, j, row in block:
            value = int(gram[i, j]) * (1 if i == j else 2)
            if value:
                for k, coefficient in row.items():
                    coefficients[k] += value * coefficient
    return coefficients


def rounded_factor(matrix, denominator):
    """Round a Gram factor from the positive part of a numeric dual block."""
    values, vectors = np.linalg.eigh(matrix)
    positive = values > 0
    factor = vectors[:, positive] * np.sqrt(values[positive])
    return [[round(float(value)*denominator) for value in row] for row in factor]


def exact_dual_report(chart, entries, bounds, factors, denominator, tighten=False, weights=None):
    if type(denominator) is not int or denominator <= 0:
        raise ValueError("invalid Gram factor denominator")
    for block, factor in zip(entries, factors, strict=True):
        size = 1 + max(max(i, j) for i, j, _ in block)
        if (len(factor) != size or any(len(row) != len(factor[0]) for row in factor)
                or any(type(value) is not int for row in factor for value in row)):
            raise ValueError("invalid integer Gram factor")
    coefficients = assemble_affine_objective(entries, factors, len(chart.parity))
    # Keep the denominator outside the objective to avoid dividing every entry
    # twice. Normalize by the largest coefficient before floating LP work.
    norm = max(map(abs, coefficients), default=0) or 1
    objective = chart.linear_form([Fraction(c, norm) for c in coefficients])
    box_upper = bounded_upper(chart, objective, bounds)
    if tighten and box_upper >= 0 and weights is None:
        weights = propose_box_weights(chart, objective, bounds, denominator)
    tightened = bounded_upper(chart, objective, bounds, weights)
    scale = Fraction(norm, denominator**2)
    result = {"passed": min(box_upper, tightened) < 0, "gram_factor_denominator": denominator,
              "box_upper": str(scale * box_upper), "box_upper_with_lifted_bounds": str(scale * tightened),
              "box_upper_approx": float(scale * min(box_upper, tightened)),
              "lifted_bound_multipliers": len(weights or {}),
              "upper_at_chart_origin_approx": float(scale * beta_interval(objective[0])[1]),
              "chart_linear_coefficients_max_approx": max(
                  (abs(float(a)+float(b)*BETA)*float(scale) for a, b in objective[1:]), default=0.0)}
    return result, weights or {}


def exact_moment_entries(model):
    """Read integer entries after the independent face audit has checked them."""
    result = []
    for block in model["psd_blocks"]:
        entries = []
        for i, j, row in block["entries"]:
            parsed = {}
            for index, value in row.items():
                if type(value) is list and len(value) == 2 and value[1] == 0:
                    value = value[0]
                if type(value) is not int:
                    raise ValueError("expected integer unscaled moment entry")
                parsed[int(index)] = value
            entries.append((i, j, parsed))
        result.append(entries)
    return result


def certify_matrix(real, radical, scale):
    """Check (real + beta*radical)/scale with exact quadratic forms."""
    size = len(real)
    if scale <= 0 or not size or len(radical) != size or any(len(row) != size for row in (*real, *radical)):
        raise ValueError("invalid exact PSD matrix dimensions or scale")
    if any(matrix[i][j] != matrix[j][i] for matrix in (real, radical) for i in range(size) for j in range(i)):
        return {"holds": False, "reason": "asymmetric", "size": size}
    approx = (np.array(real, dtype=object) / scale).astype(float)
    approx += BETA * (np.array(radical, dtype=object) / scale).astype(float)
    eigenvalues, vectors = np.linalg.eigh(approx)
    report = {"min_eigenvalue_approx": float(eigenvalues[0]), "size": len(real)}
    if eigenvalues[0] < 0:
        vector = [round(float(v)*10**10) for v in vectors[:, 0]]
        a, b = [sum(v_i * v_j * matrix[i][j] for i, v_i in enumerate(vector)
                    for j, v_j in enumerate(vector)) for matrix in (real, radical)]
        upper = beta_interval((a, b))[1]
        if upper < 0:
            return {**report, "holds": False, "method": "exact_negative_quadratic_form",
                    "integer_test_vector": vector, "quadratic_form_components": [str(Fraction(a, scale)),
                                                                                 str(Fraction(b, scale))],
                    "quadratic_form_upper": str(upper/scale),
                    "normalized_quadratic_form_upper_approx": float(upper/(scale*sum(v*v for v in vector)))}
    try:
        # If W is a nonsingular integer triangular approximation of the
        # inverse Cholesky factor, diagonal dominance of W M W^T is sufficient.
        from flint import fmpz_mat
        from scipy.linalg import solve_triangular

        chol = np.linalg.cholesky(approx)
        inverse = solve_triangular(chol, np.eye(len(real)), lower=True)
        max_entry = maxabs(inverse)
        if np.isfinite(max_entry) and max_entry:
            integer = [[round(float(v/max_entry)*10**10) if j <= i else 0 for j, v in enumerate(row)]
                       for i, row in enumerate(inverse)]
            if all(row[i] for i, row in enumerate(integer)):
                w = fmpz_mat(integer)
                a, b = [w * fmpz_mat(matrix) * w.transpose() for matrix in (real, radical)]
                for i in range(len(real)):
                    lower = beta_interval((int(a[i, i]), int(b[i, i])))[0]
                    radius = sum(max(map(abs, beta_interval((int(a[i, j]), int(b[i, j])))))
                                 for j in range(len(real)) if i != j)
                    if lower <= radius:
                        break
                else:
                    return {**report, "holds": True, "rank": len(real), "method": "exact_diagonal_dominance"}
    except np.linalg.LinAlgError:
        pass
    field = RealField(10)
    domain = field.domain
    beta = field.scalar([0, "1/5"])
    matrix = [[domain.convert(a)+domain.convert(b)*beta for a, b in zip(a_row, b_row, strict=True)]
              for a_row, b_row in zip(real, radical, strict=True)]
    return {**report, **exact_psd(matrix, field), "method": "exact_ldl"}


def read_primal_moments(candidate, count):
    if candidate.get("schema") != "anchored-exact-primal-v1" or candidate.get("coefficient_basis") != BASIS:
        raise ValueError("unexpected primal schema or coefficient basis")
    if len(candidate["moments"]) != count:
        raise ValueError("primal moment count disagrees with face manifest")
    return list(map(pair, candidate["moments"]))


def scaled_moments(moments):
    scale = lcm(*(Fraction(part).denominator for value in moments for part in value))
    return [(int(a*scale), int(b*scale)) for a, b in moments], scale


def failed_equalities(model, scaled, scale):
    failures = [0] if scaled[0] != (scale, 0) else []
    for i, row in enumerate(model["equalities"], 1):
        a = b = 0
        for index, coefficient in row.items():
            left_real, left_radical = (coefficient, 0) if type(coefficient) is int else coefficient
            if type(left_real) is not int or type(left_radical) is not int:
                left_real, left_radical = rational(left_real), rational(left_radical)
            right_real, right_radical = scaled[int(index)]
            a += 5*left_real*right_real + 2*left_radical*right_radical
            b += left_real*right_radical + left_radical*right_real
        if a or b:
            failures.append(i)
    return failures


def exact_primal_report(model, entries, moments):
    scaled, scale = scaled_moments(moments)
    failures = failed_equalities(model, scaled, scale)
    blocks = []
    for block, metadata in zip(entries, model["psd_blocks"], strict=True):
        size = len(metadata["basis"])
        a = [[0]*size for _ in range(size)]
        b = [[0]*size for _ in range(size)]
        for i, j, coefficients in block:
            for component, matrix in enumerate((a, b)):
                value = sum(scaled[k][component]*coefficient for k, coefficient in coefficients.items())
                matrix[i][j] = matrix[j][i] = value
        blocks.append({"character": metadata["character"], **certify_matrix(a, b, scale)})
    passed = not failures and all(block["holds"] for block in blocks)
    return {"passed": passed, "relaxation_feasibility_verified": passed,
            "failed_equality_count": len(failures), "first_failed_equalities": failures[:20], "blocks": blocks}


def lift_numeric_free(x, kept, full_dimension, denominator):
    free = [Fraction(0)] * full_dimension
    for index, value in zip(kept, x, strict=True):
        free[index] = Fraction(round(float(value)*denominator), denominator)
    return free


def compare_implemented_pencil(chart, entries, sizes, matrix, b, packing, kept):
    """Report the largest coefficient discrepancy against the checked chart."""
    slots = {column+1: j+1 for j, column in enumerate(kept)}
    slots[0] = 0
    chart_rows = np.zeros((len(chart.parity), len(kept)+1))
    for i, (scale, terms) in enumerate(chart.rows):
        for slot, numerator in terms.items():
            if slot in slots:
                chart_rows[i, slots[slot]] = float(Fraction(numerator, scale))*(BETA if chart.parity[i] else 1)
    matrix = matrix.tocsr()
    errors = {"A_inf": 0.0, "b_inf": 0.0}
    offset = 0
    for block, size in zip(entries, sizes, strict=True):
        entry_map = {(i, j): row for i, j, row in block}
        rows = np.zeros((size * (size + 1) // 2, len(kept)+1))
        for k, (i, j) in enumerate(triangle(size, packing)):
            for column, coefficient in entry_map.get((max(i, j), min(i, j)), {}).items():
                rows[k] += coefficient * chart_rows[column]
            if i != j:
                rows[k] *= sqrt(2)
        end = offset + len(rows)
        errors["A_inf"] = max(errors["A_inf"], maxabs(rows[:, 1:] + matrix[offset:end, :].toarray()))
        errors["b_inf"] = max(errors["b_inf"], maxabs(rows[:, 0]-b[offset:end]))
        offset = end
    return errors


def selected_equalities(chart, model):
    """Extract the independently audited rows while preserving global numbering."""
    equalities = [None] * (len(model["equalities"])+1)
    equalities[0] = {0: (1, 0)}
    for i in chart.selected:
        if i >= len(equalities):
            raise ValueError("selected equality index outside the manifest")
        if i:
            equalities[i] = row_terms(model["equalities"][i-1], len(chart.parity))
    return equalities


def model_sha256(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conic", type=Path, help="numeric reduced LMI CSC archive")
    parser.add_argument("--solution", type=Path, help="solver NPZ containing x and/or z (or y)")
    parser.add_argument("--packing", choices=("lower", "upper"), default="lower")
    parser.add_argument("--model", type=Path, help="JSON or gzipped Round 6 face manifest")
    parser.add_argument("--chart", type=Path, help="JSON or gzipped exact affine chart")
    parser.add_argument("--exact-primal", action="store_true")
    parser.add_argument("--primal-certificate", type=Path, help="check an exact original-moment JSON certificate")
    parser.add_argument("--exact-dual", action="store_true")
    parser.add_argument("--tighten-box", action="store_true", help="suggest exact weight corrections from a box LP")
    parser.add_argument("--compare-pencil", action="store_true")
    parser.add_argument("--audit-chart", action="store_true", help="prove exact chart coverage of the selected rows")
    parser.add_argument("--scaled-solution", action="store_true",
                        help="solution z and s are in the congruence-scaled cone, rather than the archive cone")
    parser.add_argument("--denominator", type=int, default=10**10, help="rounding denominator for exact candidates")
    parser.add_argument("--dual-certificate", type=Path, help="replay archived exact Gram factors and box weights")
    parser.add_argument("--export-dual", type=Path, help="save the exact factors and weights for independent replay")
    args = parser.parse_args()
    needs_chart = args.exact_primal or args.exact_dual or args.dual_certificate or args.compare_pencil or args.audit_chart
    wants_exact = bool(args.exact_primal or args.exact_dual or args.dual_certificate or args.primal_certificate)
    if needs_chart and (args.chart is None or args.model is None or args.conic is None):
        parser.error("chart checks require --chart, --model, and --conic")
    if args.primal_certificate and (args.model is None or args.exact_primal):
        parser.error("--primal-certificate requires --model and cannot be combined with --exact-primal")
    if args.conic is None and (args.primal_certificate is None or args.solution or args.scaled_solution):
        parser.error("numerical diagnostics require --conic")
    if (args.exact_primal or (args.exact_dual and args.dual_certificate is None)) and args.solution is None:
        parser.error("candidate recovery requires --solution")
    if args.export_dual and not (args.exact_dual or args.dual_certificate):
        parser.error("--export-dual requires a dual candidate")
    if args.denominator <= 0:
        parser.error("--denominator must be positive")
    if args.conic:
        matrix, b, q, sizes, kept, full_dimension, congruence = load_conic(args.conic)
    solution = {}
    if args.solution:
        with np.load(args.solution, allow_pickle=False) as raw:
            solution = {key: raw[key] for key in ("x", "t", "z", "y", "s") if key in raw}
    if args.scaled_solution:
        if congruence is None or args.packing != "lower":
            parser.error("--scaled-solution requires archived lower-packed congruence weights")
        for name in ("z", "y"):
            if name in solution:
                solution[name] = solution[name] * congruence
        if "s" in solution:
            solution["s"] = solution["s"] / congruence
    report = {}
    if args.conic:
        report["numeric"] = conic_diagnostics(matrix, b, q, sizes, solution, args.packing)
    if needs_chart or args.primal_certificate:
        model_hash = model_sha256(args.model)
        model = read_json(args.model)
        if needs_chart:
            document = read_json(args.chart)
            if document["model_sha256"] != model_hash:
                raise ValueError("chart hash differs from the uncompressed face manifest bytes")
            words = MomentManifest(model).representatives
            chart = AffineChart(document, [len(word) % 2 for word in words])
            if full_dimension != len(chart.free) or sizes != [len(block["basis"]) for block in model["psd_blocks"]]:
                raise ValueError("conic dimensions disagree with chart or face manifest")
        face = check_exact_face(model)
        if not face["passed"]:
            raise ValueError("Round 6 face equivalence failed")
        entries = exact_moment_entries(model)
        report["structural_face_equivalence_verified"] = True
        report["stage"] = model["stage"]
        report["model_sha256"] = model_hash
        if args.primal_certificate:
            candidate = read_json(args.primal_certificate)
            moments = read_primal_moments(candidate, len(model["moments"]))
            report["exact_primal"] = exact_primal_report(model, entries, moments)
            report["candidate_sha256"] = model_sha256(args.primal_certificate)
        if args.compare_pencil:
            report["numeric"]["pencil_coefficient_discrepancy"] = compare_implemented_pencil(
                chart, entries, sizes, matrix, b, args.packing, kept)
        if args.audit_chart or args.exact_dual or args.dual_certificate:
            rank = chart.verify_coverage(selected_equalities(chart, model))
            report["chart"] = {"selected_row_rank_verified": rank, "chart_covers_selected_equalities": True,
                               "free_dimension": full_dimension, "archived_pencil_columns": len(kept)}
        if args.exact_primal:
            x = finite_vector(solution, ("x", "t"), matrix.shape[1])
            if x is None:
                report["exact_primal"] = {"passed": False, "reason": "no finite reduced primal coordinates"}
            else:
                free = lift_numeric_free(x, kept, full_dimension, args.denominator)
                report["exact_primal"] = exact_primal_report(model, entries, chart.lift(free))
                report["exact_primal"].update(free_coordinate_denominator=args.denominator,
                                              retained_chart_columns=kept)
        if args.exact_dual or args.dual_certificate:
            if args.dual_certificate:
                candidate = read_json(args.dual_certificate)
                if candidate["model_sha256"] != model_hash:
                    raise ValueError("dual certificate manifest hash mismatch")
                denominator = candidate["gram_factor_denominator"]
                factors = candidate["factors"]
                weights = {int(i): rational(w) for i, w in candidate.get("box_weights", {}).items()}
            else:
                z = finite_vector(solution, ("z", "y"), matrix.shape[0])
                if z is None:
                    raise ValueError("no finite reduced dual vector")
                if maxabs(z):
                    z = z/maxabs(z)
                denominator = args.denominator
                factors = [rounded_factor(block, denominator) for block in unpack_blocks(z, sizes, args.packing)]
                weights = None
            conditional = any(isinstance(w, dict) and w.get("type") == "fixed_moment"
                              for w in model["equality_witnesses"])
            dual, exact_weights = exact_dual_report(chart, entries, moment_bounds(words), factors,
                                                    denominator, args.tighten_box, weights)
            report["exact_dual"] = {**dual, "selected_row_rank_verified": rank, "conditional": conditional,
                                    "bounded_relaxation_refuted": dual["passed"],
                                    "etf_nonexistence_verified": dual["passed"] and not conditional}
            if args.export_dual:
                output = {"model_sha256": model_hash, "gram_factor_denominator": denominator, "factors": factors,
                          "box_weights": {i: str(w) for i, w in exact_weights.items()}}
                args.export_dual.write_text(json.dumps(output, separators=(",", ":"))+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return int(wants_exact and not any(report.get(key, {}).get("passed") for key in ("exact_primal", "exact_dual")))


if __name__ == "__main__":
    raise SystemExit(main())
