"""Independent audit of the principal-coordinate facial reduction."""

from collections import defaultdict
from fractions import Fraction

from verify_sdp_anchored import EDGE_INDEX, VARIABLES
from verify_sdp_factor_cert import monomial
from verify_sdp_level3 import (
    MomentManifest, beta_interval, canonical_psd_bases, declared_polynomials, equation_terms,
    pair, row_terms, selected_monomials, validate_polynomial_manifest,
)


def validate_complement(size, directions, pivots, keep):
    if sorted(pivots + keep) != list(range(size)) or len(directions) != len(pivots):
        raise ValueError("invalid partition into pivot and retained coordinates")
    # Restriction to the pivot columns is invertible iff the indicated
    # principal coordinates span a complement of the specified kernel.
    from sympy.polys.matrices import DomainMatrix
    from verify_sdp_cert import RealField

    domain = RealField(10).domain
    beta = domain.unit / domain.convert(5)
    zero = (0, 0)
    matrix = [[domain.convert(a) + beta*domain.convert(b) for a, b in (q.get(i, zero) for i in pivots)]
              for q in directions]
    if pivots and DomainMatrix(matrix, (len(pivots), len(pivots)), domain).rank() != len(pivots):
        raise ValueError("kernel directions are not independent on the pivot coordinates")


def normalized(row):
    """Remove a nonzero scalar in Q from a row for exact membership tests."""
    ordered = sorted((i, a, b) for i, (a, b) in row.items() if a or b)
    if not ordered:
        return ()
    first = ordered[0][1] or ordered[0][2]
    return tuple((i, Fraction(a, first), Fraction(b, first)) for i, a, b in ordered)


def shifted_origin(origin, allowed):
    name, perm, multiplier = origin
    if name not in {"circle", "row_real", "row_imag", "tight_real", "tight_imag"}:
        raise ValueError("unsupported structural kernel equation")
    if sorted(perm) != list(range(8)):
        raise ValueError("invalid kernel witness permutation")
    multiplier = monomial(multiplier, len(VARIABLES))
    equation = equation_terms(name)
    image = {}
    for word, (a, b) in equation.items():
        target = list(multiplier)
        sign = 1
        for var in word:
            kind, i, j = VARIABLES[var]
            left, right = perm[i], perm[j]
            is_imag = kind == "y"
            target.append(EDGE_INDEX[min(left, right), max(left, right)] + is_imag)
            if is_imag and left > right:
                sign = -sign
        power = tuple(sorted(target))
        if power not in allowed:
            raise ValueError("kernel polynomial is outside the PSD basis support")
        image[power] = sign*a, sign*b
    return name, multiplier, equation, image


def check_origin_norm(manifest, name, multiplier, equation, perm, necessary_rows):
    # By S8 invariance undo the permutation in h times its monomial factor.
    # Expand L((h*m)^2) into witnessed rows L(h*t), with t=m? times one term of h.
    inverse = {v: i for i, v in enumerate(perm)}
    old_monomial = []
    for var in multiplier:
        kind, a, b = VARIABLES[var]
        a, b = inverse[a], inverse[b]
        old_monomial.append(EDGE_INDEX[min(a, b), max(a, b)] + (kind == "y"))
    for word in equation:
        power = sorted(word + tuple(old_monomial)*2)
        row, _ = manifest.row([name, power, 1])
        primitive = normalized(row)
        if primitive and primitive not in necessary_rows:
            return False
    return True


def check_exact_face(model, source=None):
    """Rebuild original affine blocks, then check every quotient and cross row."""
    manifest = MomentManifest(model)
    face = model["face_reduction"]
    if face["schema"] != "level3-structural-face-v1":
        raise ValueError("unsupported structural facial reduction schema")
    base_count = face["input_equality_count"]
    rows = model["equalities"]
    witnesses = model["equality_witnesses"]
    if len(rows) != len(witnesses) or len(rows) != base_count + face["added_equality_count"]:
        raise ValueError("facial equality counts disagree")
    if source is not None:
        for field in ("stage", "variables", "moments", "coefficient_basis", "base_equations"):
            if source[field] != model[field]:
                raise ValueError(f"source manifest disagrees on {field}")
        if source["equalities"] != rows[:base_count] or source["equality_witnesses"] != witnesses[:base_count]:
            raise ValueError("source manifest original equality rows disagree")
    moment_count = len(manifest.representatives)
    checked_rows = []
    for i, (row, witness) in enumerate(zip(rows[:base_count], witnesses[:base_count])):
        expected, kind = manifest.row(witness)
        if expected != row_terms(row, moment_count):
            raise ValueError(f"original equality witness {i} disagrees")
        checked_rows.append((expected, kind))
    polynomial_count = validate_polynomial_manifest(model)
    all_base = {normalized(row) for row, kind in checked_rows if kind != "fixed_moment"}
    all_declared = {normalized(row) for row, _ in checked_rows}
    cross = {}
    for i in range(base_count, len(rows)):
        name, block, direction, coordinate, divisor = witnesses[i]
        if name != "face_kernel" or (block, direction, coordinate) in cross or type(divisor) is not int or not divisor:
            raise ValueError("invalid or duplicate face cross-row witness")
        cross[block, direction, coordinate] = row_terms(rows[i], moment_count), divisor
    all_declared.update(normalized(r) for r, _ in cross.values())

    stage = model["stage"]
    allowed = set(selected_monomials(stage))
    canonical = canonical_psd_bases(stage)
    seen_characters = set()
    unsupported = []
    block_results = []
    checked_crosses = 0
    for bi, (metadata, quotient) in enumerate(zip(face["blocks"], model["psd_blocks"], strict=True)):
        character = tuple(quotient["character"])
        if character in seen_characters or tuple(metadata["character"]) != character or character not in canonical:
            raise ValueError("invalid facial block character")
        seen_characters.add(character)
        basis = canonical[character]
        size = len(basis)
        if metadata["original_size"] != size or metadata["kernel_dimension"] != len(metadata["directions"]):
            raise ValueError("facial block sizes disagree")
        pivots, keep = metadata["pivot_indices"], metadata["retained_indices"]
        directions = [row_terms(w["coordinates"], size) for w in metadata["directions"]]
        validate_complement(size, directions, pivots, keep)
        if declared_polynomials(quotient) != [basis[i] for i in keep]:
            raise ValueError("quotient basis differs from chosen principal coordinates")
        projection = {term: (i, weight, len(poly)) for i, poly in enumerate(basis) for term, weight in poly.items()}
        by_column = set().union(*(set(direction) for direction in directions))

        # Recompute each affine entry needed for a kernel cross-row or the quotient.
        # Store rows sparsely and reuse them across directions.
        entries = {}
        for i in by_column:
            entries[i] = [manifest.character_product(basis[i], p) for p in basis]
        declared = {}
        for qi, qj, value in quotient["entries"]:
            if (qi, qj) in declared or not 0 <= qj <= qi < len(keep):
                raise ValueError("invalid or duplicate quotient entry")
            declared[qi, qj] = row_terms(value, moment_count)
        for qi, i in enumerate(keep):
            for qj, j in enumerate(keep[:qi+1]):
                value = entries[i][j] if i in entries else (
                    entries[j][i] if j in entries else manifest.character_product(basis[i], basis[j]))
                if {k: (c, 0) for k, c in value.items()} != declared.get((qi, qj), {}):
                    raise ValueError(f"quotient entry does not match the original moment form in block {bi}")

        before = checked_crosses
        for di, (direction, witness) in enumerate(zip(directions, metadata["directions"], strict=True)):
            name, multiplier, polynomial, image = shifted_origin(witness["origin"], allowed)
            numerator, denominator = witness["projection_scale"]
            scale = Fraction(numerator, denominator)
            if not scale:
                raise ValueError("zero kernel projection scale")
            q = defaultdict(lambda: [0, 0])
            for word, (a, b) in image.items():
                if word in projection:
                    i, sign, length = projection[word]
                    w = scale * Fraction(sign, length)
                    q[i][0] += w*a
                    q[i][1] += w*b
            if {i: tuple(c) for i, c in q.items() if any(c)} != direction:
                raise ValueError(f"kernel projection witness disagrees in block {bi}, direction {di}")
            if not check_origin_norm(manifest, name, multiplier, polynomial, witness["origin"][1], all_base):
                unsupported.append([bi, di])

            for i in range(size):
                result = defaultdict(lambda: [0, 0])
                for j, (a, b) in direction.items():
                    for moment, coefficient in entries[j][i].items():
                        result[moment][0] += a*coefficient
                        result[moment][1] += b*coefficient
                kernel_row = {k: tuple(c) for k, c in result.items() if any(c)}
                if normalized(kernel_row) and normalized(kernel_row) not in all_declared:
                    raise ValueError(f"missing kernel cross-row in block {bi}, direction {di}, coordinate {i}")
                raw = cross.pop((bi, di, i), None)
                if raw is not None:
                    supplied, divisor = raw
                    if {k: (a/divisor, b/divisor) for k, (a, b) in kernel_row.items()} != supplied:
                        raise ValueError(f"kernel cross-row disagrees in block {bi}, direction {di}, coordinate {i}")
                    checked_crosses += 1
        if metadata["added_cross_constraints"] != checked_crosses-before:
            raise ValueError("block cross-row count disagrees")
        block_results.append({"character": list(character), "original_size": size,
                              "kernel_dimension": len(directions), "quotient_size": len(keep)})
    if seen_characters != set(canonical) or cross:
        raise ValueError("missing character space or unmatched cross-row")
    return {"mode": "exact_structural_face_audit", "passed": not unsupported,
            "etf_preserving_face_verified": True, "equivalence_to_original_relaxation_verified": not unsupported,
            "stage": stage, "declared_polynomials_verified": polynomial_count,
            "original_equalities_checked": base_count, "cross_rows_checked": checked_crosses,
            "norm_witnesses_outside_direct_original_row_span": unsupported, "blocks": block_results,
            **({"source_equality_prefix_verified": True} if source is not None else {})}


def verify_negative_direction(candidate, model, witness):
    """Certify a negative quadratic form in a subspace of the original PSD block."""
    stage = model["stage"]
    manifest = MomentManifest(model)
    block = model["psd_blocks"][witness["block"]]
    character = tuple(block["character"])
    basis = declared_polynomials(block)
    exact_basis = canonical_psd_bases(stage)[character]
    keep = model["face_reduction"]["blocks"][witness["block"]]["retained_indices"]
    if basis != [exact_basis[i] for i in keep]:
        raise ValueError("witness basis is not the indicated principal subspace")
    vector = witness["integer_test_vector"]
    if len(vector) != len(basis) or any(type(x) is not int for x in vector) or not any(vector):
        raise ValueError("invalid integer quadratic-form witness")
    values = [pair(x) for x in candidate["moments"]]
    if len(values) != len(manifest.representatives):
        raise ValueError("candidate moment count does not match the face manifest")
    coefficients = defaultdict(int)
    for i, left in enumerate(basis):
        for j, right in enumerate(basis[:i+1]):
            weight = vector[i]*vector[j]*(1 if i == j else 2)
            if weight:
                for index, c in manifest.character_product(left, right).items():
                    coefficients[index] += c*weight
    exact = [sum(weight*values[index][slot] for index, weight in coefficients.items()) for slot in range(2)]
    if tuple(exact) != pair(witness["quadratic_form_components"]):
        raise ValueError("reported negative quadratic form disagrees with exact evaluation")
    _, upper = beta_interval(exact)
    normalized_upper = upper / sum(x*x for x in vector)
    return {"mode": "exact_negative_psd_direction", "passed": upper < 0, "stage": stage,
            "passes_psd": False if upper < 0 else None, "block_character": list(character),
            "quadratic_form_components": list(map(str, exact)),
            "normalized_quadratic_form_upper_bound_exact": str(normalized_upper),
            "normalized_quadratic_form_upper_bound_approx": float(normalized_upper)}
