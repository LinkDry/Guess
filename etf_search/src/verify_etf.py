"""Independently verify a complex ETF Gram matrix.

Default input is a numeric .npy array. With --symbolic, supply a UTF-8 JSON
object {"format": "etf-gram-v1", "gram": [[...], ...]}. Each entry is a JSON
integer or a string with integer literals, parentheses, +, -, *, /, ** with a
signed integer literal exponent, I, sqrt(q), or zeta(k). Here q is a nonnegative
integer or a ratio of integer literals, and k is a positive integer with
zeta(k) = exp(2*pi*I/k). For example, "(1+I)/(2*sqrt(3))" is exact. Symbols,
decimals, attributes, and general function calls are outside the format.

Exit codes are 0 for pass, 1 for a failed ETF condition, and 2 for invalid input.
Symbolic first_failure indices are zero-based. Numeric metrics are null if
arithmetic overflows to a nonfinite value, in which case that check fails.
"""

import argparse
import ast
from fractions import Fraction
import json
import math
import operator
from pathlib import Path
from zipfile import BadZipFile


TOLERANCE = 1e-12
CHECK_METRICS = {
    "hermitian": "hermitian_max_error",
    "unit_diagonal": "diagonal_max_error",
    "welch_modulus": "welch_max_error",
    "tightness": "tightness_frobenius",
}
BINARY_OPERATIONS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def verify_numeric(path, d, n):
    import numpy as np

    with path.open("rb") as stream:
        gram = np.load(stream, allow_pickle=False)
        if isinstance(gram, np.lib.npyio.NpzFile):
            gram.close()
            raise ValueError("expected a single NPY array")
    if gram.shape != (n, n):
        raise ValueError(f"expected matrix shape {(n, n)}, got {gram.shape}")
    if gram.dtype.kind not in "iufc":
        raise ValueError("expected a real or complex numeric dtype")
    if not np.isfinite(gram).all():
        raise ValueError("matrix contains NaN or infinity")

    gram = gram.astype(np.promote_types(gram.dtype, np.complex128), copy=False)
    real_type = gram.real.dtype.type
    mu = np.sqrt(real_type(n - d) / real_type(d * (n - 1)))
    off_diagonal = ~np.eye(n, dtype=bool)
    with np.errstate(over="ignore", invalid="ignore"):
        residual = gram @ gram - (real_type(n) / d) * gram
        metrics = {
            "hermitian_max_error": np.max(np.abs(gram - gram.conj().T)),
            "diagonal_max_error": np.max(np.abs(gram.diagonal() - 1)),
            "welch_max_error": np.max(np.abs(np.abs(gram[off_diagonal]) - mu)),
            "tightness_frobenius": np.linalg.norm(residual, ord="fro"),
        }
    checks = {
        check: bool(np.isfinite(metrics[key]) and metrics[key] < TOLERANCE)
        for check, key in CHECK_METRICS.items()
    }

    return {
        "passed": all(checks.values()),
        "mu": float(mu),
        "tolerance": TOLERANCE,
        "checks": checks,
        "metrics": {
            key: float(value) if math.isfinite(float(value)) else None
            for key, value in metrics.items()
        },
    }


def _integer_literal(node):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        if isinstance(node.operand, ast.Constant) and type(node.operand.value) is int:
            return -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value
    raise ValueError("expected an integer literal")


def _rational_literal(node):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return Fraction(_integer_literal(node.left), _integer_literal(node.right))
    return Fraction(_integer_literal(node))


def _parse_expression(node, generators):
    """Translate the AST into operations on integers and named algebraic generators."""
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _parse_expression(node.operand, generators)
        return operand if isinstance(node.op, ast.UAdd) else (operator.neg, operand)
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow):
            return ("power", _parse_expression(node.left, generators), _integer_literal(node.right))
        operation = BINARY_OPERATIONS.get(type(node.op))
        if operation is not None:
            return (
                operation,
                _parse_expression(node.left, generators),
                _parse_expression(node.right, generators),
            )
    if isinstance(node, ast.Name) and node.id == "I":
        generator = ("zeta", 4)
        generators.add(generator)
        return generator
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and len(node.args) == 1 and not node.keywords:
        name, arg = node.func.id, node.args[0]
        if name == "zeta":
            order = _integer_literal(arg)
            if order < 1:
                raise ValueError("zeta(k) requires k >= 1")
            generator = (name, order)
            generators.add(generator)
            return generator
        if name == "sqrt":
            radicand = _rational_literal(arg)
            if radicand < 0:
                raise ValueError("sqrt(q) requires q >= 0")
            generator = (name, radicand)
            generators.add(generator)
            return generator
    raise ValueError("unsupported exact expression")


def _exact_value(expression, field, generators):
    if type(expression) is int:
        return field(expression)
    operation, *arguments = expression
    if operation in ("sqrt", "zeta"):
        return generators[expression]
    if operation == "power":
        base = _exact_value(arguments[0], field, generators)
        exponent = arguments[1]
        if base == field.zero and exponent < 0:
            raise ZeroDivisionError("zero cannot be raised to a negative power")
        return base ** exponent
    operands = [_exact_value(arg, field, generators) for arg in arguments]
    if operation is operator.truediv and operands[1] == field.zero:
        raise ZeroDivisionError("division by zero")
    return operation(*operands)


def verify_symbolic(path, d, n):
    import sympy as sp

    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict) or document.get("format") != "etf-gram-v1":
        raise ValueError('expected JSON object with "format": "etf-gram-v1"')
    rows = document.get("gram")
    if not isinstance(rows, list) or len(rows) != n or any(
        not isinstance(row, list) or len(row) != n for row in rows
    ):
        raise ValueError(f"expected matrix shape {(n, n)} in 'gram'")

    generators = set()
    expressions = []
    for i, row in enumerate(rows):
        parsed_row = []
        for j, entry in enumerate(row):
            try:
                if type(entry) is int:
                    parsed_row.append(entry)
                elif isinstance(entry, str):
                    parsed_row.append(_parse_expression(ast.parse(entry.strip(), mode="eval").body, generators))
                else:
                    raise ValueError("expected a JSON integer or an exact expression string")
            except (SyntaxError, ValueError, ArithmeticError, RecursionError) as exc:
                raise ValueError(f"gram[{i},{j}]: {exc}") from exc
        expressions.append(parsed_row)

    # A common algebraic number field gives canonical element equality.
    exact_generators = {
        (name, value): sp.sqrt(sp.Rational(value.numerator, value.denominator))
        if name == "sqrt" else sp.exp(2 * sp.pi * sp.I / value)
        for name, value in sorted(generators)
    }
    extensions = list(dict.fromkeys(g for g in exact_generators.values() if not g.is_Rational))
    field = sp.QQ.algebraic_field(*extensions) if extensions else sp.QQ
    normal = {key: field.from_sympy(value) for key, value in exact_generators.items()}
    conjugated = {
        key: field.one / value if key[0] == "zeta" else value
        for key, value in normal.items()
    }
    gram = []
    conjugate = []
    cached = {}
    for i, row in enumerate(expressions):
        exact_row, conjugate_row = [], []
        for j, expression in enumerate(row):
            try:
                if expression not in cached:
                    cached[expression] = (
                        _exact_value(expression, field, normal),
                        _exact_value(expression, field, conjugated),
                    )
                entry, entry_conjugate = cached[expression]
            except (ValueError, ArithmeticError, RecursionError) as exc:
                raise ValueError(f"gram[{i},{j}]: {exc}") from exc
            exact_row.append(entry)
            conjugate_row.append(entry_conjugate)
        gram.append(exact_row)
        conjugate.append(conjugate_row)

    mu_squared = sp.Rational(n - d, d * (n - 1))
    exact_mu_squared = field.from_sympy(mu_squared)
    scale = field.from_sympy(sp.Rational(n, d))
    first_failure = {}
    columns = list(zip(*gram))
    for i, row in enumerate(gram):
        for j, entry in enumerate(row):
            if entry != conjugate[j][i]:
                first_failure.setdefault("hermitian", [i, j])
            if i == j:
                if entry != field.one:
                    first_failure.setdefault("unit_diagonal", [i, j])
            elif entry * conjugate[i][j] != exact_mu_squared:
                first_failure.setdefault("welch_modulus", [i, j])

            square_entry = sum((left * right for left, right in zip(row, columns[j])), field.zero)
            if square_entry != scale * entry:
                first_failure.setdefault("tightness", [i, j])

    return {
        "passed": not first_failure,
        "mu_squared": str(mu_squared),
        "checks": {check: check not in first_failure for check in CHECK_METRICS},
        "first_failure": first_failure,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("matrix", metavar="MATRIX", type=Path)
    parser.add_argument("--d", type=int, required=True, help="frame dimension")
    parser.add_argument("--n", type=int, required=True, help="number of frame vectors")
    parser.add_argument("--symbolic", action="store_true", help="read exact etf-gram-v1 JSON")
    args = parser.parse_args()

    verify = verify_symbolic if args.symbolic else verify_numeric
    report = {"mode": "symbolic" if args.symbolic else "numeric", "d": args.d, "n": args.n}
    try:
        if not 1 <= args.d < args.n <= args.d**2:
            raise ValueError("require positive integers with d < n <= d^2")
        report.update(verify(args.matrix, args.d, args.n))
        exit_code = 0 if report["passed"] else 1
    except (ValueError, OSError, EOFError, BadZipFile, ArithmeticError, RecursionError) as exc:
        report.update(passed=False, error=str(exc) or type(exc).__name__)
        exit_code = 2
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
