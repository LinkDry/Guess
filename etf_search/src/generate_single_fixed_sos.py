"""Emit a rational SOS refutation for the single-fixed-line antiunitary model."""

import argparse
from itertools import combinations_with_replacement
import json
from pathlib import Path

import sympy as sp


def make_certificate():
    """Use real columns X,Y and a normalized kernel vector of the bivectors."""
    x = sp.Matrix(3, 4, lambda row, col: sp.Symbol(f"x{row}{col}"))
    y = sp.Matrix(3, 4, lambda row, col: sp.Symbol(f"y{row}{col}"))
    h = sp.Matrix([sp.Symbol(f"h{j}") for j in range(4)])
    variables = [*x, *y, *h]
    scale = sp.Rational(9, 4)
    anchor_squared = sp.Rational(5, 16)
    bivector_squared = sp.Rational(27, 32)
    gap = bivector_squared - anchor_squared * scale
    w = sp.Matrix.hstack(*(x[:, j].cross(y[:, j]) for j in range(4)))
    e = w.T * w + anchor_squared * y.T * y - bivector_squared * sp.eye(4)
    t = x * x.T + y * y.T - scale * sp.eye(3)
    yh = y * h
    wh = w * h

    def encode(coefficient):
        return int(coefficient) if coefficient.q == 1 else str(coefficient)

    def sparse(expression):
        return [[list(exponents), encode(coefficient)]
                for exponents, coefficient in sp.Poly(expression, *variables, domain=sp.QQ).terms()
                if coefficient]

    constraints, multipliers = {}, {}

    def add(name, equation, multiplier):
        constraints[name] = sparse(equation)
        multipliers[name] = sparse(multiplier)

    add("h_unit", h.dot(h) - 1, sp.Integer(1))
    for i, j in combinations_with_replacement(range(4), 2):
        multiplicity = 1 if i == j else 2
        add(f"bivector_{i}{j}", e[i, j], multiplicity * h[i] * h[j] / gap)
    for i, j in combinations_with_replacement(range(3), 2):
        multiplicity = 1 if i == j else 2
        add(f"tight_{i}{j}", t[i, j], -multiplicity * anchor_squared * yh[i] * yh[j] / (scale * gap))
    for i in range(3):
        add(f"kernel_{i}", wh[i], -wh[i] / gap)

    blocks = []
    for expression in [*(scale * h - y.T * yh), *(x.T * yh)]:
        terms = sp.Poly(expression, *variables, domain=sp.QQ).terms()
        blocks.append({
            "basis": [list(exponents) for exponents, _ in terms],
            "Q": [[encode(anchor_squared * c * d / (scale * gap)) for _, d in terms] for _, c in terms],
        })
    trusted = {"format": "etf-sos-constraints-v1", "variables": [str(v) for v in variables],
               "constraints": constraints}
    certificate = {**trusted, "format": "etf-sos-v1", "mode": "exact",
                   "multipliers": multipliers, "sos": blocks}
    return certificate, trusted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    out = parser.parse_args().output_dir
    out.mkdir(parents=True, exist_ok=True)
    certificate, trusted = make_certificate()
    for name, data in [("certificate.json", certificate), ("constraints.json", trusted)]:
        (out / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
