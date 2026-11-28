"""CLI regressions for SOS infeasibility certificates."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "src" / "verify_sdp_cert.py"


def certificate():
    # (-1) = x^2 + y^2 - (1+x^2+y^2), with one unused basis vector.
    return {
        "format": "etf-sos-v1", "mode": "exact", "variables": ["x", "y"],
        "constraints": {"g": [[[0, 0], 1], [[2, 0], 1], [[0, 2], 1]]},
        "multipliers": {"g": [[[0, 0], -1]]},
        "sos": [{"basis": [[0, 0], [1, 0], [0, 1]], "Q": [[0, 0, 0], [0, 1, 0], [0, 0, 1]]}],
    }


class SDPVerifierTests(unittest.TestCase):
    def run_cli(self, doc, trusted=None):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "candidate.json"
            path.write_text(json.dumps(doc), encoding="utf-8")
            arguments = [str(path)]
            if trusted is not None:
                path = Path(folder) / "trusted.json"
                path.write_text(json.dumps(trusted), encoding="utf-8")
                arguments.extend(["--trusted-constraints", str(path)])
            run = subprocess.run([sys.executable, "-B", str(SCRIPT), *arguments], capture_output=True, text=True)
            self.assertEqual(run.stderr, "")
            return run.returncode, json.loads(run.stdout)

    def test_exact_singular_psd(self):
        code, report = self.run_cli(certificate())
        self.assertEqual(code, 0)
        self.assertTrue(report["polynomial_system_refuted"])
        self.assertEqual(report["constraint_provenance"]["status"], "unchecked")
        self.assertEqual(report["psd"]["blocks"][0]["rank"], 2)

    def test_weighted_square_and_numeric_psd_repair(self):
        doc = {
            "format": "etf-sos-v1", "mode": "exact", "variables": ["x"],
            "constraints": {"g": [[[0], 1], [[2], "3/2"]]},
            "multipliers": {"g": [[[0], -1]]},
            "sos": [{"weight": "3/2", "polynomial": [[[1], 1]]}],
        }
        self.assertEqual(self.run_cli(doc)[0], 0)
        doc["mode"] = "numeric"
        doc["sos"].append({"weight": -1e-10, "polynomial": [[[0], 1]]})
        code, report = self.run_cli(doc)
        self.assertEqual(code, 0)
        self.assertEqual(report["psd"]["blocks"][1]["certified_gershgorin_psd_shift"], "1/10000000000")
        self.assertEqual(report["psd_repair_identity_coefficient_l1_bound"], "1/10000000000")
        self.assertFalse(report["polynomial_system_refuted"])

    def test_tampered_identity_with_psd_matrix(self):
        doc = certificate()
        doc["multipliers"]["g"][0][1] = -2
        code, report = self.run_cli(doc)
        self.assertEqual(code, 1)
        self.assertFalse(report["identity"]["holds"])
        self.assertTrue(report["psd"]["holds"])
        self.assertFalse(report["polynomial_system_refuted"])

    def test_tampered_psd_with_exact_identity(self):
        doc = certificate()
        # This matrix has a positive diagonal and an indefinite 2-by-2 principal submatrix.
        doc["sos"][0]["Q"][1][2] = doc["sos"][0]["Q"][2][1] = 2
        doc["constraints"]["g"].append([[1, 1], 4])
        code, report = self.run_cli(doc)
        self.assertEqual(code, 1)
        self.assertTrue(report["identity"]["holds"])
        self.assertFalse(report["psd"]["holds"])

    def test_quadratic_field_and_zero_pivot(self):
        # (sqrt(2)-1)(x+y)^2 - (1 + (sqrt(2)-1)(x+y)^2) = -1.
        doc = certificate()
        doc["radicand"] = 2
        a = [-1, 1]
        doc["sos"][0]["Q"] = [[0, 0, 0], [0, a, a], [0, a, a]]
        doc["constraints"]["g"] = [[[0, 0], 1], [[2, 0], a], [[1, 1], [-2, 2]], [[0, 2], a]]
        code, report = self.run_cli(doc)
        self.assertEqual(code, 0)
        self.assertEqual(report["psd"]["blocks"][0]["rank"], 1)

    def test_quadratic_field_negative_pivot_near_zero(self):
        doc = certificate()
        doc["radicand"] = 2
        # This positive rational convergent is strictly larger than sqrt(2).
        a = ["-114243/80782", 1]
        doc["sos"][0]["Q"] = [[0, 0, 0], [0, a, 0], [0, 0, 1]]
        doc["constraints"]["g"][1][1] = a
        code, report = self.run_cli(doc)
        self.assertEqual(code, 1)
        self.assertTrue(report["identity"]["holds"])
        self.assertFalse(report["psd"]["holds"])

    def test_zero_diagonal_requires_zero_row(self):
        doc = certificate()
        doc["sos"][0]["Q"][0][1] = doc["sos"][0]["Q"][1][0] = 1
        doc["constraints"]["g"].append([[1, 0], 2])
        code, report = self.run_cli(doc)
        self.assertEqual(code, 1)
        self.assertTrue(report["identity"]["holds"])
        self.assertFalse(report["psd"]["holds"])

    def test_trusted_constraints_are_compared_as_polynomials(self):
        doc = certificate()
        trusted = {"format": "etf-sos-constraints-v1", "variables": doc["variables"],
                   "constraints": {"g": list(reversed(doc["constraints"]["g"]))}}
        code, report = self.run_cli(doc, trusted)
        self.assertEqual(code, 0)
        self.assertEqual(report["constraint_provenance"]["status"], "matched")
        self.assertFalse(report["etf_nonexistence_verified"])
        changed = deepcopy(doc)
        changed["constraints"]["g"].append([[1, 1], 4])
        changed["sos"][0]["Q"][1][2] = changed["sos"][0]["Q"][2][1] = 2
        code, report = self.run_cli(changed, trusted)
        self.assertEqual(code, 1)
        self.assertEqual(report["constraint_provenance"]["status"], "mismatch")

    def test_numeric_candidate_cannot_pass_as_exact(self):
        doc = certificate()
        doc["mode"] = "numeric"
        doc["sos"][0]["Q"][1][1] = 1.00000000001
        code, report = self.run_cli(doc)
        self.assertEqual(code, 0)
        self.assertTrue(report["numerically_consistent"])
        self.assertFalse(report["polynomial_system_refuted"])
        self.assertFalse(report["etf_nonexistence_verified"])
        doc["mode"] = "exact"
        self.assertEqual(self.run_cli(doc)[0], 2)


class RoundedFactorVerifierTests(unittest.TestCase):
    def run_fixture(self, factor, coefficient="-408", false_etf_model=False):
        roots = ["1", "sqrt(2)", "sqrt(5)", "sqrt(10)"]
        model = {"schema": "search-sdp-model-v1", "variables": ["x"], "coefficient_basis": roots,
                 "equations": {"g": [[[], ["1", "0", "0", "0"]],
                                      [[0, 0], ["1", "0", "0", "0"]],
                                      [[0], ["577", coefficient, "0", "0"]]]}}
        cert = {"schema": "search-sdp-certificate-v1", "coefficient_basis": roots,
                "factor_denominator": 1, "multipliers": {"g": [[[], "-1"]]},
                "psd_blocks": [{"basis": [[0]], "factor_integer_rows": [[factor]]}],
                "exact_proof": True, "residual_upper_bound_on_unit_box": "0"}
        if false_etf_model:
            model["branch"] = "fixed3_negative"
            model["variables"] = [f"x{i}" for i in range(30)]
            model["equations"]["circle_0"] = model["equations"].pop("g")
            cert["multipliers"]["circle_0"] = cert["multipliers"].pop("g")
        with TemporaryDirectory() as folder:
            base = Path(folder)
            for name, document in (("model", model), ("cert", cert)):
                (base / f"{name}.json").write_text(json.dumps(document), encoding="ascii")
            run = subprocess.run([sys.executable, "-B", str(SCRIPT), str(base / "cert.json"),
                                  "--model", str(base / "model.json")], capture_output=True, text=True)
            self.assertEqual(run.stderr, "")
            return run.returncode, json.loads(run.stdout)

    def test_exact_interval_cancels_biquadratic_components(self):
        from fractions import Fraction

        code, report = self.run_fixture(1)
        self.assertEqual(code, 0)
        self.assertLess(Fraction(report["residual_upper_bound_on_unit_box"]), Fraction(1, 1000))
        self.assertTrue(report["polynomial_system_refuted_on_unit_box"])
        self.assertFalse(report["etf_nonexistence_verified"])

    def test_rejects_tampered_factor_and_false_claim(self):
        code, report = self.run_fixture(2)
        self.assertEqual(code, 1)
        self.assertFalse(report["polynomial_system_refuted_on_unit_box"])
        self.assertTrue(report["psd_by_exact_factorization"])

    def test_refuted_forged_etf_equation_fails_provenance(self):
        code, report = self.run_fixture(1, false_etf_model=True)
        self.assertEqual(code, 1)
        self.assertTrue(report["polynomial_system_refuted_on_unit_box"])
        self.assertEqual(report["constraint_provenance"]["mismatched_equations"], ["circle_0"])
        self.assertFalse(report["etf_nonexistence_verified"])


class AnchoredVerifierTests(unittest.TestCase):
    def test_anchored_reynolds_residual_and_provenance(self):
        roots = ["1", "sqrt(2)", "sqrt(5)", "sqrt(10)"]
        variables = [[kind, i, j] for i in range(8) for j in range(i+1, 8) for kind in ("x", "y")]
        model = {"schema": "search-sdp-model-v1", "branch": "anchored_general_4_9",
                 "coefficient_basis": roots, "variables": variables,
                 "equations": {"row_imag": [[[i], [1, 0, 0, 0]] for i in range(1, 14, 2)]}}
        cert = {"schema": "search-sdp-certificate-v1", "coefficient_basis": roots,
                "factor_denominator": 1, "multipliers": {"row_imag": [[[], 1]]},
                "psd_blocks": [], "reynolds": "s8_conjugation"}
        with TemporaryDirectory() as folder:
            base = Path(folder)
            (base / "cert.json").write_text(json.dumps(cert), encoding="ascii")
            for bound, status in (("1", "matched"), ("2", "mismatch")):
                (base / "model.json").write_text(json.dumps(model), encoding="ascii")
                run = subprocess.run([sys.executable, "-B", str(SCRIPT), str(base / "cert.json")],
                                     capture_output=True, text=True)
                self.assertEqual(run.stderr, "")
                self.assertEqual(run.returncode, 1, run.stdout)
                report = json.loads(run.stdout)
                self.assertEqual(report["constraint_provenance"]["status"], status)
                self.assertEqual(report["residual_upper_bound_on_unit_box"], bound)
                self.assertFalse(report["etf_nonexistence_verified"])
                model["equations"]["row_imag"].append([[], [1, 0, 0, 0]])

    def test_conditional_box_dual_replays_the_exact_psd_and_margin(self):
        variables = [[kind, i, j] for i in range(8) for j in range(i+1, 8) for kind in ("x", "y")]
        model = {"variables": variables, "coefficient_basis": ["1", "sqrt(10)/5"],
                 "moments": [[], [0]], "equalities": [{"0": [-2, 0], "1": [1, 0]}],
                 "equality_witnesses": [{"type": "fixed_moment", "monomial": [0], "value": ["2", "0"]}],
                 "psd_blocks": [{"basis": [[[[], 1]]]}]}
        cert = {"schema": "anchored-moment-box-dual-v1", "coefficient_basis": model["coefficient_basis"],
                "equality_dual": ["-1", "-1/2"], "psd_factor_denominator": 4,
                "psd_factor_integer_rows": [[[1]]]}
        with TemporaryDirectory() as folder:
            base = Path(folder)
            (base / "candidate.json").write_text(json.dumps(cert), encoding="ascii")
            for valid in (True, False):
                (base / "solver_model.json").write_text(json.dumps(model), encoding="ascii")
                run = subprocess.run([sys.executable, "-B", str(SCRIPT), str(base / "candidate.json")],
                                     capture_output=True, text=True)
                self.assertEqual(run.stderr, "")
                report = json.loads(run.stdout)
                if valid:
                    self.assertEqual(run.returncode, 0, run.stdout)
                    self.assertEqual(report["margin"], "7/16")
                    self.assertTrue(report["conditional"])
                    self.assertFalse(report["etf_nonexistence_verified"])
                else:
                    self.assertEqual(run.returncode, 2, run.stdout)
                    self.assertIn("independent expansion", report["error"])
                model["equalities"][0]["0"] = [-3, 0]

    def test_schur_rank_minor_matches_full_gram_minor(self):
        import sympy as sp

        sys.path.insert(0, str(SCRIPT.parent))
        self.addCleanup(sys.path.pop, 0)
        from verify_sdp_anchored import canonical_rank_minor

        symbols, real, imag = canonical_rank_minor("schur", (0, 1, 2, 3), (0, 1, 2, 4))
        for offset in (0, 4):
            values = {x: sp.Rational((i + offset) % 7 - 3, 11) for i, x in enumerate(symbols)}
            _, full_real, full_imag = canonical_rank_minor("gram", (0, 1, 2, 3, 8), (0, 1, 2, 4, 8))
            self.assertEqual(sp.simplify(real.subs(values) - full_real.subs(values)), 0)
            self.assertEqual(sp.simplify(imag.subs(values) - full_imag.subs(values)), 0)

    def test_anchored_primal_rejects_wrong_constant_moment(self):
        with TemporaryDirectory() as folder:
            base = Path(folder)
            model = {"variables": [[kind, i, j] for i in range(8) for j in range(i+1, 8) for kind in ("x", "y")],
                     "coefficient_basis": ["1", "sqrt(10)/5"], "moments": [[]]}
            candidate = {"schema": "anchored-exact-primal-v1", "coefficient_basis": model["coefficient_basis"],
                         "moments": [["2", "0"]]}
            (base / "anchored_model.json").write_text(json.dumps(model), encoding="ascii")
            (base / "candidate.json").write_text(json.dumps(candidate), encoding="ascii")
            run = subprocess.run([sys.executable, "-B", str(SCRIPT), str(base / "candidate.json")],
                                 capture_output=True, text=True)
            self.assertEqual(run.stderr, "")
            self.assertEqual(run.returncode, 2)
            self.assertIn("constant moment must equal 1", json.loads(run.stdout)["error"])

    def test_factor_for_nonmonomial_basis_and_tamper(self):
        roots = ["1", "sqrt(2)", "sqrt(5)", "sqrt(10)"]
        model = {"schema": "search-sdp-model-v1", "variables": ["x", "y"], "coefficient_basis": roots,
                 "equations": {"g": [[[], [1, 0, 0, 0]], [[0, 0], [1, 0, 0, 0]],
                                      [[0, 1], [-2, 0, 0, 0]], [[1, 1], [1, 0, 0, 0]]]}}
        cert = {"schema": "search-sdp-certificate-v1", "coefficient_basis": roots,
                "factor_denominator": 1, "multipliers": {"g": [[[], -1]]},
                "psd_blocks": [{"polynomial_basis": [[[[0], 1], [[1], -1]]],
                                "factor_integer_rows": [[1]]}]}
        with TemporaryDirectory() as folder:
            base = Path(folder)
            (base / "model.json").write_text(json.dumps(model), encoding="ascii")
            for factor, status, bound in ((1, 0, "0"), (2, 1, "12")):
                cert["psd_blocks"][0]["factor_integer_rows"] = [[factor]]
                (base / "cert.json").write_text(json.dumps(cert), encoding="ascii")
                run = subprocess.run([sys.executable, "-B", str(SCRIPT), str(base / "cert.json")],
                                     capture_output=True, text=True)
                self.assertEqual(run.stderr, "")
                self.assertEqual(run.returncode, status, run.stdout)
                self.assertEqual(json.loads(run.stdout)["residual_upper_bound_on_unit_box"], bound)


if __name__ == "__main__":
    unittest.main()
