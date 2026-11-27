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


if __name__ == "__main__":
    unittest.main()
