"""Replay the independent numeric verifier on saved Round 2 results."""

import argparse
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from verify_etf import TOLERANCE, verify_numeric


D, N = 4, 9
NORM_ATOL = 1e-14
RESIDUAL = "tightness_frobenius"


def result_file(directory, name):
    path = (directory / name).resolve()
    if not path.is_relative_to(directory):
        raise ValueError(f"Artifact is outside the run directory: {name}")
    return path


def audit(directory):
    directory = directory.resolve()
    summary_bytes = (directory / "summary.json").read_bytes()
    summary = json.loads(summary_bytes)
    unrestricted = "all_final_grams_file" in summary
    model = summary["model"]
    runs = summary["runs"]
    count = summary["settings"]["seed_count" if unrestricted else "seeds"]
    seed_start = summary["settings"]["seed_start"]
    if sorted(row["seed"] for row in runs) != list(range(seed_start, seed_start + count)) or not runs:
        raise ValueError("Run seeds do not match the declared seed range")
    if unrestricted and (model["d"], model["n"]) != (D, N):
        raise ValueError("Expected an unrestricted ETF(4,9) run")

    stack_file = result_file(directory, summary.get("all_final_grams_file", "final_grams.npy"))
    stack_bytes = stack_file.read_bytes()
    stack = np.load(BytesIO(stack_bytes), allow_pickle=False)
    if stack.shape != (count, N, N) or (unrestricted and summary["all_final_grams_shape"] != list(stack.shape)):
        raise ValueError("Final stack shape, summary, and run count disagree")
    rows = sorted(runs, key=lambda row: row["matrix_index"]) if unrestricted else runs
    if unrestricted and [row["matrix_index"] for row in rows] != list(range(count)):
        raise ValueError("Unrestricted matrix indices must enumerate the whole stack exactly once")
    indices_by_seed = {row["seed"]: i for i, row in enumerate(rows)}

    symmetry = None
    if not unrestricted:
        fixed = model["fixed_points"]
        sign = model["triangle_sign"]
        if (fixed, sign) not in ((1, 1), (3, 1), (3, -1)):
            raise ValueError("Invalid antiunitary fixed-point count or triangle sign")
        permutation = list(range(N))
        for i in range(fixed, N, 2):
            permutation[i:i+2] = [i+1, i]
        if model["permutation"] != permutation or model["phase_dimension"] != {1: 16, 3: 15}[fixed]:
            raise ValueError("The antiunitary permutation or phase dimension disagrees with the declared model")
        symmetry = {"permutation": permutation, "fixed_points": fixed, "triangle_sign": sign,
                    "max_conjugation_error": 0.0, "max_anchor_error": 0.0}

    mismatches = []
    metric_maxima = {}
    norm_comparisons = 0

    def compare(label, calculated, reported):
        nonlocal norm_comparisons
        for key, actual in calculated.items():
            if key not in reported:
                continue
            norm_comparisons += 1
            given = reported[key]
            if actual is None or not isinstance(given, (int, float)) or not math.isfinite(given):
                mismatches.append({"matrix": label, "metric": key, "reported": given, "independent": actual})
                continue
            delta = abs(actual - given)
            metric_maxima[key] = max(metric_maxima.get(key, 0.0), delta)
            if delta > NORM_ATOL:
                mismatches.append({"matrix": label, "metric": key, "reported": given, "independent": actual})

    def evaluate(label, gram, work, reported, prior=None):
        np.save(work, gram, allow_pickle=False)
        checked = verify_numeric(work, D, N)
        compare(label, checked["metrics"], reported)
        if prior is not None:
            compare(label + " (earlier verifier)", checked["metrics"], prior["metrics"])
            if checked["passed"] != prior["passed"] or checked["checks"] != prior["checks"]:
                mismatches.append({"matrix": label, "issue": "earlier verifier verdict disagrees"})
        if symmetry is not None:
            permuted = gram[np.ix_(permutation, permutation)]
            symmetry_error = float(np.max(np.abs(permuted - gram.conj())))
            anchors = np.r_[gram[0, 1:], gram[1:, 0]]
            anchor_error = float(np.max(np.abs(anchors - checked["mu"])))
            if fixed == 3:
                anchor_error = max(anchor_error, float(np.max(np.abs(gram[[1, 2], [2, 1]] - sign * checked["mu"]))))
            symmetry["max_conjugation_error"] = max(symmetry["max_conjugation_error"], symmetry_error)
            symmetry["max_anchor_error"] = max(symmetry["max_anchor_error"], anchor_error)
            if max(symmetry_error, anchor_error) >= TOLERANCE:
                mismatches.append({"matrix": label, "issue": "antiunitary or fixed-anchor check failed"})
        return {key: checked[key] for key in ("passed", "checks", "metrics")}

    best_row = summary["top10"][0] if unrestricted else summary["best"]
    best_seed = best_row["seed"]
    best_index = indices_by_seed[best_seed]
    source_key = "final" if unrestricted else "retained"
    if unrestricted and best_row["matrix_index"] != best_index:
        mismatches.append({"matrix": "best", "issue": "top seed and matrix index disagree"})
    best_path = result_file(directory, summary.get("best_gram_file", "best_gram.npy"))
    best_bytes = best_path.read_bytes()
    best_gram = np.load(BytesIO(best_bytes), allow_pickle=False)
    matched_final = bool(np.array_equal(best_gram, stack[best_index]))
    if (unrestricted or best_row[source_key] == rows[best_index]["final"]) and not matched_final:
        mismatches.append({"matrix": "best", "issue": "best file disagrees with corresponding final matrix"})
    if best_row[source_key][RESIDUAL] != min(row[source_key][RESIDUAL] for row in rows):
        mismatches.append({"matrix": "best", "issue": "reported best is not the minimum in its stage"})

    declared_hits = {result_file(directory, row["gram_file"]): row for row in summary.get("double_gate_hits", [])}
    hit_paths = set(declared_hits)
    discovered_hits = [*directory.glob("hit_*.npy"), *(directory / "double_gate_hits").rglob("gram.npy")]
    hit_paths.update(result_file(directory, path) for path in discovered_hits)
    deferred_retained = []
    with TemporaryDirectory(prefix="etf_round2_verify_") as tmp:
        work = Path(tmp) / "gram.npy"
        finals = []
        for i, row in enumerate(rows):
            label = f"final[{i}]"
            checked = evaluate(label, stack[i], work, row["final"], row.get("independent_verification"))
            if not unrestricted and row["retained"] != row["final"]:
                deferred_retained.append(row["seed"])
            if unrestricted:
                metrics = checked["metrics"]
                double_gate = all(metrics[key] is not None and metrics[key] < TOLERANCE
                                  for key in (RESIDUAL, "welch_max_error"))
                if row["final"]["double_gate_passed"] != double_gate:
                    mismatches.append({"matrix": label, "issue": "recorded double-gate flag disagrees"})
            finals.append({"index": i, "seed": row["seed"], **checked})
        best = {"file": best_path.relative_to(directory).as_posix(), "sha256": sha256(best_bytes).hexdigest(),
                "seed": best_seed, "corresponding_final_index": best_index, "matches_corresponding_final": matched_final,
                **evaluate("best", best_gram, work, best_row[source_key])}
        if unrestricted:
            metadata_file = result_file(directory, summary["best_gram_metadata_file"])
            metadata_bytes = metadata_file.read_bytes()
            best_metadata = json.loads(metadata_bytes)
            best["metadata_sha256"] = sha256(metadata_bytes).hexdigest()
            if (best_metadata["seed"], best_metadata["matrix_index"], best_metadata["gram_file"]) != (
                best_seed, best_index, best["file"]
            ):
                mismatches.append({"matrix": "best", "issue": "best metadata seed, index, or file disagrees"})
            compare("best metadata", best["metrics"], best_metadata["final"])
            prior = best_metadata["independent_verification"]
            compare("best metadata (earlier verifier)", best["metrics"], prior["metrics"])
            if best["passed"] != prior["passed"] or best["checks"] != prior["checks"]:
                mismatches.append({"matrix": "best", "issue": "best metadata verifier verdict disagrees"})
            earlier_totals = summary["independent_verification"]
            if earlier_totals["checked"] != len(finals) or earlier_totals["passed"] != sum(
                entry["passed"] for entry in finals
            ):
                mismatches.append({"matrix": "final stack", "issue": "earlier verifier totals disagree"})
        hits = []
        for path in sorted(hit_paths):
            name = path.relative_to(directory).as_posix()
            data = path.read_bytes()
            metadata = declared_hits.get(path, {})
            entry = {"file": name, "sha256": sha256(data).hexdigest()}
            if name.startswith("double_gate_hits/"):
                hit_report = path.with_name("numeric_verification.json")
                recorded = json.loads(hit_report.read_bytes()) if hit_report.exists() else {}
                reported = recorded.get("metrics", {})
                entry.update({key: metadata.get(key, recorded.get(key)) for key in ("seed", "stage")})
            else:
                recorded = {}
                seed = int(path.stem.removeprefix("hit_"))
                entry["seed"] = seed
                reported = rows[indices_by_seed[seed]]["retained"]
            entry.update(evaluate(name, np.load(BytesIO(data), allow_pickle=False), work, reported,
                                  recorded.get("independent_verification")))
            if "independent_pass" in metadata and entry["passed"] != metadata["independent_pass"]:
                mismatches.append({"matrix": name, "issue": "summary hit verdict disagrees"})
            hits.append(entry)

    verified_retained = {entry["seed"] for entry in [best, *hits]}
    counts = {"final": len(finals), "best": 1, "hit": len(hits),
              "passed": sum(item["passed"] for item in [*finals, best, *hits])}
    return {"d": D, "n": N, "verifier": "verify_etf.verify_numeric", "numeric_gate": TOLERANCE,
            "summary_sha256": sha256(summary_bytes).hexdigest(),
            "final_stack": {"file": stack_file.name, "sha256": sha256(stack_bytes).hexdigest(), "shape": list(stack.shape)},
            "counts": counts, "norm_comparison": {"absolute_tolerance": NORM_ATOL, "comparisons": norm_comparisons,
                "max_absolute_discrepancies": metric_maxima, "mismatches": mismatches,
                "retained_seeds_without_saved_iterate": sorted(set(deferred_retained) - verified_retained)},
            "antiunitary_checks": symmetry, "best": best, "hits": hits, "finals": finals}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="*", type=Path, help="Directories containing completed Round 2 runs")
    args = parser.parse_args()
    results_root = Path(__file__).resolve().parents[1] / "results"
    directories = args.directories or [p.parent for p in results_root.glob("round2*/**/summary.json")]
    if not directories:
        parser.error("No completed Round 2 result directories found")
    status = 0
    for directory in sorted({p.resolve() for p in directories}):
        report = audit(directory)
        output = directory / "numeric_verification.json"
        # Stage on the same filesystem so replacing an existing report is atomic.
        with TemporaryDirectory(prefix="audit_report_", dir=directory) as tmp:
            temporary_output = Path(tmp) / output.name
            temporary_output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                                        encoding="utf-8", newline="\n")
            temporary_output.replace(output)
        mismatches = report["norm_comparison"]["mismatches"]
        print(f"{directory.name}: {report['counts']}, best={report['best']['metrics'][RESIDUAL]:.17g}, "
              f"mismatches={len(mismatches)}", flush=True)
        status |= bool(mismatches)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
