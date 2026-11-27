# Round 1 implementation plan

**Goal**: Document three open small complex ETF parameters and explore one constrained construction with independently checked results.

**Architecture**: Maintain separate literature, algebra, search, and verification paths. The search emits an unmodified Gram matrix and per-seed records. The verifier reads the matrix directly and implements the Gram identities independently.

**Tech stack**: Python, NumPy, SciPy, SymPy, unittest and Git.

| Phase | Owner | Acceptance |
| --- | --- | --- |
| Clone and check Git | Coordinator | `origin` matches the designated repository and the current branch is `main` |
| Literature | Agent 1 | Three checked targets and an identified primary target in `targets.md` |
| Independent verifier | Agent 4 | Positive and negative tests pass; initial Git commit precedes search implementation |
| Structured model | Agent 2 | Parameterization, gauge count, Fourier projection, and scope of algebraic obstructions documented |
| Search | Agent 3 | Fixed-seed AP and DRS runs, phase refinement, full Gram matrix and recorded residuals |
| Audit and archive | Agent 4 and coordinator | Independent raw-matrix check, finite-phase obstruction review, and remote commit confirmation |

Record research observations in `findings.md` and milestone evidence in `progress.md`.

All Round 1 phases executed on 2026-11-27. See [progress.md](progress.md) for evidence and the next model.

## Round 2

1. Agent 4 audits the prime-order proof and its complete list of permutation types.
2. After audit, run three configurations in parallel. Agent 3 runs 28 phases without symmetry constraints. Coordination runs the 16- and 15-phase antiunitary configurations with the same projection/refinement objectives. Agent 2 examines exact antiunitary restrictions and the best observed minima. Agent 1 checks published approaches relevant to choosing the next target.
3. Record every seed, distribution and ten smallest final residuals. Independently verify every final Gram and each provisional numerical pass. Attempt exact recovery for any tightness residual below $10^{-12}$.
4. Record a Round 3 decision and compare the local and remote commit after pushing.

## Round 3

1. Generalize the unrestricted search to `(4,10)`, keep the `(4,9)` calibration callable, and run 1,000 fixed seeds.
2. Implement moment constraints for the three Schur-complement models, bound solver execution by 900 seconds and 2 GiB per branch, and preserve observed statuses and any numerical dual candidates.
3. Run the numerical Gram verifier on the best `(4,10)` input. Check any SDP certificate independently for constraint provenance, sign and coefficient identity. Record the arithmetic guarantees of each artifact.
4. Verify the test suite, commit the experimental artifacts, and confirm that remote `main` matches the local commit.
