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

## Round 4

1. Anchor one line and average the moment functional over the remaining $S_8$ action and conjugation. Record the representation, degree, compressed sizes, and any additional validated equations.
2. Run the resulting SDP with guarded solver processes. Preserve primal moments and dual diagnostics. Investigate the full-label gauge-invariant Bargmann hierarchy in parallel.
3. Verify constraint provenance and PSD independently. A refutation requires an exact positive margin. If the relaxation is feasible, report which constraints and PSD blocks were tested and whether feasibility is exact or numerical.
4. Commit code, evidence, and progress. Verify the remote head.

## Round 5

1. Generate exact Gram and Schur rank minors and test every new equality that is evaluable by the old moment vector.
2. Select cubic monomial orbits and build the symmetry-compressed degree-six model. Solve both the fixed quartic extension and the free model under a monitored budget.
3. Independently validate any dual obstruction or feasible moment vector, including the equation provenance and the selected blocks.
4. Archive all decisions and compare the remote head after pushing.

## Round 6

1. Derive common null directions from degree-three polynomial identities and include their cross constraints. Compare exact compression sizes with Round 5.
2. Run guarded Clarabel and SCS on the reduced three- and four-point problems. Use the independent verifier on any stored witness or dual.
3. Extend the PSD monomial space to the five-point and six-point cubic shapes in that order, with the corresponding tail subgroups. Report measured or estimated resource requirements for stages that do not reach a solve.
4. Record exact versus numerical status, document the branch decision if applicable, and push verified artifacts.

## Round 7

1. Select candidate pivots from the saved row compression, solve for an affine chart over the rationals, and verify it against every face equation. Archive pivot and free coordinates, the source hash, the exact chart, and the PSD-only conic data.
2. Independently assess certificate reconstruction while running guarded Clarabel and SCS on the same coefficient data. Record unscaled primal and dual diagnostics and measured resource use.
3. Check any feasible point or refutation in exact arithmetic against the original moment model. If exact feasibility is established, assess the comparable five-point build. Archive and push with fresh tests and remote-head verification.

## Round 8

1. Generate five-point face equations by orbit representatives of the products of structural kernel polynomials and the monomial basis. Recover the same face without materializing the unreduced moment blocks.
2. Seed row reduction with the certified Round 7 chart. Select independent added equations, solve over the rationals, and verify every generated row. Reduce the packed PSD image and record the exact rank and budget.
3. Use a guarded blockwise margin solve if the resource bound permits it. Independently certify any positive primal or negative dual result. Archive the numerical result with the next-round decision.

## Round 9

1. Assemble and eliminate the support-six model under a single Windows Job and wall deadline. Save exact stencils and rank evidence. Launch the blockwise Newton phase only if the remaining resources suffice.
2. Draft the English manuscript and Lean algebra files in parallel. Integrate the observed probe outcome and audit every quantitative statement against its source artifact.
3. Replay the verification and compile the manuscript. Record Lean compilation status, then commit and verify remote main.
