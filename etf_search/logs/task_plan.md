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
