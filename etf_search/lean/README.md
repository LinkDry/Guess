# Lean 4 algebra for ETF(4,9)

The source draft has two targets. `ETF49.Signature` expands and inverts the signature change of variables over complex matrices, specializes the tightness equation to the ETF parameters, and derives Gram tightness from frame tightness. `ETF49.FixedOne` checks the finite coordinate polynomial identity over a characteristic-zero field and applies real nonnegativity to refute the lifted constraints with a normalized kernel vector.

The two interfaces match the archival proofs. The signature theorem starts from matrix tightness. The real refutation starts from the tightness, bivector Gram, kernel, and unit-sphere equations. Geometric derivation of those equations from a projective antiunitary, existence of the normalized kernel vector, the other symmetry branches, and SDP witnesses are outside this source draft. The Lean SOS proof uses a direct identity with the same eight squares as the archived JSON. The existing Python verifier separately checks the JSON coefficients, Gram factors, and provenance against its independent coordinate model.

## Build

Windows `PATH`, the standard Elan location, and the two Ubuntu WSL distributions were checked on November 28, 2026. No installed Lean, Lake, or Elan command or Lean executable was found. This draft has no local Lean build result.

With Elan installed, from this directory run the following commands. `lake update` fetches the pinned mathlib release and `lake exe cache get` fetches its compiled files.

```sh
lake update
lake exe cache get
lake build ETF49
lake env lean Check.lean
```

The final command instantiates the identity over the rationals and prints the declaration axiom dependencies. Verify that `sorryAx` is absent.

## Archive replay

From the repository root on the original Windows workspace, the following command replays the exact JSON verification separately from Lean.

```powershell
.venv/Scripts/python.exe -B etf_search/src/verify_sdp_cert.py etf_search/results/round3_sdp/identity_fixed1/certificate.json --trusted-constraints etf_search/results/round3_sdp/identity_fixed1/constraints.json --fixed1
```
