# Support-six probe

Start from the exact round-eight artifacts. Run the bounded assembly, rank reduction, and barrier solve in one Windows Job. The supervisor stops the worker before 3.5 GiB or 30 minutes. The dense floating-point block files can be regenerated from the archived integer stencils and affine chart. The archive records their checksums and keeps the numerical primal/dual checkpoints.

```powershell
$prior = 'etf_search/results/round8_support5_lmi'
$probe = 'etf_search/results/round9_support6_probe'
$scratchProbe = Join-Path $env:TEMP ('etf-support6-' + [guid]::NewGuid())
.venv/Scripts/python.exe -B etf_search/src/run_support6_probe.py --prior $prior --output $scratchProbe
.venv/Scripts/python.exe -B etf_search/src/verify_support6_probe.py --input $probe --prior $prior --inherited-primal "$prior/barrier_rank75/exact_original_primal.json"
```

The initial run assembled all eight exact stencil files. The worker had imported the prior-only elimination module before the support-six change landed. Its attempt to decompress the plain JSON parent model raised `BadGzipFile`. The resumed worker used the saved assembly and deducted 187.06 seconds from the original wall allowance. Both workers had the same 3,520 MiB hard cap. A fresh replay starts with the corrected module and uses a single worker.

The exact five-point moment vector violates six of the 21 additional row stencils. The 7,048 cross rows add two independent pivots to the support-five equality system. The new chart retains 75 directions, all visible to the PSD blocks. The independent verifier certifies the lower ranks using nonzero minors modulo 65,537 and checks the affine substitution against the archived sparse data. Its optional `--audit-stencils` mode regenerates the polynomials and packed entries.

The bounded solve completed after 35 Newton steps. The resumed worker used 999.61 seconds and the two guarded workers used 1,186.67 seconds in total. Peak sampled RSS was 806,137,856 bytes. The numerical minimum eigenvalue across the unshifted scaled blocks was approximately 3.06e-8. The exact replay independently rebuilt every PSD stencil and certified all eight reduced blocks positive definite. The replay embeds the reconstructed witness, which is also stored separately in `exact_original_primal.json`.

```powershell
.venv/Scripts/python.exe -B etf_search/src/verify_support6_probe.py --input $probe --prior $prior --solution "$probe/barrier/solver_primal_dual.npz" --audit-stencils --include-certificates
```
