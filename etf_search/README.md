# 复数 ETF 存在性搜索

首轮研究参数为 $(d,N)=(4,9)$。文献筛选覆盖 $(4,9)$、$(4,10)$ 和 $(4,11)$。自由 $C_3$ 模型的精确排除与数值基线分别记录在 [模型推导](ansatz.md) 和 [首轮结果](results/round1/summary.json)。[对称性证明](obstructions.md) 覆盖任意非平凡酉投影自同构，[独立验算](verification.md) 给出该证明的复核及数值验证。

## 复现首轮

从仓库根目录运行。实测环境使用 Python 3.11.9、NumPy 2.4.6、SciPy 1.17.1、SymPy 1.14.0。SymPy 依赖 mpmath，亦用于达到数值门槛后的 PSLQ 识别。

```powershell
python -m pip install numpy==2.4.6 scipy==1.17.1 sympy==1.14.0
python -B -m unittest discover -s etf_search/tests -v
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -B etf_search/src/search_etf.py --seeds 48 --seed-start 0 --steps 1800 --relaxation 1.0 --refine-maxiter 750 --output-dir etf_search/results/reproduction
python -B etf_search/src/verify_etf.py etf_search/results/reproduction/best_gram.npy --d 4 --n 9
```

该实验的最佳近似矩阵应由独立验收器返回退出码 `1`，因为紧性残差超过 $10^{-12}$。不同数值库可以在同一残差平台上选择不同的最佳种子。可复验的精确 Gram 输入采用 `{"format":"etf-gram-v1","gram":[...]}`。矩阵元素使用整数、四则运算、整数幂、`I`、`sqrt(q)` 或 `zeta(k)`。传入 `--symbolic` 可用 SymPy 精确验算。

[文献与目标](targets.md) · [实验日志](logs/progress.md)

## 复现第二轮

从仓库根目录运行。路线 B 的三个配置可分别运行。

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -B etf_search/src/search_unconstrained.py --seeds 1000 --seed-start 0 --steps 1800 --relaxation 1.0 --refine-maxiter 750 --workers 4 --batch-size 25 --output-dir etf_search/results/round2_unconstrained
python -B etf_search/src/search_antiunitary.py --fixed 1 --seeds 128 --steps 1800 --output-dir etf_search/results/round2_antiunitary/fixed1
python -B etf_search/src/search_antiunitary.py --fixed 3 --sign 1 --seeds 128 --steps 1800 --output-dir etf_search/results/round2_antiunitary/fixed3_positive
python -B etf_search/src/search_antiunitary.py --fixed 3 --sign -1 --seeds 128 --steps 1800 --output-dir etf_search/results/round2_antiunitary/fixed3_negative
python -B etf_search/src/audit_round2.py
python -B etf_search/src/analyze_round2.py --output etf_search/results/round2_unconstrained/plateau_analysis.json
```

[第一轮证明审计](audit_round1.md) · [反酉模型的多项式化简](antiunitary_analysis.md) · [第三轮决策](logs/progress.md#agent-2--round-3-决策)

## 复现第三轮

```powershell
python -B etf_search/src/search_unconstrained.py --d 4 --n 10 --seeds 1000 --seed-start 0 --steps 1800 --relaxation 1.0 --refine-maxiter 750 --workers 4 --batch-size 25 --output-dir etf_search/results/round3_4_10
python -B etf_search/src/verify_etf.py etf_search/results/round3_4_10/best_gram.npy --d 4 --n 10
python -B etf_search/src/generate_single_fixed_sos.py etf_search/results/round3_sdp/identity_fixed1
python -B etf_search/src/verify_sdp_cert.py etf_search/results/round3_sdp/identity_fixed1/certificate.json --fixed1
```

Gram 验收器应返回退出码 `1`。SOS 验收器应返回 `0`，并将结论限定在 ETF$(4,9)$ 的单固定点反酉分支。证书验证只需 Python 与 mpmath（数值模式）或标准库（精确模式）。生成证书需 SymPy。SDP 实验的依赖和命令见 [实验记录](results/round3_sdp/schur/search_sdp_plan.md)。

复核三固定点的两个对偶证书。验收会独立重建被证书使用的 Schur 方程，并使用整数与有理区间计算。

```powershell
$results = 'etf_search/results/round3_sdp/schur'
foreach ($branch in @('fixed3_positive_dense', 'fixed3_negative_dense')) {
    $dir = Join-Path $results $branch
    python -B etf_search/src/verify_sdp_cert.py "$dir/certificate.json" --model "$dir/model.json"
}
```

## 复现第四轮

从仓库根目录运行。求解器环境沿用第三轮的 SCS 3.3.2、NumPy、SciPy、SymPy 与 psutil。

```powershell
$src = 'etf_search/src'
$res = 'etf_search/results/round4_general_4_9'
.venv/Scripts/python.exe -B "$src/solve_general_4_9.py" --output "$res/anchored_s8_qdldl" --seconds 1800 --memory-mib 2048 --eps 1e-8 --verbose
python -B "$src/reanchor_general_4_9.py" "$res/anchored_s8_qdldl/anchored_model.json" "$res/reanchor_equalities.json"
.venv/Scripts/python.exe -B "$src/solve_general_4_9.py" --output "$res/anchored_s8_reanchored" --extra-equalities "$res/reanchor_equalities.json" --seconds 1800 --memory-mib 2048 --eps 1e-8 --verbose
foreach ($mode in @('anchored_s8_qdldl', 'anchored_s8_reanchored')) {
    $dir = Join-Path $res $mode
    python -B "$src/verify_sdp_cert.py" "$dir/anchored_exact_primal_candidate.json" --model "$dir/anchored_model.json"
}
python -B "$src/reanchor_general_4_9.py" "$res/anchored_s8_qdldl/anchored_model.json" "$res/reanchor_equalities.json" --rank-output "$res/reanchor_exact_rank.json" --candidate "$res/anchored_s8_qdldl/anchored_exact_primal_candidate.json" --candidate "$res/anchored_s8_reanchored/anchored_exact_primal_candidate.json"
python -B "$src/bargmann_4_9.py" --output "$res/bargmann_exact_moments.json"
```

[第四轮模型推导](general_4_9_sdp_plan.md) · [第四轮日志](logs/progress.md)

## 复现第五轮

从仓库根目录运行。两种阶段的三次基分别使用至多三个或四个顶点。求解命令为每个子进程设置内存和时间上限。归档的 `solver_guard.json` 记录完整命令和实测资源。

```powershell
$src = 'etf_search/src'
$res = 'etf_search/results/round5_level3_4_9'
$r4 = 'etf_search/results/round4_general_4_9/anchored_s8_qdldl'
python -B "$src/probe_level3_quartic.py" "$r4/anchored_model.json" "$r4/anchored_exact_primal_candidate.json" "$res/quartic_minor_probe.json"
.venv/Scripts/python.exe -B "$src/solve_level3_4_9.py" --minors --builder-options '{"stage":"support3"}' --output "$res/reproduction_support3" --seconds 450 --memory-mib 3072 --eps 1e-8 --verbose
.venv/Scripts/python.exe -B "$src/solve_level3_4_9.py" --minors --builder-options '{"stage":"support4"}' --output "$res/reproduction_support4" --seconds 1790 --memory-mib 4096 --eps 1e-8 --verbose
.venv/Scripts/python.exe -B "$src/recover_level3_primal.py" "$res/solver_free_support3_scaled" --model "$res/solver_free_support3_scs/solver_model.json"
python -B "$src/verify_sdp_cert.py" "$res/solver_free_support3_scaled/exact_primal_candidate.json" --model "$res/solver_free_support3_scs/solver_model.json"
```

固定旧矩的探针额外传入 `--fixed-candidate` 和 `--fixed-model`，指向第四轮的 `anchored_exact_primal_candidate.json` 和 `anchored_model.json`。精确线性重建使用 `python-flint` 0.9.0 加速。精确验收为测试对象重新构造子式、乘子及矩块。[第五轮模型推导](level3_4_9_sdp_plan.md)

## 复现第六轮

```powershell
$src = 'etf_search/src'
$r5 = 'etf_search/results/round5_level3_4_9'
$r6 = 'etf_search/results/round6_facial_reduction'
.venv/Scripts/python.exe -B "$src/solve_level3_face_4_9.py" --input "$r5/solver_free_support3_scs" --output "$r6/reproduction_support3" --backend scs --seconds 400 --memory-mib 1400 --iterations 12000
.venv/Scripts/python.exe -B "$src/solve_level3_face_4_9.py" --reconstruct "$r6/reproduction_support3" --output "$r6/reproduction_support3_exact" --seconds 400 --memory-mib 1400
python -B "$src/verify_sdp_cert.py" "$r6/reproduction_support3_exact/solver_exact_primal_candidate.json" --model "$r5/solver_free_support3_scs/solver_model.json"
python -B "$src/verify_sdp_cert.py" "$r6/solver_support4_clarabel/solver_face_model.json.gz" --audit-face --model "$r5/solver_free_support4_minors/solver_model.json"
```

较大的模型使用无损 gzip 归档。`compressed_manifests.json` 记录原始字节数与未压缩数据的 SHA-256。验收入口支持直接读 gzip，重建入口可自动从同一目录载入 `solver_face_model.json.gz`。完整参数与实测资源见每次运行的 `solver_guard.json`。[第六轮计划与块尺寸](/E:/Research/Guess/etf_search/level3_facial_reduction_plan.md)
