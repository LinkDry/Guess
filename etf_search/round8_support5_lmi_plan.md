# Round 8 · 五点六阶矩松弛

## 分批构造与精确消元

设 \(V_5\) 是总次数至多二的全部单项式加上支撑至多五点的三次单项式空间。对每个结构核生成式 \(hm\in V_5\)，其中 \(h\) 是基础 ETF 方程，施加 \(L(hmv)=0\)（\(v\in V_5\)）。这些等式对每个 ETF 的点值矩成立。先固定 \(h\) 的一个或两个标记顶点，再对 \(m\) 按尾部置换取轨道代表。随后固定 \(m\) 涉及的全部顶点，对 \(v\) 按剩余置换取轨道代表。由于线性泛函对 \(S_8\) 与共轭不变，重标号只会改变整体符号。共轭奇的乘积自动为零。五点空间包含四点空间，且使用相同的原始秩子式，因此可直接沿用 Round 7 的四点仿射图表。生成 66,581 个乘积，归并为 7,027 条不同的非零方程。

固定前五点后的子群为 \(S_3\)。在其固定子空间上以两个换位与共轭作字符分解。已有的五点轨道固定子空间论证说明这八块足以检验原矩阵的 PSD 性。核向量由 \(hm\) 投影得到。交叉等式使每个簇中的核方向消失，因此每块仅需构造主坐标补空间的矩映射。实现不生成未约简的 4,565,353 个 PSD 打包行。

以 Round 7 的 2,716 个矩坐标和 370 个自由坐标为起点，追加三个轨道坐标。浮点 QR 仅选择新等式与枢轴候选，随后用有理 Dixon 解 296 阶可逆子系统。把最终仿射图表代回全部新增等式，用整数点积或带绝对界的 CRT 验证。最终共有 2,719 个矩坐标，线性等式秩为 2,642（含归一化），自由矩坐标为 77。

## PSD 像秩与内存

PSD 线性映射中的每行只含同次数奇偶性的矩。一行共有的 \(\beta=\sqrt{10}/5\) 因子可去掉，故可在有理数上认证 PSD 方向像秩。对八块分批选行，抽出可逆的 75 阶子式。剩余两个坐标方向均在 PSD 映射的精确核中，逐块验证全部 357,260 个打包行后得到 **\(r_5=75\)**。四、五点的像秩都在各自消元后的等式系统中计算。五点新增的等式能够缩小旧模型的可移动方向。

| 项目 | 四点 | 五点 |
| --- | ---: | ---: |
| 有效 PSD 像秩 | 97 | 75 |
| 最大 PSD 块 | 128 | 373 |
| 打包行 | 36,243 | 357,260 |
| 五点八块 | | 373、342、284、302、284、302、239、235 |

模型构造实测约 28.5 秒，峰值采样 RSS 约 861 MiB。精确消元、像秩及八份浮点块导出约 60.9 秒，峰值采样 RSS 约 395 MiB。按最大块四份全对称双精度工作区、紧凑 Hessian 与额外 256 MiB 估算，未优化的逐块方案需约 579 MiB。实际实现将系数放在磁盘映射中，在块内再按方向分批，累加 \(76\times76\) 的 Hessian。两步构建使用 4,000 MiB 的 Windows Job 提交内存硬限额。实际求解限额为 3,584 MiB，墙钟上限 1,800 秒。

## 复现

```powershell
$round8 = 'etf_search/results/round8_support5_lmi'
$round6Face = 'etf_search/results/round6_facial_reduction/solver_support4_clarabel_streamed/solver_face_model.json.gz'
$round7Chart = 'etf_search/results/round7_support4_lmi/affine_chart.json.gz'
.venv/Scripts/python.exe -B etf_search/src/build_support5_lmi.py --face $round6Face --output $round8
.venv/Scripts/python.exe -B etf_search/src/build_support5_lmi.py --phase reduce --face $round6Face --chart $round7Chart --output $round8
.venv/Scripts/python.exe -B etf_search/src/solve_support5_barrier.py --input "$round8/pencil" --output "$round8/barrier_rank75" --memory-mib 3584 --seconds 1800
```

## 共同谱裕量与第二次面约简

逐块 Newton 法用了 83 步，受限进程用时 365.2 秒，峰值采样 RSS 为 196.4 MiB。最终浮点障碍下界约为 \(-4.10\times10^{-12}\)，对偶目标约为 \(+2.32\times10^{-10}\)。未平移八块在双精度误差范围内出现 41 个近零方向。

设 \(D_k\) 为按基多项式次数奇偶性取 \(1\) 或 \(\beta^{-1}\) 的对角矩阵。\(2D_k M_k(y)D_k\) 在有理矩坐标中是有理仿射矩阵。由舍入点的非枢轴主矩阵解出候选核向量，再在常数及全部 77 个自由列上逐项验证，得到公共核维数 \((9,6,4,6,4,6,3,3)\)。每个核基对枢轴坐标的限制是单位矩阵，故原八块的 PSD 性等价于主坐标补空间上的 PSD 性。补块阶数为 \((364,336,280,296,280,296,236,232)\)。原八块含公共核，其共同谱裕量至多为零。同一个有理矩见证在主补空间上的正定性给出可行的零裕量点。

```powershell
$priorAudited = 'etf_search/results/round6_facial_reduction/solver_support4_clarabel'
$point = "$round8/barrier_rank75/solver_primal_dual.npz"
$witness = "$round8/barrier_rank75/exact_original_primal.json"
.venv/Scripts/python.exe -B etf_search/src/discover_support5_common_kernel.py --root $round8 --solution $point --output "$round8/common_kernel.json"
.venv/Scripts/python.exe -B etf_search/src/export_support5_primal.py --root $round8 --solution $point --output $witness
.venv/Scripts/python.exe -B etf_search/src/verify_support5_rank.py --input $round8 --prior-face "$priorAudited/solver_face_model.json.gz" --prior-chart $round7Chart --common-kernel "$round8/common_kernel.json" --primal $witness
```
