# Round 7 · 四点六阶矩的零等式 LMI

## 精确消元

设 $\beta=\sqrt{10}/5$，令原矩坐标 $x_i=\beta^{\deg(m_i)\bmod 2}z_i$。每条齐次等式乘上该替换后，其非零项共同落在 $\mathbb Q$ 或 $\beta\mathbb Q$。去掉公共因子并清分母，得到整数等式 $Ez=e_0$。归一化行 $z_0=1$ 位于第 0 行。此变换在实数上可逆。

复用 Round 6 保存的 2,346 行。列主元 QR 仅用于挑选候选列。FLINT Dixon 有理求解在对应的可逆方阵上求出常数列及全部 370 个自由列，再用整数点积核对 $E[X_0,N]=[e_0,0]$ 的全部 202,138 行。自由行是单位阵。可逆方阵给出秩下界，370 个独立核向量给出秩上界，因此秩恰为 2,346。精确图表的每行存同一分母及非零分子，原矩模型 SHA-256 一同归档。

把 $x_i=\beta^{\deg(m_i)\bmod2}(X_0+Ny)_i$ 代入八块，得到 $M_k(y)=F_{k,0}+\sum_i y_i F_{k,i}$。八块阶数为 **128、99、86、98、86、98、77、75**。基础数据以三角向量化保存，非对角系数乘 $\sqrt2$，因此向量内积对应矩阵迹内积。SCS 与 Clarabel 的排列转换在求解器入口执行。可选的块基向量归一化是正对角合同变换。

## 精确限制到 97 个 PSD 可见方向

全部 370 个自由方向中，只需 97 个即可张成整个 LMI 方向像。将未打包的有理 PSD 映射按 $1,\beta$ 分成两组后，以 97 行上的可逆有理方阵求出被删除列的线性组合。逐列对全部 72,486 条分离的系数行做精确整数核对。数值探针识别 271 个零列与其余 99 列中的两个依赖关系。最终证书逐列保存全部删除关系，包括零列。将被删除的自由坐标置零并调整保留坐标可生成同一个 PSD 矩阵，故在任意实数坐标下保留可行性。对于精确原始点，将受限坐标放回保留列，其余置零后用原图表提升。

| 规格 | 完整仿射空间 | 精确限制后的 LMI |
|---|---:|---:|
| 实自由坐标 | 370 | 97 |
| 求解器零锥行 | 0 | 0 |
| PSD 打包行 | 36,243 | 36,243 |
| 浮点 CSC 非零项 | 3,290,649 | 3,226,358 |
| 浮点 CSC 密度 | 24.54% | 91.77% |

完整 CSC 的数值和索引占约 37.7 MiB，受限 CSC 约 36.9 MiB。Clarabel 的内点法主要开销预计仍在稠密 PSD 锥缩放和稀疏 KKT 因子。若每块的打包缩放贡献一个稠密三角，八块共产生 92,447,676 个位置。按 8 字节数值和 4 字节索引，单份约 1.03 GiB。此数不含分解、工作区、Rust 与 Python 复制。实际峰值以 4 GiB 受限进程的统计为准。SCS 使用同一 LMI 数据和 `eps_abs=eps_rel=1e-9`，目标迭代上限 20,000。

Dixon 耗时 13.54 秒，全等式核对耗时 20.71 秒，完整构建耗时 44.54 秒。数据和行参数见 [Round 7 归档](results/round7_support4_lmi/)。四点模型继续沿用 [Round 6 面约简](level3_facial_reduction_plan.md) 的原多项式及必要性证明。

## 紧凑内点计算与精确重建

在八个块上用正对角矩阵 $D_k$ 归一化，并最大化共同谱裕量 $t$，约束为 $D_k M_k(y)D_k-tI\succeq0$。取严格足够低的 $t$ 可启动对数行列式障碍法。每步从 Cholesky 因子累加梯度和 Hessian，仅分解 $98\times98$ 的 Newton 矩阵。障碍参数按十倍递减，从 $10^{-2}$ 到 $10^{-9}$。当前浮点候选的 $t=1.5993364\times10^{-9}$，耗时约 9.1 秒。精确见证按 $10^{14}$ 的分母舍入保留自由坐标后代入精确仿射图表，得到 $\mathbb Q(\sqrt{10})$ 上的矩。独立验收以精确面约简等价性把约简块的正定结果转回原始四点模型。验收器直接读取归档的精确矩候选，重新检查全部面等式，并对每个块用有理合同变换验证严格对角占优。

```powershell
$archive = 'etf_search/results/round7_support4_lmi'
$face = 'etf_search/results/round6_facial_reduction/solver_support4_clarabel_streamed'
.venv/Scripts/python.exe -B etf_search/src/build_level3_lmi_4_9.py --source $face --output $archive
.venv/Scripts/python.exe -B etf_search/src/build_level3_lmi_4_9.py --source $face --output $archive --restrict-only
.venv/Scripts/python.exe -B etf_search/src/solve_level3_lmi_barrier.py --input "$archive/lmi_conic_data_rank97.npz" --output "$archive/barrier_rank97" --chart "$archive/affine_chart.json.gz"
.venv/Scripts/python.exe -B etf_search/src/verify_sdp_lmi.py --model "$face/solver_face_model.json.gz" --primal-certificate "$archive/barrier_rank97/exact_original_primal.json"
```

## 五点预评估

五点面约简的八块阶数为 **373、342、284、302、284、302、239、235**，有 **357,260** 个 PSD 打包行。在同样的稠密缩放策略下，扁平三角的单份数值数组约 **64.7 GiB**。此估计使用的同一公式恰好给出四点 Clarabel 报错中的 739,581,408 字节。五点消元应分批生成并约去结构核的交叉等式，再按有理或模素数行基选择作零空间消元。交叉等式去重前的候选数上界为 6,582,480。对于紧凑障碍法，每个可见方向的打包系数占 2.73 MiB，每个可见方向的全对称块稠密模板占 5.43 MiB。下一步先测五点像秩和单块模板内存，再在 4 GiB 内分块累计 Hessian。五点的 LMI 像秩与可行性仍待测。[原始规模数据](results/round7_support4_lmi/support5_resource_preflight.json)
