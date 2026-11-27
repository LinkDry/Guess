# Agent 2 · Round 2 反酉 ETF(4,9) 的代数化简

取 $\lambda=9/4$、$\mu=\sqrt{10}/8$。将反酉对称写成共轭，固定线取实代表，交换对取 $(z_j,\bar z_j)$。设固定点数 $f\in\{1,3\}$，交换对数 $s=(9-f)/2$。已知酉自同构限制及反酉正规化见 [obstructions.md](obstructions.md)。

## 两个分支共用的实 Gram 条件

将每对列酉变换为 $(\sqrt2\Re z_j,\sqrt2\Im z_j)$，以每对的实、虚列相邻排序。变换后的 Gram 为实对称矩阵

$$
R=\begin{pmatrix}K&C\\ C^T&D\end{pmatrix}.
$$

$K$ 是固定列的 Gram，$C$ 的第 $j$ 个双列块为 $\sqrt2[\Re B_{:j},\Im B_{:j}]$，其中 $B_{ij}=\langle f_i,z_j\rangle$。精确等角、单位范数及紧性可写为

$$
\begin{aligned}
&K_{ii}=1,\qquad K_{i\ell}\in\{\mu,-\mu\}\quad(i\ne\ell),\\
&|B_{ij}|=\mu,\qquad CC^T=2\Re(BB^*)=\lambda K-K^2,\\
&\operatorname{tr}D_{jj}=2,\qquad (D_{jj,11}-D_{jj,22})^2+4D_{jj,12}^2=4\mu^2,\\
&\det D_{jk}=0,\qquad \|D_{jk}\|_F^2=4\mu^2\quad(j<k).
\end{aligned}
$$

块条件来自 $z_j=(u_j+i v_j)/\sqrt2$，其中 $u_j,v_j$ 是变换后的实列。若 $D_{jk}=\begin{psmallmatrix}a&b\\c&d\end{psmallmatrix}$，对 $\langle z_j,z_k\rangle$ 和 $\langle z_j,\bar z_k\rangle$ 的模平方分别为 $((a+d)^2+(b-c)^2)/4$ 与 $((a-d)^2+(b+c)^2)/4$。两式之差为 $ad-bc$，和为 $\|D_{jk}\|_F^2/2$。

$K\succ0$，因为 $f\le3$ 且 $1-2\mu>0$。设 $D_0=C^TK^{-1}C$，将下块写成 $D=D_0+Z$。满足上面的 $CC^T$ 方程后，完整紧性条件等价于

$$
CZ=0,\qquad Z^2=\lambda Z,\qquad Z=Z^T,\qquad \operatorname{tr}Z=\lambda(4-f).
$$

证明来自分解 $R=\binom{K}{C^T}K^{-1}(K\ \ C)+\operatorname{diag}(0,Z)$。第一项平方等于自身乘 $\lambda$，秩为 $f$。紧性的右上块给出 $CZ=0$，于是两项的乘积为零，下块只剩 $Z^2=\lambda Z$。反向代入即得紧性。每个 $Z$ 自动半正定。另因 $\lambda I-K\succ0$，$C$ 的行满秩，故 $\dim\ker C=2s-f$。

## $2^3 1^3$ 的两个实三角符号

固定点的实符号切换使 $K=I+\sigma\mu(J-I)$，其中 $\sigma=+1$ 或 $-1$ 是固定三角的符号。用三对的独立相位将 $B_{1j}=\mu$。余下六个跨块相位可写成 $B_{2j}=\mu e^{i\theta_j}$、$B_{3j}=\mu e^{i\phi_j}$。令 $\beta=(\lambda-2)/\mu$，则 $CC^T$ 方程在对角约束外恰为

$$
\sum_{j=1}^3\cos\theta_j=\sum_{j=1}^3\cos\phi_j=\sum_{j=1}^3\cos(\theta_j-\phi_j)=\frac{\sigma\beta-1}{2}.
$$

因为 $Z$ 的秩为一，$Z=\lambda ww^T$，其中 $w\in\ker C\subset\mathbb R^6$、$w^Tw=1$。对指定的 $B$，$w$ 在二维球面上，且其符号无效。必要且充分的余下方程是将 $D=D_0+\lambda ww^T$ 代入三个双列对角块与三个双列交叉块的条件。可取六个单位圆上的实坐标及 $w\in\mathbb R^6$ 写成多项式系统。$Cw=0$ 为二次式，块的迹为二次式，模长及行列式等式至多四次。也可同时保留六阶实对称矩阵 $D,Z$，单独加入 $D=D_0+Z$。此时连接方程、$CZ=0$、幂等、迹和等角条件均至多二次。$K^{-1}$ 是每个分支的固定代数矩阵，可清掉其常系数分母。

## $2^4 1$ 的固定跨块

$K=[1]$，通过四对各自的相位选定 $B_{1j}=\mu$，所以 $C=\sqrt2\mu(1,0,1,0,1,0,1,0)$。由 $8\mu^2=\lambda-1$，$CC^T$ 方程自动成立。此时 $D_0=C^TC$，$Z/\lambda$ 是七维实空间 $\ker C$ 内的三维正交投影。块的迹、模长和行列式条件仍在 $D_0+Z$ 上施加。也可使用有理投影矩阵 $P_{\ker C}=I-C^TC/(CC^T)$ 在八维坐标直接写 $CZ=0$。

## 可复验的半定必要条件

用实变量 $x$ 收集 $C,D,Z$ 的独立元，并加入 $D=D_0+Z$。$f=1$ 的 $C$ 已固定，$f=3$ 也可只保留 $B$ 的实虚部作为 $C$ 的自由元。取包含二次项的一阶矩矩阵 $Y\succeq0$、$Y_{00}=1$，以 $Y_{0i}$ 代替 $x_i$、$Y_{ij}$ 代替 $x_ix_j$。对单位圆、$CC^T$、$CZ$、幂等、迹、块行列式和模长方程逐项线性化。任何 ETF 都给出 $Y=(1,x)^T(1,x)$。还可加入一次矩的 $0\preceq\mathbb E Z\preceq\lambda I$。$f=1$ 可加强到 $\mathbb E Z\preceq \lambda P_{\ker C}$。$f=3$ 可用矩阵不等式 $0\preceq Z\preceq \lambda I-D_0$，后者在提升后也是线性的。理由是 $0\preceq R\preceq\lambda I$ 的下主块给出 $D_0+Z\preceq\lambda I$。

若只含二次矩的松弛可行，可增加四次矩以及二次等式乘次数至多二的所有单项式。$f=3$ 的秩一形式可用 $w$ 代替 $Z$，或继续在 $Z$ 形式中显式加入全部二阶子式为零。求解器如报告不可行，验收应重建对偶恒等式并检验其系数和半正定性。系数可在相应的代数数域内精确核对。方法参考 [Lasserre, *Global Optimization with Polynomials and the Problem of Moments*](https://doi.org/10.1137/S1052623400366802)。

| 分支 | 保存矩阵的 $\|G^2-\lambda G\|_F$ | 精确状态 |
| --- | ---: | --- |
| $2^4 1$ | 0.7851319973351303 | 待定 |
| $2^3 1^3$，$\sigma=+1$ | 0.7750742931531713 | 待定 |
| $2^3 1^3$，$\sigma=-1$ | 0.7635247089153765 | 待定 |

上表直接从 [Round 2 保存矩阵](results/round2_antiunitary/) 重算。三个分支各有 128 个初始种子。两级矩松弛的可行性仍待求解。
