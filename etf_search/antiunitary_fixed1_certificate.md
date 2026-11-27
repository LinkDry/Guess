# ETF(4,9) 单固定点反酉分支的 SOS 证书

设 $ho=9/4$，$
u^2=5/32$。假设一个 ETF 有反酉置换 $2^4 1$。反酉算子可化为共轭。取固定线代表为单位向量 $e_0$，并将四组交换线取为 $(z_j,\overline z_j)$。给交换对施加共轭的相位，使 $\langle e_0,z_j\rangle=\nu$。在实形式中记 $u_j=\sqrt2\operatorname{Re}z_j$ 与 $v_j=\sqrt2\operatorname{Im}z_j$，于是 $u_j=a e_0+x_j$、$v_j=y_j$，其中 $a=\sqrt2\nu$，$x_j,y_j\in e_0^\perp\cong\mathbb R^3$。令 $X=[x_j]$ 和 $Y=[y_j]$。

成对的酉换基保留紧性。将实框架的紧性限制到 $e_0^\perp$ 得 $T=XX^T+YY^T-\rho I_3=0$。根据 Schur 补实 Gram 模型，$\det D_{jk}=0$（$j\ne k$），而 $\det D_{jj}=1-\nu^2$。这些小行列式恰为外积向量 $u_j\wedge v_j$ 的相互内积。因为 $\Lambda^2 e_0^\perp$ 与 $e_0\wedge e_0^\perp$ 正交，在 $e_0^\perp$ 的实坐标中取 $W=[x_j\times y_j]$，得到

$$
E=W^TW+a^2Y^TY-bI_4=0,\qquad b=1-\nu^2=27/32.
$$

由 $T=0$ 得 $YY^T\preceq\rho I_3$，所以 $Y^TY\preceq\rho I_4$。此时 $W^TW\succeq(b-a^2\rho)I_4=\tfrac9{64}I_4$。但 $W$ 只有三行，需要一个非零核向量。

## 有理数恒等式

加辅助变量 $h\in\mathbb R^4$，满足 $Wh=0$ 与 $h^Th=1$。这些变量可由 $W$ 的任意单位核向量给出。令 $\Delta=b-a^2\rho=9/64$，则以下为逐系数的多项式恒等式。

$$
-1=\frac{a^2}{\rho\Delta}
\bigl(\|\rho h-Y^TYh\|_2^2+\|X^TYh\|_2^2\bigr)
+(h^Th-1)+\frac{h^TEh-(Wh)^T(Wh)-(a^2/\rho)(Yh)^TT(Yh)}{\Delta}.
$$

将 $E,T,Wh,h^Th-1$ 置零，右侧为八个实多项式平方的正倍数。证书采用 Schur 补系统的实坐标提升，全部系数为有理数。每个平方储存为一个单项式基上的秩一 Gram 块。严格正的权重及 Gram 块的精确半正定性允许保留其结构性的零特征值。

[Schur 补实 Gram 模型](antiunitary_analysis.md) · [证书和约束](results/round3_sdp/identity_fixed1/)
