# Agent 2 · ETF(4,9) 的自由 $C_3$ 模型

取 $\mu=\sqrt{10}/8$、$\lambda=9/4$ 及 $\omega=e^{2\pi i/3}$。三个轨道各含三个顶点，自由作用为 $(a,x)\mapsto(a,x+1)$。

## Fourier 接口与两种投影

顶点的零基索引为 $i=3a+x$，$a,x\in\{0,1,2\}$。偏移量、频率和轨道内坐标均按模 $3$ 运算。设

$$
G_{3a+x,3b+y}=g_{ab}[(y-x)\bmod 3],\qquad
\widehat G_k[a,b]=\sum_{t=0}^{2}g_{ab}[t]\omega^{kt},\qquad
g_{ab}[t]=\frac13\sum_{k=0}^{2}\widehat G_k[a,b]\omega^{-kt}.
$$

每个 $3\times3$ 循环块的首行为 $(g_{ab}[0],g_{ab}[1],g_{ab}[2])$，次行为 $(g_{ab}[2],g_{ab}[0],g_{ab}[1])$。共同平移不变等价于此块结构。以 $3k+b$ 为 Fourier **列**索引，$U_{3a+x,3k+b}=\delta_{ab}\omega^{kx}/\sqrt3$，故 $U^*GU=\operatorname{diag}(\widehat G_0,\widehat G_1,\widehat G_2)$。若 `g.shape == (3,3,3)` 且最后一轴是 $t$，沿该轴的正变换为 `3*ifft`，逆变换为 `fft/3`。频块堆叠顺序为 $k,a,b$。

- **模长投影 $P_{\mathcal A}$。** 在 Hermitian 循环块子空间中，把 $g_{aa}[0]$ 置为 $1$，将每个独立的非对角复数径向投影至半径 $\mu$，再填入 $g_{ba}[t]=\overline{g_{ab}[-t]}$。零输入可任选相位。若输入在该子空间之外，先取厄米部，再对三次共同平移共轭取平均。
- **谱投影 $P_{\mathcal B}$。** Fourier 变换后，将频块取厄米部，每块按**代数值**从大到小取前 $r_k$ 个特征向量，再以 $\lambda$ 乘它们的正交投影，固定 $(r_0,r_1,r_2)=(2,1,1)$。频率间的非对角块置零，再逆变换。重根处任选相应维数的子空间。这是 Frobenius 度量下到固定块秩谱集合的精确最近投影。若 $Q$ 是待投影频块、$P$ 是秩 $r_k$ 的正交投影，则 $\|Q-\lambda P\|_F^2=\|Q\|_F^2+\lambda^2r_k-2\lambda\operatorname{tr}(QP)$，最大化末项中的迹即得所述谱截断。

独立相位共有 $12$ 个实变量。每个对角块有 $g_{aa}[1]=\mu e^{i\theta_a}$ 和 $g_{aa}[2]=\overline{g_{aa}[1]}$，共 $3$ 个。三个无序轨道对各有 $g_{ab}[t]=\mu e^{i\phi_{ab,t}}$（$a<b$），共 $9$ 个。按轨道作对角酉重相位 $G\mapsto D^*GD$，$D_{3a+x,3a+x}=e^{i\alpha_a}$。整体相位作用平凡，余下两个自由度可固定 $g_{01}[0]=g_{02}[0]=\mu$，留下 $10$ 个连续变量。在所有非对角元非零时，保持共同平移不变的顶点重相位也只能在此基础上额外乘一个离散共同字符 $\omega^{mx}$。若用去相位模长集合执行迭代，径向投影时直接把那两个代表元置为 $+\mu$，并填入其 Hermitian 对应元。

任意精确解中，$\widehat G_k/\lambda$ 均为正交投影。每个频块对角元至少为 $1-2\mu>0$，故三个块均非零。秩之和为 $\operatorname{tr}(G)/\lambda=4$，因此块秩恰为 $(2,1,1)$ 的循环移位。取 $D_{3a+x,3a+x}=\omega^{mx}$，$D^*GD$ 使 $g_{ab}[t]\mapsto\omega^{mt}g_{ab}[t]$，因而 $\widehat G'_k=\widehat G_{k+m}$。选取 $m$ 使秩二块移到 $k=0$ 即覆盖所有秩情形。

## 自由 $C_3$ 的严格路障

对三块 Hermitian 矩阵 $B=(B_0,B_1,B_2)$，定义

$$
M(B)=\sum_{k=0}^{2} B_{k+1}\odot\overline{B_k}
-\operatorname{Diag}\bigl((B_1+B_2)_{00},(B_1+B_2)_{11},(B_1+B_2)_{22}\bigr),
$$

其中 $\odot$ 为逐元素乘积，复共轭也逐元素取。若 $G$ 满足模长条件，令 $F_k=\widehat G_k/\lambda$。对任意 $a,b$，Fourier 正交性给出

$$
\sum_{k=0}^{2}F_{k+1}[a,b]\overline{F_k[a,b]}
=\frac{3}{\lambda^2}\sum_{t=0}^{2}|g_{ab}[t]|^2\omega^t.
$$

当 $a\ne b$，右侧为零。当 $a=b$，右侧为 $3(1-\mu^2)/\lambda^2=1/2$，且 $\sum_k F_k[a,a]=3/\lambda=4/3$。因此

$$
M(F)_{ab}=0\quad(a\ne b),\qquad
M(F)_{aa}=F_0[a,a]-\frac56\le\frac{1+2\mu}{\lambda}-\frac56=-\gamma,\qquad
\gamma=\frac{7-2\sqrt{10}}{18}>\frac1{30}.
$$

另一方面，若三个归一化谱块为秩 $(2,1,1)$ 的投影，则可写成 $P_0=I-uu^*$、$P_1=vv^*$、$P_2=ww^*$，其中 $u,v,w\in\mathbb C^3$ 均为单位向量。令 $p=v\odot\bar u$、$q=w\odot\bar v$、$r=u\odot\bar w$，展开定义可得

$$
M(P)=-pp^*+qq^*-rr^*.
$$

$P_0$ 中的单位阵在第一项和第三项产生的对角修正已由 $M$ 的对角减项抵消。取非零 $h\perp\operatorname{span}\{p,r\}$，则 $h^*M(P)h=|q^*h|^2\ge0$。精确 ETF 将给出 $P=F$，与 $M(F)\preceq-\gamma I$ 矛盾。因此三个自由轨道的 $C_3$ 模型无解。

该结论也覆盖通过逐列相位表示的自由 $C_3$ 对称。若单项酉矩阵 $T$ 的置换部分是三个 3-循环且 $T^*GT=G$，则 $T^3$ 为与 $G$ 对易的对角矩阵。非对角元全非零迫使 $T^3$ 为标量。缩放 $T$ 使 $T^3=I$ 后，每个循环的相位乘积为 $1$，可再用对角酉换基将 $T$ 化为共同平移。

## 精确模长迭代点的残差下界

同一证明给出可用于保留自由 $C_3$ 实验的解析对照。任何符合此模型且精确满足模长和对角条件的 $G$ 都有

$$
\|G^2-\lambda G\|_F\ge 10^{-2}.
$$

设反之残差 $\varepsilon<10^{-2}$。把每个频块的特征值 $s$ 舍入到较近的 $0$ 或 $\lambda$，得到组装矩阵 $H$。因 $\max(|s|,|s-\lambda|)\ge\lambda/2$，有 $\eta:=\|G-H\|_F\le2\varepsilon/\lambda$。于是 $|\operatorname{tr}(H)-9|\le3\eta<\lambda/2$，所以 $\operatorname{rank}(H)=4$。同时 $\eta<1-2\mu$，每个频块仍非零。用共同字符把秩二块移至频率零，并在该频率顺序定义 $F_k=\widehat G_k/\lambda$、$P_k=\widehat H_k/\lambda$。

记 $\delta=(\sum_k\|P_k-F_k\|_F^2)^{1/2}=\eta/\lambda\le2\varepsilon/\lambda^2$。$P_k$ 是正交投影，元素模长至多 $1$。$F_k$ 元素模长也至多 $1$，由对角元界 $(1+2\mu)/\lambda$ 与非对角元界 $3\mu/\lambda$ 得到。逐项对 $M$ 作差并取 Frobenius 范数，三个逐元素乘积合计至多 $2\sum_k\|P_k-F_k\|_F\le2\sqrt3\,\delta$，对角减项至多 $\sqrt2\,\delta$。因此

$$
\|M(P)-M(F)\|_2\le (2\sqrt3+\sqrt2)\delta\le5\delta<\frac1{50}<\gamma.
$$

这迫使 $M(P)$ 负定，与前节的非负方向矛盾。实验应在 **模长投影后的点**记录 $\|G^2-\lambda G\|_F$，以便和此界比较。

## 单循环与有限单位根模型

- **单循环 $C_9$。** 九循环取三次幂就是三个同步的 3-循环，已被上述路障排除。另可直接检验差集条件。循环谱选择 $D\subset\mathbb Z_9$ 中的四个频率。对 $t\ne0$，等角条件给出 $|\sum_{j\in D}e^{2\pi i jt/9}|^2=4^2\mu^2=5/2$。对其作逆变换，每个非零差值出现 $(4^2-5/2)/9=3/2$ 次，与次数必须为整数矛盾。
- **有限单位根。** 令 $S=(G-I)/\mu$，则 $S^2=8I+\beta S$，其中 $\beta=(\lambda-2)/\mu=2/\sqrt{10}$。对任意 $i\ne j$，$\beta=\sum_{\ell\ne i,j}S_{i\ell}S_{\ell j}\overline{S_{ij}}$。若非对角相位全为单位根，即便阶数各异，右边仍是代数整数。其平方 $\beta^2=2/5$ 是非整数有理数，产生矛盾。这个条件也覆盖经重相位后能化为单位根的矩阵，因为式中的三角相位乘积在重相位下不变。
