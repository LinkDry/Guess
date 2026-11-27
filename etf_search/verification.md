# Agent 4 · ETF(4,9) 独立验算

## 自由 $C_3$ 障碍

[结构模型](ansatz.md)中的 Fourier 约定、块秩、逐元素乘积恒等式和 $10^{-2}$ 残差下界均通过复核。设 $\mu=\sqrt{10}/8$，$\lambda=9/4$，$\omega=(-1+i\sqrt3)/2$。取 $F_k=\widehat G_k/\lambda$，用下式直接展开三个 Fourier 块的逐元素相关系数。

$$
\sum_{k=0}^2\omega^{(k+1)t}\overline{\omega^{ku}}
=3\omega^t\mathbf1_{t=u},\qquad
\sum_k F_{k+1}[a,b]\overline{F_k[a,b]}
=\frac{3}{\lambda^2}\sum_t |g_{ab}[t]|^2\omega^t.
$$

下标按模三理解。对角循环块的系数模长平方为 $(1,\mu^2,\mu^2)$，其余块为 $(\mu^2,\mu^2,\mu^2)$。故相关量分别为 $1/2$ 与 $0$。各对角位置的三个频值之和为 $4/3$。记 $M$ 为相关量矩阵减去 $\operatorname{Diag}(F_1+F_2)$ 后的矩阵，其中 $\operatorname{Diag}$ 取对角部分，则

$$
M(F)_{aa}=F_0[a,a]-5/6,\qquad
M(F)\preceq-\gamma I,\qquad \gamma=\frac{7-2\sqrt{10}}{18}.
$$

各频块的对角元至少为 $(1-2\mu)/\lambda>0$。精确投影块的秩之和为四，故共同字符重相位后依次为 $(2,1,1)$。对 $P_0=I-uu^*$、$P_1=vv^*$、$P_2=ww^*$，逐元素展开得到

$$
M(P)=-pp^*+qq^*-rr^*,\quad
p=v\odot\bar u,\quad q=w\odot\bar v,\quad r=u\odot\bar w.
$$

其中两个含 $I$ 的乘积各贡献 $\operatorname{Diag}(P_1)$ 和 $\operatorname{Diag}(P_2)$，恰被减项抵消。取在 $p,r$ 的共同正交补中的非零向量 $h$，有 $h^*M(P)h=|q^*h|^2\ge0$，与 $M(F)$ 负定矛盾。SymPy 以精确代数数复核了全部九个 Fourier 相关系数，又用独立的复符号展开验证了 $M(P)$ 的全部九个元素。

## 残差常数

对任意满足同一自由 $C_3$ 模型并精确满足模长和单位对角条件的 Hermitian $G$，设 $\varepsilon=\lVert G^2-\lambda G\rVert_F$。分块舍入到最近的 $0$ 或 $\lambda$，记舍入结果为 $H$，则 $\eta=\lVert G-H\rVert_F\le 2\varepsilon/\lambda$。这是由实特征值 $s$ 满足 $\max(|s|,|s-\lambda|)\ge\lambda/2$ 得出。若 $\varepsilon<\lambda^2/12$，则迹偏差 $3\eta<\lambda/2$ 强制舍入后的总秩为四。若同时 $\varepsilon<\lambda(1-2\mu)/2$，则每个频块仍非零，可将秩二块重相位到零频率。

令 $P_k=\widehat H_k/\lambda$，并记 $\delta=\eta/\lambda$。所有 $F_k$ 的元素模长均至多一，因为对角上界为 $(1+2\mu)/\lambda$，非对角上界为 $3\mu/\lambda$。所有 $P_k$ 的元素模长也至多一。对逐元素乘积的两因子分别作差，再用 Cauchy–Schwarz 估计三项总和与对角修正，得到

$$
\lVert M(P)-M(F)\rVert_2\le(2\sqrt3+\sqrt2)\delta,
\qquad \delta\le 2\varepsilon/\lambda^2.
$$

这给出稍强的精确下界

$$
\varepsilon\ge\varepsilon_*:=\frac{\lambda^2\gamma}{2(2\sqrt3+\sqrt2)}
\approx 0.01947074033172840.
$$

SymPy 确认了 $10^{-2}<\varepsilon_*<\min\{\lambda^2/12,\lambda(1-2\mu)/2\}$。因此若 $\varepsilon<\varepsilon_*$，舍入块的秩满足上述前提，且扰动严格小于 $\gamma$，迫使 $M(P)$ 负定。公开记录搜索残差时，该下界适用于模长投影后的自由 $C_3$ 迭代点。浮点输出与精确模长集合的距离应一并检查。

## 酉投影对称性的范围

若酉算子 $U$ 诱导置换 $\sigma$，写 $Uf_i=t_i f_{\sigma(i)}$。对应单项酉矩阵 $T$ 满足 $T^*GT=G$。置换为三个 3-循环时，$T^3$ 是与 $G$ 对易的对角矩阵，且所有 $G_{ij}$ 非零，因此 $T^3$ 为标量。整体重标定使 $T^3=I$。每个轨道内取 $D_{\sigma(i)}=t_iD_i$，三个相位乘积均为一，故可取对角酉 $D$ 使 $D^*TD$ 成为共同平移置换。这样排除的是全部自由酉投影 $C_3$ 作用。

[对称性证明](obstructions.md)中的素数阶排除覆盖全部置换循环型。酉算子若固定所有直线，则两两非正交使其共同位于一个特征空间，再由张成性知该算子为标量。酉投影自同构群因而忠实嵌入 $S_9$。归一化后可令任一素数阶元素的酉提升满足 $U^p=I$。

| 素数阶 $p$ | 循环型 | 独立复核 |
| --- | --- | --- |
| 2 | $2^s1^{9-2s}$，$1\le s\le4$ | 紧性给出 $\lambda\operatorname{tr} U=a+2\mu b$。固定直线的符号和 $a$ 是 $[-7,7]$ 内奇数。交换对贡献 $\pm2\mu$，由 $\mu$ 无理得到 $b=0$。$\lambda\operatorname{tr}U$ 的整数可能值只有 $-9,0,9$。 |
| 3 | $3^s1^{9-3s}$，$1\le s\le3$ | 任一三循环的轨道秩至少 $16/7$，故秩为三。循环子空间含三个不同特征值，全空间任一特征空间至多二维。有固定直线时至少三条，且都在同一特征空间，三线的 Welch 相干度至少 $1/2>\mu$。无固定直线的类型进入已排除的自由 $C_3$ 模型。 |
| 5 | $5^1 1^4$ | 单个轨道秩至少 $40/13>3$，故张成四维。任一特征空间内，该轨道的投影共线，所以四个特征空间各一维。四条固定直线两两非正交，故都须在同一特征空间，与简单谱矛盾。 |
| 7 | $7^1 1^2$ | 单个轨道秩至少 $112/31>3$。同一论证给出简单谱，两条固定直线非正交，产生矛盾。 |

对 $m$ 个轨道向量，秩估计由子 Gram 矩阵 $K$ 的精确迹恒等式 $\operatorname{tr}K=m$、$\operatorname{tr}K^2=m(1+(m-1)\mu^2)$ 及 $\operatorname{rank}K\operatorname{tr}K^2\ge(\operatorname{tr}K)^2$ 得出。对单个循环生成向量 $f$ 和谱投影 $E_\xi$，有 $E_\xi U^j f=\xi^jE_\xi f$，故轨道在每个特征空间内至多张成一维。仅 $2,3,5,7$ 可整除 $|S_9|$。由 Cauchy 定理，**假设存在复数 ETF$(4,9)$，其酉投影自同构群必为平凡群。** 一般存在性仍开放。

补充核查了反酉情形。若反酉算子 $A$ 置换九条直线，其平方为标量，诱导置换的阶数整除二。奇数条直线保证有固定直线，在该直线上代入可知 $A^2=I$。固定直线代表可取在实形式中，五条固定直线会令某个整数符号矩阵具有特征值 $-1/\mu$，其平方 $32/5$ 违反代数整数条件。故反酉搜索允许的置换类型分别为 $2^4 1$ 和 $2^3 1^3$。令 $s$ 为交换对数、$f$ 为固定点数，边轨道计数验证连续相位在去除配对重相位规范后为 $(36-s-\binom f2)/2$，分别为 16 和 15。

## 第一轮数值验算

对 `results/round1/best_gram.npy` 运行独立验证器，退出码为 `1`。原始文件的 SHA-256 为 `21c4f3f5d2ccc46dd08521bbcabf8c9ae3d2cd81e2efd13989b09ac995b6291f`。

| 条件 | 原始矩阵的误差 | 验收 |
| --- | ---: | --- |
| Hermitian | $0$ | 通过 |
| 单位对角线 | $0$ | 通过 |
| Welch 模长（最大误差） | $5.551115123125783\times10^{-17}$ | 通过 |
| 紧性（Frobenius 残差） | $0.7168843846433282$ | 未通过 |

将 `complex128` 的实部和虚部按其存储位解释为精确二进有理数，再用 SymPy 对原始矩阵独立计算。矩阵严格 Hermitian，对角线严格为一，并在置换 $(0\ 1\ 2)(3\ 4\ 5)(6\ 7\ 8)$ 的同时行列作用下严格不变。精确有理运算得到的残差开方约为 $0.7168843846433285719$，与浮点验证器相符。

记 $E=\max_{i\ne j}\left||G_{ij}|^2-5/32\right|$。精确计算给出 $E\approx4.994101781151379\times10^{-17}$。将非对角元径向调整到精确模长 $\mu$ 得到 $G_\star$，保留 Hermitian 对称和共同平移。由于 $\mu>1/3$，任一元素的调整量至多 $3E$。全矩阵有 72 个非对角元，故 $\lVert G-G_\star\rVert_F\le27E$。用 $\lVert G\rVert_F<5$、$\lVert G_\star\rVert_F=9/2$ 和 $G^2-G_\star^2=G(G-G_\star)+(G-G_\star)G_\star$，可将两者紧性残差的差值控制在 $(5+9/2+9/4)27E<1.6\times10^{-14}$。因此该点的精确模长版本同样高于理论下界 $0.0194707403317284\ldots$。

`numeric_verification.json` 中保存了原始文件哈希、CLI 的四条件判定、精确有理数形式的残差平方和径向调整的误差上界。此轮的数值值是已测候选的残差。自由 $C_3$ 模型的普适排除来自上面的解析证明。
