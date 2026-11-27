# Agent 1 · 小尺寸复数 ETF 目标筛选

核查截至 **2026-11-27（北京时间）**。在已核查的公开原始文献和作者维护的数据中，以下三组参数仍可标为 **?**。项目中的问号表示尚未找到精确构造或一般不存在性证明。预印本提交与修订日期按 arXiv 所载日期记录。

## 排序与首攻

| 优先级 | $(d,N)$ | Naimark 补 | 精确 Welch 相干度 $\mu$ | Welch 相干度（约） | 作者维护榜单的已提交最好相干度（约） | 状态 |
| ---: | --- | --- | --- | ---: | ---: | ---: | :---: |
| **1 · 首攻** | $(4,9)$ | $(5,9)$ | $\sqrt{10}/8$ | 0.39528471 | 0.40185012 | ? |
| 2 | $(4,10)$ | $(6,10)$ | $\sqrt{6}/6$ | 0.40824829 | 0.41077812 | ? |
| 3 | $(4,11)$ | $(7,11)$ | $\sqrt{70}/20$ | 0.41833001 | 0.42514727 | ? |

三组都满足 $N>2d$。排序优先按 Gram 阶数，其次看可用于降维的对称结构。$(4,9)$ 可先考察三个长度为 3 的轨道与连续相位的分块循环结构。$(4,10)$ 可考察两个长度为 5 的轨道。$(4,11)$ 仍可考察带固定点的多轨道对称性，其 11 为素数限制了传递型构造。轨道分解作为搜索假设，其选取依据是计算规模和结构，并无经证实的存在概率。$(4,10)$ 的榜单值最接近相应的 Welch 下界。[原作者维护的 Game of Sloanes 榜单](https://github.com/gnikylime/GameofSloanes#leader-board)

## 原表定位与开放状态依据

- Fickus–Mixon，*Tables of the existence of equiangular tight frames*，2015-04-01 首次提交，2016-06-16 提交 v2。§6.2（PDF 印刷 pp. 9–10）说明复数 Table 2 只收录已知存在的 $N>2M$ 参数，论文中的 $M$ 对应本项目的 $d$。Table 2 首页（PDF 印刷 p. 16）的 $M=4$ 仅列 $N=13$（PG(2,3) 与 Cyclo4prime(1) 差集）与 $N=16$（SIC）。所以 $N=9,10,11$ 的状态来自未收录与后续核查，项目以 `?` 表示。$N=8$ 属于原文另列的 Table 3。维数 3 在 $N>2d$ 时只余 $N=7,8,9$，其中 $7,9$ 有构造，$8$ 及其 Naimark 补 $(5,8)$ 已排除。维数 2 在此范围内没有参数。[arXiv:1504.00253v2](https://arxiv.org/abs/1504.00253v2) · [PDF Table 2](https://arxiv.org/pdf/1504.00253v2#page=16) · [Szöllősi 的 2014-02-26 原始论文](https://arxiv.org/abs/1402.6429v1)
- 三组均满足复数 Gerzon 界及 Naimark 对偶界 $N\le \min\{d^2,(N-d)^2\}$。Fickus–Jasper–Mixon 于 2026-05-27 提交的 *The Singer–Zauner gap for equiangular tight frames*，Theorem 1，排除 $d^2-d+1<N<d^2$。在 $d=4$ 时排除 $N=14,15$，而 $N=9,10,11$ 与各自 Naimark 补均在该禁区之外。[arXiv:2605.28738v1，Theorem 1](https://arxiv.org/html/2605.28738v1)
- Jasper–King–Mixon 的 *Game of Sloanes* 原始论文于 2019-07-18 提交。其作者维护榜单在 2026-11-27 查阅时，三项目标依次由 `hlc`、`hlc`、`bmem` 提交对应密堆积，且均未标为 `etf` 或已证全局最优。三个 Naimark 补在榜单中也未获这两种标记。每个榜单值是由具体排布给出的最优相干度**上界**。要从密堆积排除 ETF，须证明最优相干度严格大于 Welch 下界。注意原表所列 $(5,11)$ 的 Naimark 补是 $(6,11)$，以及实 $(5,10)$ 的 Naimark 补仍在维数 5。[arXiv:1907.07848v1](https://arxiv.org/abs/1907.07848v1) · [2026-11-27 查阅的作者榜单](https://github.com/gnikylime/GameofSloanes#leader-board) · [Fickus–Mixon 原表 PDF](https://arxiv.org/pdf/1504.00253v2)
- Dhillon–Heath–Strohmer–Tropp 的 *Constructing Packings in Grassmannian Manifolds via Alternating Projection*（2007-09-04 提交预印本，2008 年刊出）在 Table 2 给出三组参数的数值密堆积。最小夹角依次比各自的 Rankin 上界低约 $0.51^\circ, 0.19^\circ, 0.63^\circ$。[arXiv:0709.0535](https://arxiv.org/abs/0709.0535) · [预印本 Table 2](https://arxiv.org/html/0709.0535)
- Goyeneche–Turek 的 *Equiangular tight frames and unistochastic matrices*（2016-07-15 首次提交，2017-02-07 提交 v2），§VI–VII 报告了小尺寸数值实验。该算法在定模、厄米及固定对角约束与 Schmidt 正交化之间迭代，在 $N\leq19$ 的比较中用 $10^3$ 个随机起点重现已知类，未报告三项目标的精确构造。有限随机起点未找到解只能作为历史数值基线。[arXiv:1607.04528v2，§VI–VII](https://arxiv.org/pdf/1607.04528v2#page=10) · [期刊 DOI](https://doi.org/10.1088/1751-8121/aa6e16)
- **Fickus 整除猜想。** Mixon 于 2015-07-08 记载该猜想，Goyeneche–Turek §VII 将其列为开放问题。若复数 ETF 存在，则 $d,N-d,N-1$ 三者中至少一个应整除其余两者之积。$(4,9)$ 有 $4\mid5\cdot8$，$(4,10)$ 有 $6\mid4\cdot9$，$(4,11)$ 对 $4,7,10$ 的三种检验均失败。检索中未发现该猜想的一般证明或反例。$(4,11)$ 因此仍记为 `?`，若找到精确构造，它会反驳整除猜想。[Mixon 的 SampTA 记录](https://dustingmixon.wordpress.com/2015/07/08/conjectures-from-sampta/) · [Goyeneche–Turek §VII](https://arxiv.org/pdf/1607.04528v2#page=12)

## 对搜索有约束力的条件

- **普通循环或阿贝尔平移不变 Gram。** Fourier 对角化把 $dG/N$ 写成特征标子集对应的正交投影。非平凡位移的 Gram 模长恒定，逆 Fourier 变换要求每个非零差值在该子集中出现 $d(d-1)/(N-1)$ 次。按目标排序得到 $3/2,4/3,6/5$，均非整数。这排除了含单一正则阿贝尔平移轨道且 Gram 严格平移不变的 Ansatz。[Fickus–Mixon §2、§4.1](https://arxiv.org/html/1504.00253v2)
- **签名矩阵的单位根相位。** 设 $S=(G-I)/\mu$，并记 $S^2=(N-1)I+bS$，其中 $b=(N/d-2)/\mu$。对任意 $i\ne j$，有 $b=(S^2)_{ij}\overline{S_{ij}}$。若某个对角酉切换后的 $S$ 的全部非对角元为单位根，该式使 $b$ 成为代数整数。三个目标的 $b^2$ 依次为 $2/5,3/2,45/14$，均为非整数有理数，与代数整数要求矛盾。这一推导也排除签名矩阵全部非对角元同为代数整数的情形。连续相位及含非代数整数相位的模型保留。[Goyeneche–Turek，命题 3](https://arxiv.org/pdf/1607.04528v2#page=5)
- **酉作用下的双传递直线系。** Iverson–Mixon 的 Theorem 1.3 分类了 $N\ge2d>2$ 时的双传递直线系，三项目标均不在列表中。对素数阶 $N=11$，甚至可排除传递的酉直线对称性。若酉直线对称的置换群传递，由轨道—稳定子定理和 Cauchy 定理，它含有一个 11-轮换。代表该置换的酉算子 $U$ 满足 $U^{11}$ 固定每条直线。直线两两非正交且张成全空间，故 $U^{11}$ 是标量。重标定 $U$ 后可选同一向量的 11 个幂次像作代表，Gram 成为严格循环矩阵，落入上一条排除的差集情形。[Iverson–Mixon，2024-02-22，Theorem 1.3](https://www.numdam.org/item/10.5802/alco.324.pdf)

- **代数解的存在性。** 若任一目标存在复数 ETF，Greaves–Iverson–Jasper–Mixon 的 Theorem 7.3 保证该参数至少有一个向量条目为代数数的 ETF。该定理未给出所需数域次数的上界。[arXiv:2012.12977v2，2021-07-14 修订版，Theorem 7.3](https://arxiv.org/pdf/2012.12977v2#page=27)

## 2024–2026 年交叉核查

| 原始来源 | 日期 | 对三项目标的适用范围 |
| --- | --- | --- |
| Iverson–Jasper–Mixon，*More on the optimal arrangement of 2d lines in C^d*，[arXiv:2410.17379v1](https://arxiv.org/abs/2410.17379v1) | 2024-10-22 | 构造与计算机辅助证明针对 $N=2d$。 |
| Fallon–Iverson，*On the optimal arrangement of 2d lines in C^d*，[期刊论文](https://academic.oup.com/imaiai/article/14/2/iaaf008/8116503) · [arXiv:2312.09975](https://arxiv.org/abs/2312.09975) | 2025-04-19 刊出，预印本 2023-12-15 | 双倍构造输出 $N=2d$。 |
| Fickus–Lake，*Doubly transitive equiangular tight frames that contain regular simplices*，[作者单位存档](https://scholar.afit.edu/facpub/2334/) | 2025-07-31 在线刊出 | 研究含正则单纯形的双传递构造。三组已由双传递分类排除于该构造类之外。 |
| Fickus–Jasper–Mixon，*The Singer–Zauner gap for equiangular tight frames*，[arXiv:2605.28738v1](https://arxiv.org/abs/2605.28738v1) | 2026-05-27 | 给出一般复数 ETF 禁区，三组及其对偶均未触及。 |
| Glazyrin，*New constructions of optimal arrangements of 2d lines in C^d*，[arXiv:2608.16116v1](https://arxiv.org/abs/2608.16116v1) | 2026-08-17 | 张量与幂构造针对 $N=2d$。 |
| King，*k-Homogeneous Equiangular Tight Frames*，[arXiv:2505.00160v3](https://arxiv.org/html/2505.00160v3) | 2025-04-30 首次提交，2026-09-28 更新 | Theorem 3.5 的高齐次性分类适用于受对称性约束的框架。 |
| Brinkley–Fickus–Jasper–Mixon，*Incoherent filter banks*，[arXiv:2610.05302v1](https://arxiv.org/abs/2610.05302v1) | 2026-10-04 | 主要定理刻画可交换交叉 Gram 的复紧融合框架及其 ETF 分量，未发现三组参数的构造或一般排除结果。 |
| Fallon–Iverson，*Conference signals: Applications, existence, and constructions*，[arXiv:2610.09169v1，§2.3](https://arxiv.org/html/2610.09169v1#S2.SS3) | 2026-10-06 | §2.3 在辛形式下构造 ETF。Remark 2.9 指出厄米内积下的循环 Gram ETF 由差集给出。 |
| Djaljapayan–Liao–Yu，*Real equiangular lines in dimensions 18–20 and sharp fixed-angle bounds*，[arXiv:2611.34855v1](https://arxiv.org/abs/2611.34855v1) | 2026-11-25 | 针对实直线及实固定夹角界。 |

检索覆盖 2024-01-01 至 2026-11-27 的公开可索引资料，并回查原表和相关早期定理。通过网页索引交叉检索 arXiv、期刊与作者单位网页，核对命中的原论文和作者维护榜单。搜索使用 `complex equiangular tight frames`、`unit gain graphs`、`complex projective packing`、`Welch bound`、`nonexistence`、`Singer–Zauner` 等词，组合三组 $(d,N)$、写法 $d\times N$、直接的向量数与维数描述、Naimark 补参数。证据限于所检索的公开记录，可能存在尚未索引的论文、未发表结果和未命中的同义表述。
