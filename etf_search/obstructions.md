# Exact restrictions on possible complex ETF(4,9) symmetries

**Coordination, for Agent 4 review.** Let $G$ be a putative Gram matrix and $f_1,\ldots,f_9$ its unit frame vectors. Write $\mu=\sqrt{5/32}$ and $\alpha=9/4$. A projective automorphism is a permutation of the nine lines induced by a unitary on $\mathbb C^4$. The lines span and their pairwise inner products are nonzero, so a unitary fixing all of them is scalar. Indeed, it has each $f_j$ as an eigenvector and eigenvectors for distinct eigenvalues are orthogonal. The induced permutation group is therefore a faithful realization of the projective automorphism group.

## No projective involutions

Let $U$ induce a nontrivial involution and normalize the scalar so that $U^2=I$. Then $U$ is Hermitian and $\operatorname{tr} U=2r-4$ for an integer $r\in[0,4]$. The tight frame identity gives

$$\alpha(2r-4)=\sum_{j=1}^9 f_j^*U f_j=a+2\mu b.$$

Here $a$ is the sum of the signs on fixed lines, so it is odd and $|a|\le7$. Each swapped pair contributes either $2\mu$ or $-2\mu$, since both contributions are real conjugates of modulus $\mu$. Thus $b$ is an integer, and irrationality of $\mu$ forces $b=0$. Among the values of $\alpha(2r-4)$ the only integers are $-9,0,9$. None can equal $a$.

## Order-three projective symmetries act freely

Let $U$ induce a nontrivial permutation of order three and normalize it to $U^3=I$. For a line in a moved triplet, $f,Uf,U^2f$ have a positive definite $3\times 3$ Gram matrix. The lower bound on its least eigenvalue is $1-2\mu>0$ by diagonal dominance. These vectors thus span a copy of the three-dimensional cyclic shift representation. All three eigenvalues occur and each eigenspace in the full four-dimensional space has dimension at most two.

If the permutation has a fixed line, it has at least three fixed lines, since the degree is nine. Their representatives are eigenvectors and are pairwise nonorthogonal, so they lie in the same eigenspace. Three unit vectors in dimension at most two have coherence at least $1/2$ by the Welch bound. This contradicts $\mu<1/2$.

## No projective elements of order five or seven

If an element has prime order $p\in\{5,7\}$, its permutation has one $p$-cycle and $9-p$ fixed points. Normalize a unitary lift to $U^p=I$. One moved representative $f$ generates all the vectors $U^j f$ of the cycle. Their Gram matrix $K$ satisfies $\operatorname{tr}K=p$ and $\operatorname{tr}K^2=p(1+(p-1)\mu^2)$. Cauchy–Schwarz on the nonzero eigenvalues gives

$$\operatorname{rank}K\ge\frac{p}{1+(p-1)\mu^2}=\begin{cases}40/13>3&p=5,\\112/31>3&p=7.\end{cases}$$

The cyclic span is therefore all of $\mathbb C^4$. For each eigenvalue of a normal operator, the projections of every $U^j f$ to its eigenspace are proportional. The full cyclic span forces all eigenspaces to be one-dimensional. There are at least two fixed lines. Being nonorthogonal eigenlines, they must share an eigenspace, which is impossible.

## The unitary projective automorphism group must be trivial

Every element of order three must act freely. A free order-three permutation has three 3-cycles. The Gram can be rephased into the three-orbit model, which is ruled out by the exact obstruction in [the $C_3$ analysis](ansatz.md). The only prime divisors of the order of a subgroup of $S_9$ are $2,3,5,7$. Cauchy's theorem and the preceding exclusions therefore force the projective automorphism group to be trivial.

## Antiunitary symmetry for a subsequent search

If an antiunitary permutes the lines, its square acts as a scalar, by triviality of the unitary projective automorphism group. Its permutation is an involution and hence has a fixed line in degree nine. Evaluating the square on a unit vector of that line gives $+1$. Choose phases so that a fixed vector is real in the corresponding real form and a swapped pair is represented by conjugate vectors. With $P$ the permutation matrix, $P^TGP=\bar G$.

Five fixed lines would give a $5\times5$ real Gram matrix $I+\mu S$ of rank at most four, where $S$ is integral. This would give $-1/\mu$ as an eigenvalue of $S$. Its square $32/5$ is rational and fails to be an integer, contradicting algebraic integrality. The two possible cycle types therefore have one or three fixed points. The numbers of continuous Gram phases, after removing the gauge that rephases each swapped pair oppositely, are respectively 16 and 15. Indeed, if there are $s$ pairs and $f=9-2s$ fixed vertices, each two-element orbit of unordered edges gives one phase, and the $s$ edges internal to swapped pairs each give one phase. The remaining $\binom f2$ individually fixed edges have real signs. Subtracting $s$ gauge phases leaves $(36-s-\binom f2)/2$.

The general complex ETF$(4,9)$ existence question remains open. Any solution would have to lack unitary projective symmetries, while the antiunitary cases above remain available for structured exploration.
