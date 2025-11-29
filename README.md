## 1. Setup & Original DRO Formulation

We observe training samples

$$
\xi^i = (x^i, z^i, g_i, y^i), \quad i=1,\dots,N,
$$

where

- $x^i \in \mathbb{R}^{n_x}$: numerical features  
- $z^i = (z^i_1,\dots,z^i_m)$, with $z^i_\ell \in \{1,\dots,k_\ell\},\ \ell=1,\dots,m$: categorical features  
- $g_i \in \{1,\dots,\texttt{num\_g}\}$: group index  
- $y^i \in \{-1,+1\}$: label  

We use a logistic model
$$
f_\beta(x,z,g)
= \beta_0 + \beta_x^\top x + \beta_z^\top \phi_z(z) + \beta_{\text{grp}}^\top \phi_g(g),
$$
and logistic loss
$$
\ell_\beta(x,z,g,y) = \log\bigl(1 + \exp(-y\,f_\beta(x,z,g))\bigr).
$$

### 1.1 Group-dependent ground metric

Given a training sample $\xi^i = (x^i,z^i,g_i,y^i)$, we define
$$
d(\xi^i,\xi)
= A_{g} \sum_{j=1}^{n_x} \gamma_j \lvert x_j - x^i_j\rvert
+ B_{g_i} \sum_{\ell=1}^{m} \delta_\ell \mathbf{1}[z_\ell \neq z^i_\ell]
+ C_{g_i} \mathbf{1}[g \neq g_i]
+ \infty \cdot \mathbf{1}[y \neq y^i].
$$

...

## 2. Graph-Based Dual Formulation

For each sample $i\in[N]$, we build a directed acyclic graph
$$
G^i = (\mathcal{V}^i,\mathcal{A}^i),
$$
whose nodes are DP states $(k,d)$, with
$$
k = 0,1,\dots,m,m+1,\qquad
d = \sum_{\ell=1}^{m} \delta_\ell \mathbf{1}[z_\ell \neq z^i_\ell] \ge 0.
$$

Source node $(0,0)$, sink node $(m+1,0)$.

...
