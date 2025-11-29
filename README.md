# Group–Dependent Graph-Based DRO Logistic Regression

This project implements a **Wasserstein distributionally robust logistic regression** model for mixed numerical / categorical features with **group-dependent ground metric**, using a **graph-based (DAG) reformulation** of the dual problem.

---

## 1. Setup & Original DRO Formulation

We observe training samples
$$
  \xi^i = (x^i, z^i, g_i, y^i), \quad i=1,\dots,N,
$$
where

- \(x^i \in \mathbb{R}^{n_x}\): numerical features  
- \(z^i = (z^i_1,\dots,z^i_m)\), with
  \[
    z^i_\ell \in \{1,\dots,k_\ell\}, \quad \ell=1,\dots,m
  \]
- \(g_i \in \{1,\dots,\texttt{num\_g}\}\): group index  
- \(y^i \in \{-1,+1\}\): label  

We use a logistic model
\[
  f_\beta(x,z,g)
  = \beta_0 + \beta_x^\top x + \beta_z^\top \phi_z(z) + \beta_{\text{grp}}^\top \phi_g(g),
\]
with reduced-dummy encodings \(\phi_z(\cdot)\) and \(\phi_g(\cdot)\), and logistic loss
\[
  \ell_\beta(x,z,g,y) = \log\bigl(1 + \exp(-y\,f_\beta(x,z,g))\bigr).
\]

### 1.1 Group-dependent ground metric

Given a training sample \(\xi^i = (x^i,z^i,g_i,y^i)\), we define
\[
  d(\xi^i,\xi)
  = A_{g} \sum_{j=1}^{n_x} \gamma_j \lvert x_j - x^i_j\rvert
  + B_{g_i} \sum_{\ell=1}^{m} \delta_\ell \mathbf{1}[z_\ell \neq z^i_\ell]
  + C_{g_i} \mathbf{1}[g \neq g_i]
  + \infty \cdot \mathbf{1}[y \neq y^i].
\]

- \(A_g>0\): continuous part weight for destination group \(g\)  
- \(B_{g_i}>0\), \(C_{g_i}\ge0\): categorical / group-change penalties based on origin group \(g_i\)  
- \(\gamma_j>0\), \(\delta_\ell>0\): feature scalings  

Let \(\widehat P_N = \frac1N \sum_{i=1}^N \delta_{\xi^i}\) be the empirical distribution and \(W_d\) the 1-Wasserstein distance induced by \(d(\cdot,\cdot)\).  
The DRO problem with radius \(\varepsilon>0\) is
\[
  \min_{\beta} \;
  \sup_{Q : W_d(Q,\widehat P_N)\le\varepsilon}
  \mathbb{E}_{\xi\sim Q}\bigl[\ell_\beta(\xi)\bigr].
\]

### 1.2 Dual (non-graph) formulation

Using standard Wasserstein duality (no label shift), the problem can be written as
\[
\begin{aligned}
\min_{\lambda\ge0,\; r\in\mathbb{R}^N,\;\beta}\quad
  & \lambda \varepsilon + \frac1N \sum_{i=1}^N r_i \\[2pt]
\text{s.t.}\quad
  & \ell_\beta(x^i,z,g,y^i)
    - \lambda\Bigl(
        B_{g_i} \sum_{\ell=1}^{m} \delta_\ell \mathbf{1}[z_\ell\neq z^i_\ell]
        + C_{g_i}\mathbf{1}[g\neq g_i]
      \Bigr)
    \;\le\; r_i,\\
  &\qquad \forall i\in[N],\;\forall z\in\prod_{\ell=1}^m\{1,\dots,k_\ell\},\;
    \forall g\in\{1,\dots,\texttt{num\_g}\},\\[2pt]
  & \lvert \beta_{xj}\rvert \;\le\; \lambda\,\gamma_j\,A_{\min},
    \quad \forall j=1,\dots,n_x,
\end{aligned}
\]
where
\[
  A_{\min} := \min_{g} A_g.
\]

The bottleneck is the **infinite family of constraints** over all categorical realizations \((z,g)\). The graph-based formulation replaces these with a finite set of constraints on a DAG.

---

## 2. Graph-Based Dual Formulation

For each sample \(i\in[N]\), we build a directed acyclic graph
\[
  G^i = (\mathcal{V}^i,\mathcal{A}^i),
\]
whose nodes are DP states \((k,d)\):

- \(k = 0,1,\dots,m,m+1\): number of categorical components processed  
- \(d \ge 0\): accumulated categorical distance
  \[
    d = \sum_{\ell=1}^{m} \delta_\ell \mathbf{1}[z_\ell \neq z^i_\ell].
  \]

- Source node: \((0,0)\)  
- Sink node: \((m+1,0)\)

### 2.1 Edges

- **Categorical edges** (`CatArc`):  
  From \((k-1,d_{\text{prev}})\) to \((k,d)\) encoding the choice of category \(c\in\{1,\dots,k_\ell\}\) at component \(k\), and updating \(d\) accordingly.  
  Dual weight:
  \[
    w^i(e)
    = -y^i \,\beta_{z,k}^\top \phi_{z_k}(c),
  \]
  where \(\phi_{z_k}(c)\) is the reduced dummy vector for feature \(k\).

- **Terminal edges** (`TermArc`):  
  From \((m,d)\) to \((m+1,0)\) for each group choice \(g\in\{1,\dots,\texttt{num\_g}\}\).  
  Dual weight:
  \[
  \begin{aligned}
    w^i(e) &=
      - y^i \,\beta_{\text{grp}}^\top \phi_g(g) \\
      &\quad - \log\Bigl(
        \exp\bigl(
          r_i + \lambda\bigl(B_{g_i} d + C_{g_i}\mathbf{1}[g\neq g_i]\bigr)
        \bigr) - 1
      \Bigr),
  \end{aligned}
  \]
  with \(\phi_g(g)\) the reduced dummy encoding of group \(g\).

For each node \(v\in\mathcal{V}^i\), we introduce a dual potential \(\mu^i_v\).

### 2.2 Graph-based convex problem

The infinite constraints in the non-graph dual are replaced by edge constraints:
\[
  \mu^i_{t(e)} - \mu^i_{s(e)} \;\ge\; w^i(e;\beta,\lambda,r_i),
  \quad \forall i,\;\forall e\in\mathcal{A}^i.
\]

The full graph-based formulation is:
\[
\begin{aligned}
\min_{\lambda, r, \beta, \mu}\quad
  & \lambda \varepsilon + \frac1N \sum_{i=1}^N r_i \\[3pt]
\text{s.t.}\quad
  & y^i\bigl(\beta_x^\top x^i + \beta_0\bigr)
    \;\ge\; -\mu^i_{(0,0)} + \mu^i_{(m+1,0)},
    && \forall i\in[N], \\[4pt]
  & \mu^i_{t(e)} - \mu^i_{s(e)} \;\ge\; w^i(e;\beta,\lambda,r_i),
    && \forall i\in[N],\;\forall e\in\mathcal{A}^i, \\[4pt]
  & \lvert \beta_{xj}\rvert \;\le\; \lambda\,\gamma_j A_{\min},
    && \forall j=1,\dots,n_x, \\[4pt]
  & \lambda \ge 0,\quad r\in\mathbb{R}^N,\quad
    \beta \in \mathbb{R}^{1 + n_x + p_z + (\texttt{num\_g}-1)}, \\
  & \mu^i \in \mathbb{R}^{|\mathcal{V}^i|},\quad \forall i\in[N].
\end{aligned}
\]

In the implementation, we additionally enforce domain constraints
\[
  r_i + \lambda\bigl(B_{g_i} d + C_{g_i}\mathbf{1}[g\neq g_i]\bigr) \;>\; 0
\]
for each terminal edge, ensuring that the term \(\log(\exp(\cdot)-1)\) is well-defined.

---

## 3. How to Call the Main Function

The main entry point is:

```julia
model, meta = build_group_dro_graph_model(
    X,         # N × n_x continuous features
    Z,         # N × m categorical features (original labels 1..k_ℓ)
    group,     # length-N group indices g_i ∈ {1,..,num_g}
    y,         # length-N labels in {-1, +1}
    encinfo,   # ZGEncodingInfo(k_z, z_start, num_g)
    delta,     # length-m Hamming weights δ_ℓ
    A_group,   # length-num_g continuous weights A_g
    B_group,   # length-num_g categorical weights B_g
    C_group,   # length-num_g group-change penalties C_g
    gamma_x,   # length n_x continuous scalings γ_j
    ε,         # Wasserstein radius
    optimizer, # e.g. optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
)
