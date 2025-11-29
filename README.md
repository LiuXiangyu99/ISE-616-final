```math
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

```