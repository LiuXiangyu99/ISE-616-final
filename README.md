# Group–Dependent Graph-Based DRO Logistic Regression

This project implements a **Wasserstein distributionally robust logistic regression** model for mixed numerical / categorical features with a **group-dependent ground metric**, using a **graph-based (DAG) reformulation** of the dual problem.

---

## 1. Setup & Original DRO Formulation

We observe training samples

```math
\begin{align}
    \xi^i = (x^i, z^i, g_i, y^i), \quad i=1,\dots,N,
\end{align}
```
where we have:

```math
\begin{itemize}
    \item 11
\end{itemize}
```