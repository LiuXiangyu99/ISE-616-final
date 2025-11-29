# ISE-616-final

# Mixed-Feature Training Data and Distance Function

This document summarizes the format of the training data and the definition of the distance function used in the model.

---

## 1. Training Data Format

We work with a supervised binary classification setting. The training set contains \(N\) samples. For each sample \(i = 1, \dots, N\), we observe:

- A vector of **numerical features** \(x^i \in \mathbb{R}^n\)
- A vector of **categorical features** \(z^i \in \prod_{l=1}^m C_l\)
- A **group index** \(g^i \in [\texttt{num\_g}] = \{1, 2, \dots, \texttt{num\_g}\}\)
- A **binary label** \(y^i \in \{-1, +1\}\)

The raw training data is stored as:

- `X` (size \(N \times n\)): numerical feature matrix  
- `Z` (size \(N \times m\)): categorical feature matrix  
- `group` (length \(N\)): group index for each sample  
- `y` (length \(N\)): labels

### 1.1 Categorical Features

We assume there are \(m\) categorical components. For each component \(l = 1, \dots, m\), we define a finite category set
\[
C_l = \{1, 2, \dots, k_l\}.
\]

For each sample \(i\) and component \(l\), the entry \(Z[i, l]\) (denoted \(z_l^i\)) is an integer in \(C_l\). We do **not** use one-hot encoding at the level of the raw training set: each categorical component is stored as a single integer code.

### 1.2 Group Index

The group index `group[i]` of sample \(i\) is an integer
\[
g^i \in [\texttt{num\_g}] = \{1, 2, \dots, \texttt{num\_g}\},
\]
where `num_g` is the total number of possible groups.

The group index may later be one-hot encoded if needed for a specific model, but the core definitions in this document use the integer representation.

---

## 2. Unified Sample Representation

For convenience, we write a single sample as
\[
\xi = (x, z, g, y),
\]
where
- \(x \in \mathbb{R}^n\) is the numerical feature vector,
- \(z \in \prod_{l=1}^m C_l\) is the categorical feature vector,
- \(g \in [\texttt{num\_g}]\) is the group index,
- \(y \in \{-1, +1\}\) is the label.

Similarly, we denote another sample as
\[
\xi' = (x', z', g', y').
\]

---

## 3. Distance Function

We define a distance function \(d(\xi, \xi')\) between two samples \(\xi = (x, z, g, y)\) and \(\xi' = (x', z', g', y')\) as follows:
\[
d(\xi, \xi')
=
A_{g'} \sum_{j=1}^n \gamma_j \lvert x_j - x'_j \rvert
\;+\;
B_{g'} \sum_{l=1}^m \delta_l \mathbf{1}[z_l \neq z'_l]
\;+\;
C_{g'} \mathbf{1}[g \neq g']
\;+\;
\infty \cdot \mathbf{1}[y \neq y'].
\]

Here:

- \(\gamma_j \ge 0\) is a weight associated with numerical feature \(j\), for \(j = 1, \dots, n\).
- \(\delta_l \ge 0\) is a weight associated with categorical component \(l\), for \(l = 1, \dots, m\).
- \(A_{g'} \ge 0\), \(B_{g'} \ge 0\), and \(C_{g'} \ge 0\) are **group-dependent coefficients** indexed by the destination group \(g'\in [\texttt{num\_g}]\).
  - \(A_{g'}\) scales the numerical feature distance when the destination sample \(\xi'\) belongs to group \(g'\).
  - \(B_{g'}\) scales the categorical feature distance for destination group \(g'\).
  - \(C_{g'}\) is the penalty for transporting mass between different groups when the destination group is \(g'\).
- \(\mathbf{1}[\cdot]\) is the indicator function, equal to \(1\) if the condition holds and \(0\) otherwise.

The last term enforces that:
- If \(y \neq y'\), then \(d(\xi, \xi') = \infty\). In other words, transportation between samples with different labels is forbidden.

### 3.1 Numerical Part

The numerical part of the distance is
\[
A_{g'} \sum_{j=1}^n \gamma_j \lvert x_j - x'_j \rvert.
\]
This is a weighted \(\ell_1\)-type distance on the numerical features, with:
- per-feature weights \(\gamma_j\), and
- an additional scaling factor \(A_{g'}\) depending on the destination group.

### 3.2 Categorical Part

The categorical part of the distance is
\[
B_{g'} \sum_{l=1}^m \delta_l \mathbf{1}[z_l \neq z'_l].
\]
For each categorical component \(l\):
- we compare the integer codes \(z_l\) and \(z'_l\),
- we incur a cost \(\delta_l\) if they are different, and
- this cost is scaled by the group-dependent factor \(B_{g'}\).

Note that each component \(z_l\) is a **single integer category**, not a one-hot vector. Therefore, each categorical component contributes at most one unit to the indicator term \(\mathbf{1}[z_l \neq z'_l]\).

### 3.3 Group Penalty

The group penalty is
\[
C_{g'} \mathbf{1}[g \neq g'].
\]
This term allows, but penalizes, moving mass between different groups:
- If \(g = g'\), this term is 0 (no extra penalty for within-group transport).
- If \(g \neq g'\), we pay a group-dependent cost \(C_{g'}\) when transporting from group \(g\) to group \(g'\).

### 3.4 Label Constraint

The label term
\[
\infty \cdot \mathbf{1}[y \neq y']
\]
ensures that transportation is **restricted to samples with the same label**:
- If \(y = y'\), the term is 0 and the distance is finite (subject to the other terms).
- If \(y \neq y'\), the distance is infinite and such transportation is disallowed.

---

## 4. Summary

- The training data consists of numerical features \(x\), categorical features \(z\), a group index \(g\), and a binary label \(y\).
- Categorical features are stored as integer codes per component (no one-hot at the raw data level).
- The distance function \(d(\xi,\xi')\) combines:
  - a weighted \(\ell_1\) distance on the numerical features,
  - a weighted, component-wise indicator distance on the categorical features,
  - a finite penalty for cross-group transportation, and
  - an infinite penalty for cross-label transportation (which effectively forbids moving mass between different labels).

This distance will be used as the ground metric in the distributionally robust optimization formulation.
