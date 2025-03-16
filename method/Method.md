# Harmony Algorithm

As most of the data have cell barcodes in rows, we adjusted the algorithm, and thus it's slightly different from the paper.

In this document, we resummarise the Harmony algorithm.

## Notations

Given an embedding of $N$ cell barcodes in $d$ dimensions, coming from $B$ batches, Harmony tries to cluster them into $K$ clusters first, then integrate data.

* $Z \in \mathbb{R}^{N \times d}$: Input embedding to be corrected by Harmony.
* $\hat{Z} \in \mathbb{R}^{N \times d}$: Output embedding which is integrated.
* $R \in [0, 1]^{N \times K}$: Soft cluster assignment matrix of cells (rows) to clusters (columns).
* $\phi \in \{0, 1\}^{N \times B}$: One-hot assignment matrix of cells (rows) to batches (columns).
* $Pr \in [0, 1]^B$: Frequency of batches.
* $O \in [0, 1]^{B \times K}$: The observed co-occurrence matrix of cells in batches (rows) and clusters (columns).
* $E \in [0, 1]^{B \times K}$: The expected co-occurrence matrix of cells in batches (rows) and clusters (columns), under assumption of independence between cluster and batch assignment.
* $Y \in [0, 1]^{K \times d}$: L2-Normalized cluster centroid locations.

## Objective Function

* K-Means error:

$$
e_1 = \sum_{i, k} R_{ik}||Z_i - Y_k||^2
$$

for $\forall 1 \leq i \leq N$ and $\forall 1 \leq k \leq K$.

Moreover, if both $Z_i$ and $Y_k$ are L2-normalized, their euclidean distance is transformed as cosine distance:

$$
e_1 = \sum_{i, k} 2R_{ik}(1 - Z_{i} \cdot Y_{k}^T) = \sum_{i, k} 2R * (1 - Z Y^T)
$$

where $*$ is element-wise product.

* Cross-entropy Error:

$$
e_2 = \sigma \sum_{i, k} R_{ik}\log{R_{ik}} = \sigma \sum_{i, k}R * \log{R}
$$

* Diversity Penalty:

$$
\begin{align*}
e_3 &= \sigma \sum_{i, k} \theta R_{ik} \sum_{b}\phi_{ib}\log{\Big( \frac{O_{bk} + 1}{E_{bk} + 1} \Big)} \\
    &= \sigma \sum_{b, k} \theta \Big[ (\phi^T R) * \log{\Big( \frac{O + 1}{E + 1} \Big)} \Big] \\
    &= \sigma \sum_{i, k} \theta \Big[ O * \log{\Big( \frac{O + 1}{E + 1} \Big)} \Big]
\end{align*}
$$

where $\theta$ of shape $1 \times B$ are the discounting hyperparameters.

Therefore, the objective function is

$$
E = e_1 + e_2 + e_3.
$$

## Algorithm Structure

```python
def harmonize(Z, phi):
    Z_hat = Z
    R, E, O = initialize_centroids(Z_hat)
    while not converged:
        R = clustering(Z_hat, phi)
        Z_hat = correction(Z, R, phi)

    return Z_hat
```

## Centroids Initialization

1. L2-normalize $\hat{Z}$ on rows.

2. $\hat{Y} = kmeans(\hat{Z}, K)$. And then L2-normalize $\hat{Y}$ on rows.

3. Initialize $R$:

$$
R = \exp{\Big(-\frac{2(1 - \hat{Z} \hat{Y}^T)}{\sigma}\Big)}
$$

Then L1-normalize $R$ on rows, so that each row sums up to 1.

4. Initialize $E$ and $O$:

```math
\begin{align*}
(E)_{bk} = Pr_b \cdot \sum_{i = 1}^N R_{ik} \qquad &\Rightarrow \qquad E = Pr^T \cdot [R_{\cdot 1}, \dots, R_{\cdot K}];\\
(O)_{bk} = \sum_{i = 1}^N \phi_{ib}R_{ik} \qquad &\Rightarrow \qquad O = \phi^T R.
\end{align*}
```

5. Compute objective value with $\hat{Y}$, $\hat{Z}$, $R$, $O$, and $E$.

## Clustering

### Block-wise Update

1. Compute $O$ and $E$ on left-out data:

```math
E = E - Pr^T \cdot [R_{in, 1}, \dots, R_{in, K}], \qquad O = O - \phi_{in}^T R_{in}.
```

where $R_{in, 1}, ..., R_{in, K}$ are the summations of $R_{ik}$ over cells in the current block regarding each cluster $k$.

2. Update and normalize $R$:

```math
\begin{align*}
R_{in} &= \exp{\Big( -\frac{2(1 - \hat{Z}_{in}\hat{Y}^T)}{\sigma} \Big)};\\
\Omega &= \phi^{in} \Big( \frac{E+1}{O+1} \Big)^\Theta; \\
R_{in} &= R_{in} \Omega; \\
R_{in} &= \text{L1-Normalize}(R_{in}, \text{row}).
\end{align*}
```

where $\Theta = [\theta^T, \dots, \theta^T]$ of shape $B \times K$.

3. Compute $O$ and $E$ with full data:

$$
E = E + Pr^T \cdot [R_{in, 1}, \dots, R_{in, K}], \qquad O = O + \phi_{in}^T R_{in}.
$$

4. Update cluster centroids:

$$
\begin{align*}
\hat{Y} &= \sum_{i = 1}^N R_{ik}\hat{Z}_{id} = R^T \hat{Z};\\
\hat{Y} &= \text{L2-Normalize}(\hat{Y}, \text{row}).
\end{align*}
$$

5. Compute objective value with updated $\hat{Y}$, $\hat{Z}$, $R$, $O$, and $E$.

## Correction

### Original Method

1. Initialize $\hat{Z}$ by $Z$.

2. Let

$$
\phi^* = \begin{bmatrix}
1 & \phi_{11} & \cdots & \phi_{1B} \\
\vdots & \vdots & \ddots & \vdots \\
1 & \phi_{N1} & \cdots & \phi_{NB}
\end{bmatrix}
$$

3. Cluster-wise correction:

For each cluster $k$,

```math
\begin{align*}
R_k &= [R_{1k}, \dots, R_{Nk}];\\
\Phi_{R,k}^* &= \phi^{*T} \otimes R_k;\\
W_k &= (\Phi_{R,k}^* \phi^* + \lambda J)^{-1} \Phi_{R,k}^* Z;\\
W_k[0, :] &= \mathbf{0};\\
\hat{Z} &= \hat{Z} - \Phi_{R,k}^{*T} W_k.
\end{align*}
```

where $\otimes$ is row-wise multiplication of a matrix and a row vector, and

$$
J = \begin{bmatrix}
0 & 0 & 0 & \cdots & 0\\
0 & 1 & & & \\
0 &   & 1 & & \\
\vdots &   &   & \ddots & \\
0 & & & & 1
\end{bmatrix}.
$$


### Improvement

We don't need to directly calculate the matrix inverse:

```math
(\Phi_{R,k}^* \phi^* + \lambda J)^{-1}
```

of shape $(B+1)\times(B+1)$, which can be time consuming when the number of batches $B$ is high.

Let $A_k = \phi^{*T}diag(R_k)\phi^* + \lambda J$, then

```math
W_k = A_k^{-1}\Phi_{R, k}^* Z.
```

Since

```math
\begin{align*}
A_k &= \begin{bmatrix}
1 & \cdots & 1 \\
\phi_{11} & \cdots & \phi_{N1} \\
\vdots & \vdots & \vdots \\
\phi_{1B} & \cdots & \phi_{NB}
\end{bmatrix} \cdot \begin{bmatrix}
R_{1k} & & \\
 & \ddots & \\
 & & R_{Nk}
\end{bmatrix} \cdot \begin{bmatrix}
1 & \phi_{11} & \cdots & \phi_{1B} \\
\vdots & \vdots & \ddots & \vdots \\
1 & \phi_{N1} & \cdots & \phi_{NB}
\end{bmatrix} + \lambda J \\
&= \begin{bmatrix}
\sum_{i = 1}^N R_{ik} & \sum_{i = 1}^N \phi_{i1}R_{ik} & \cdots & \sum_{i = 1}^N \phi_{iB}R_{ik} \\
\sum_{i = 1}^N \phi_{i1}R_{ik} & \sum_{i = 1}^N \phi_{i1}^2 R_ik & \cdots & \sum_{i = 1}^N \phi_{i1}\phi_{iB}R_{ik} \\
\vdots & \vdots & \ddots & \vdots \\
\sum_{i = 1}^N \phi_{iB}R_{ik} & \sum_{i = 1}^N \phi_{iB}\phi_{i1}R_{ik} & \cdots & \sum_{i = 1}^N \phi_{iB}^2R_{ik}
\end{bmatrix} + \lambda J,
\end{align*}
```

it's easy to see that

```math
\sum_{i = 1}^N \phi_{ib_1}\phi_{ib_2}R_{ik} = 0 \qquad \text{ for } \quad \forall b_1 \neq b_2
```

and

```math
\sum_{i = 1}^N \phi_{ib}^2 R_{ik} = \sum_{i = 1}^N \phi_{ib} R_{ik}.
```

Let

$$
\begin{align*}
N_k &= \sum_{i = 1}^N R_{ik},\\
N_{bk} &= \sum_{i = 1}^N \phi_{ib}R_{ik} \qquad \Rightarrow \qquad N = \phi^T R \qquad \Rightarrow \qquad N = O.
\end{align*}
$$

Then we have

```math
N_k = \sum_{b = 1}^B O_{bk}
```

and

```math
A_k = \begin{bmatrix}
N_k & O_{1k} & \cdots & O_{Bk} \\
O_{1k} & O_{1k} & & \\
\vdots & & \ddots & \\
O_{Bk} & & & O_{Bk}
\end{bmatrix} + \lambda J = \begin{bmatrix}
N_k & O_{1k} & \cdots & O_{Bk} \\
O_{1k} & O_{1k} + \lambda & & \\
\vdots & & \ddots & \\
O_{Bk} & & & O_{Bk} + \lambda
\end{bmatrix}.
```

Let

```math
P = \begin{bmatrix}
1 & -\frac{O_{1k}}{O_{1k} + \lambda} & \cdots & -\frac{O_{Bk}}{O_{Bk} + \lambda} \\
 & 1 &  &  \\
 & & \ddots & \\
 & & & 1
\end{bmatrix}
```

then

```math
\mathcal{B} = PAP^T = \begin{bmatrix}
c & & & \\
  & O_{1k}+\lambda & & \\
  & & \ddots & \\
  & & & O_{Bk}+\lambda
\end{bmatrix},
```

where

```math
c = N_k - \sum_{i = 1}^N \frac{O_{ik}^2}{O_{ik}+\lambda}.
```

$\mathcal{B}$ has inverse

```math
\mathcal{B}^{-1} = \begin{bmatrix}
c^{-1} & & & \\
 & \frac{1}{O_{1k}+\lambda} & & \\
 & & \ddots & \\
 & & & \frac{1}{O_{Bk}+\lambda}
\end{bmatrix}.
```

Therefore,

```math
\begin{align*}
A^{-1} &= P^T\mathcal{B}^{-1}P \\
&= \begin{bmatrix}
c^{-1} & & & \\
-\frac{O_{1k}}{O_{1k}+\lambda}c^{-1} & \frac{1}{O_{1k}+\lambda} & & \\
\vdots & & \ddots & \\
-\frac{O_{Bk}}{O_{Bk}+\lambda}c^{-1} & & & \frac{1}{O_{Bk}+\lambda}
\end{bmatrix} \cdot P
\end{align*}
```

which is decomposited into a lower-triangular, a diagonal, and an upper-triangular matrix.
