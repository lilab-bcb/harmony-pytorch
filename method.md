# Harmony Algorithm

As most of the data have cell barcodes in rows, we adjusted the algorithm, and thus it's slightly different from the paper.

Also, since Method section of the paper has several typos, we think it necessary to reformulate the algorithm for readers' convenience.

## Notations

Given an embedding of $N$ cell barcodes in $d$ dimensions, coming from $B$ batches, Harmony tries to cluster them into $K$ clusters first, then integrate data.

* $Z \in \mathbb{R}^{N \times d}$: Input embedding to be corrected by Harmony.
* $\hat{Z} \in \mathbb{R}^{N \times d}$: Output embedding which is integrated.
* $R \in [0, 1]^{N \times K}$: Soft cluster assignment matrix of cells (rows) to clusters (columns).
* $\phi \in \{0, 1\}^{N \times B}$: One-hot assignment matrix of cells (rows) to batches (columns).
* $Pr_b \in [0, 1]^B$: Frequency of batches.
* $O \in [0, 1]^{B \times K}$: The observed co-occurrence matrix of cells in batches (rows) and clusters (columns).
* $E \in [0, 1]^{B \times K}$: The expected co-occurrence matrix of cells in batches (rows) and clusters (columns), under assumption of independence between cluster and batch assignment.
* $Y \in [0, 1]^{K \times d}$: L2-Normalized cluster centroid locations.

## Objective Function


## Algorithm Structure

## Clustering

## Correction

Let

$$ 

\phi^* = \begin{bmatrix} 
1 & \phi_{11} & \cdots & \phi_{1B} \\
\vdots & \vdots & \ddots & \vdots \\
1 & \phi_{N1} & \cdots & \phi_{NB}
\end{bmatrix}

$$

and

$$

diag(R_k) = diag(R_{1k}, \dots, R_{Nk}).

$$

### Original Method

$$

W_k = (\phi^{*T}diag(R_k)\phi^* + \lambda I)^{-1} \phi^{*T}R_kZ

$$

### Improvement

Consider $A_k = \phi^{*T}R_k\phi^* + \lambda I$,

$$
\begin{aligned}
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
\end{bmatrix} + \lambda I \\
&= \begin{bmatrix}
\sum_{i = 1}^N R_{ik} & \sum_{i = 1}^N \phi_{i1}R_{ik} & \cdots & \sum_{i = 1}^N \phi_{iB}R_{ik} \\
\sum_{i = 1}^N \phi_{i1}R_{ik} & \sum_{i = 1}^N \phi_{i1}^2 R_ik & \cdots & \sum_{i = 1}^N \phi_{i1}\phi_{iB}R_{ik} \\
\vdots & \vdots & \ddots & \vdots \\
\sum_{i = 1}^N \phi_{iB}R_{ik} & \sum_{i = 1}^N \phi_{iB}\phi_{i1}R_{ik} & \cdots & \sum_{i = 1}^N \phi_{iB}^2R_{ik}
\end{bmatrix} + \lambda I.
\end{aligned}
$$

It's easy to see that 
$$
\sum_{i = 1}^N \phi_{ib_1}\phi_{ib_2}R_{ik} = 0
$$
for $\forall b_1 \neq b_2$. And $\sum_{i = 1}^N \phi_{ib}^2 R_{ik} = \sum_{i = 1}^N \phi_{ib} R_{ik}$.

Let

$$
\begin{aligned}
N_k &= \sum_{i = 1}^N R_{ik},\\
N_{bk} &= \sum_{i = 1}^N \phi_{ib}R_{ik}.
\end{aligned}
$$

Then we have $N_k = \sum_{b = 1}^B N_{bk}$, and

$$
A_k = \begin{bmatrix}
N_k & N_{1k} & \cdots & N_{Bk} \\
N_{1k} & N_{1k} & & \\
\vdots & & \ddots & \\
N_{Bk} & & & N_{Bk}
\end{bmatrix} + \lambda I = \begin{bmatrix}
N_k + \lambda & N_{1k} & \cdots & N_{Bk} \\
N_{1k} & N_{1k} + \lambda & & \\
\vdots & & \ddots & \\
N_{Bk} & & & N_{Bk} + \lambda
\end{bmatrix}.
$$

Let 
$$
P = \begin{bmatrix}
1 & -\frac{N_{1k}}{N_{1k} + \lambda} & \cdots & -\frac{N_{Bk}}{N_{Bk} + \lambda} \\
 & 1 &  &  \\
 & & \ddots & \\
 & & & 1
\end{bmatrix}
$$
then
$$
B = PAP^T = \begin{bmatrix}
c & & & \\
  & N_{1k}+\lambda & & \\
  & & \ddots & \\
  & & & N_{Bk}+\lambda
\end{bmatrix},
$$
where 
$$
c = N_k + \lambda - \sum_{i = 1}^N \frac{N_{ik}^2}{N_{ik}+\lambda}.
$$

$B$ has inverse
$$
B^{-1} = \begin{bmatrix}
c^{-1} & & & \\
 & \frac{1}{N_{1k}+\lambda} & & \\
 & & \ddots & \\
 & & & \frac{1}{N_{Bk}+\lambda}
\end{bmatrix}.
$$

Therefore,
$$
A^{-1} = P^TB^{-1}P,
$$
which is decomposited into a lower-triangular, a diagonal, and an upper-triangular matrix.

