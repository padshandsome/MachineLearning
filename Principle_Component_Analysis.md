Principle Component Analysis

Set-ups:

Suppose we have dataset X with D dimensions and n samples, that is $ X\in R^{D\times n}$.

Let $\tilde{X} = X - \overline{X} \in R^{D\times n}$ be the centered matrix for X, which means we move all the data points to the center of our coordinates, where we have $E[\tilde{X}] = 0$.

Let $C = \frac{1}{n-1}\tilde{X}\tilde{X^T} \in R^{D\times D}$ be the sample covariance matrix for our centered data.



Goal for PCA:

Sometimes there are too many features for our datasets, and we would like to reduce the dimensions. In other words, we only want features and feature combinations that contribute the most.

What kinds of feature or feature combinations contribution the most?

- That can explain the information in the dataset as much as possible.

- Information: the variance of our dataset.

That is, our goal for PCA is to find the direction that has the maximum variance of our dataset.



Derivation:
Suppose we have a transformation $u \in R^{D}$, and we can add a constrain $u^T u = 1$ for convenience.

Intuitively, if we want to find the maximum variance of our dataset, that we can find the maximum stretch in the picture. And the maximum stretch direction in variance is the the eigenvector corresponds with the maximum eigenvalue of our sample variance matrix C.
                                                             $$ C\hat{u} = \lambda \hat{u} $$

For mathematical derivation:

$$\hat{u} = \underset{u\in R^{D}}{argmax}\ Var(u^T\tilde{X}) = \underset{u\in R^{D}}{argmax}\ E[(u^T \tilde{X})(u^T \tilde{X})^T] - E^2[u^T\tilde{X}] $$

$$= \underset{u\in R^{D}}{argmax}\ u^TE[\tilde{X}\tilde{X^T}]u = \underset{u\in R^{D}}{argmax}\ u^T\frac{1}{n}\tilde{X}\tilde{X^T}u=\underset{u\in R^{D}}{argmax}\ \frac{n-1}{n}u^TCu $$

$$= \underset{u\in R^{D}}{argmax}\ u^TCu$$

Because $u$ has a constrain $u^Tu=1$, to solve this optimization problem, we can use Lagrange function:

$$L(\lambda) = u^TCu - \lambda(u^Tu -1)$$

And let the partial derivative equals 0 to find the maximum.

$$\Rightarrow\frac{\partial L}{\partial u} = 2Cu - 2\lambda u = 0$$

$$\Rightarrow Cu = \lambda u$$

That means $u$ should be the eigenvectors of $C$.

And we call the eigenvector with the maximum eigenvalue is the First Principle Component.

If $k\leq D$, we can define the first $k^{th}$ principle components as eigenvectors of $C$ corresponding to k largest eigenvalues of $C$.



In practice:

Our covariance matrix $C$ has eigendecomposition  $C = E\Lambda E^T$, where $EE^T =I $ and $\Lambda$ is diagonal.

And our centered data matrix $\tilde{X}$ has singular value decomposition $\tilde{X} = U\Sigma V^T$.

$$E\Lambda E^T = C = \frac{1}{n-1}\tilde{X}\tilde{X}^T = \frac{1}{n-1}U\Sigma V^TV \Sigma^TU = \frac{1}{n-1}U\Sigma^2 U$$

Because eigendecomposition is unique, we have $E = U$ and $\Lambda = \frac{1}{n-1}\Sigma^2$.

So, we have two approach to compute the eigenvectors and eigenvalues for $C$.

Method 1:

​     Straightforward, do the eigendecomposition for $C$ and reorder the eigenvalues and eigenvectors.

````python
C = Xtilde @ Xtilde.T / (n-1)  # Covariance Matrix
L , E = np.linalg.eig(C) # Eigendecomposition
# if numpy.linalg.eig is used, the eigenvectors and eigenvalues returned are not necessarily correctly ordered.
# Remember, we want that the first entry of L corresponds to the _largest_ eigenvalue, the second to the second largest, etc.
# For this reason, we need to reorder the entries of L in a non-increasing manner, and use the same rearrangement to reorder the columns of E.
def reorder_eigs(E,L): # reordering is the purpose of the function reorder_eigs
    ind = np.argsort(L)
    ind = ind[::-1]
    L_sorted = L[ind]
    E_sorted = E[:,ind]
    return E_sorted, L_sorted
E_sorted, L_sorted = reorder_eigs(E,L)
print(E,L)
print(E_sorted,L_sorted)
E = E_sorted
L = L_sorted
````



Method 2:

​     Do the singular value decomposition for $\tilde{X}$. $U$ has ordered eigenvectors and we can compute eigenvalues using $\Lambda = \frac{1}{n-1}\Sigma^2$.

```python
# compute PCA via (centered) data matrix
U, W, V = np.linalg.svd(Xtilde) # Singular Value Decomposition for Xtilde
Lambda = W**2 / (X[0,:].size-1)
print(U, Lambda,'\n')

print("First principal component:",U[:,0])
print("Second principal component:",U[:,1])
print("Variance explained by first principal component:",Lambda[0])
print("Variance explained by second principal component:",Lambda[1])
```



Measurements:

Variance explained by the first $k^{th}$ principle components: $\sum_{i}^{k} \lambda_i$

The ratio of the variance explained by the first $k^{th}$ principle components (contributions): $\frac{\sum_i^k\lambda_i}{\sum_i^D \lambda_i}$ 

# 补充projections 的内容

Key steps of PCA in Practice

Input: Data matrix $X=[x_1,...,x_n]\in R^{D\times n}$.

1. Mean substraction: If $\overline{X}$ is the matrix of row averages of $X$ in each column, compute

   $$\tilde{X} = X -\overline{X}$$

2. Compute principle components:

   $$E_k = [u_1,...,u_k]\in R^{D\times k}$$

   and 

   $$\Lambda_k = diag(\lambda_i)_{i=1}^{k}\in R{k\times k}$$
   
3. Projections: compute coordinates of data points in the $k$-dimensional subspace defined by $E_k$ 

   $$Z_k = E_k^T\tilde{X} \in R^{k\times n}$$
4. Obtain the coordinates of the projected data points in coordinate system of the original data matrix $X$:
 $$X_k = (E_kZ_k + \overline{X}) \in R^{D\times n}$$ 

