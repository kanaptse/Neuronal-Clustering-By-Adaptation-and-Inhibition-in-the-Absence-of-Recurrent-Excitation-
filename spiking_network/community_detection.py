import pandas as pd
import numpy as np
from sklearn.covariance import EmpiricalCovariance as EC


def checkDesignMatrix(X:np.ndarray):
    X = np.atleast_2d(X)
    T, N = X.shape
    return T, N

def marcenkoPastur(X:np.ndarray):
    T, N = checkDesignMatrix(X)
    q = N / float(T)

    lambda_min = (1 - np.sqrt(q))**2
    lambda_max = (1 + np.sqrt(q))**2

    def rho(x):
        ret = np.sqrt((lambda_max - x) * (x - lambda_min))
        ret /= 2 * np.pi * q * x
        return ret if lambda_min < x < lambda_max else 0.0

    return (lambda_min, lambda_max), rho

def Structured_component(X:np.ndarray,global_mode=False):   
    T, N = checkDesignMatrix(X)


    C = np.corrcoef(X.T)

    eigvals, eigvecs = np.linalg.eigh(C)
    eigvecs = eigvecs.T

    (lambda_min, lambda_max), _ = marcenkoPastur(X)
    xi_clipped = np.where(eigvals >= lambda_max, eigvals, np.nan)

    if global_mode:
        xi_clipped[-1] = np.nan
    
    Cl = np.zeros((N, N), dtype=float)
    for xi, eigvec in zip(xi_clipped, eigvecs):
        if np.isnan(xi):
            pass
        else:
            eigvec = eigvec.reshape(-1, 1)
            Cl += xi * eigvec.dot(eigvec.T)
    
    return C,Cl


def Louvain(X:np.ndarray,global_mode=False):
    C, ECIJ = Structured_component(X,global_mode)
    n = len(C)
    M0 = np.arange(n) 
    Cnorm = 2 * C.sum().sum()
    
    _, Mb = np.unique(M0, return_inverse=True)
    M = Mb.copy()

   

    Hnm = np.zeros((n, n))
    for m in range(np.max(Mb) + 1):
        Hnm[:, m] = np.sum(ECIJ[:, Mb == m], axis=1)
    H = np.sum(Hnm, axis=1)
    Hm = np.sum(Hnm, axis=0)

    Q0 = -np.inf
    Q = np.sum(ECIJ[np.equal.outer(M0, M0)]) / Cnorm
    first_iteration = True

    while Q - Q0 > 1e-16:
        flag = True
        while flag:
            flag = False
            for u in np.random.permutation(n):
                ma = Mb[u]
                dQ = Hnm[u, :] - Hnm[u, ma] + ECIJ[u, u]
                dQ[ma] = 0

                max_dQ, mb = np.max(dQ), np.argmax(dQ)
                if max_dQ > 1e-10:
                    flag = True
                    Mb[u] = mb
                    Hnm[:, mb] = Hnm[:, mb] + ECIJ[:, u]
                    Hnm[:, ma] = Hnm[:, ma] - ECIJ[:, u]
                    Hm[mb] = Hm[mb] + H[u]
                    Hm[ma] = Hm[ma] - H[u]

        _, Mb = np.unique(Mb, return_inverse=True)

        M0 = M.copy()
        if first_iteration:
            M = Mb.copy()
            first_iteration = False
        else:
            for u in range(n):
                M[M0 == u] = Mb[u]

        n = np.max(Mb) + 1
        B1 = np.zeros((n, n))
        for u in range(n):
            for v in range(u, n):
                bm = np.sum(ECIJ[np.ix_(Mb == u, Mb == v)])
                B1[u, v] = bm
                B1[v, u] = bm

        ECIJ = B1 / np.sum(np.triu(ECIJ))

        Mb = np.arange(n)
        Hnm = ECIJ
        H = np.sum(ECIJ, axis=0)
        Hm = H

        Q0 = Q
        Q = np.trace(ECIJ) / Cnorm
        num_of_clusters = len(np.unique(Mb))
    return M,num_of_clusters,Q
        
# def Louvain(X:np.ndarray,global_mode=False):
#     C, Cl = Structured_component(X,global_mode)
#     partition = np.arange(len(C))
#     Cnorm = C.sum().sum()
#     Cl = pd.DataFrame(Cl)
#     Cl[abs(Cl)<0.1]=0
#     Q = 0
    
#     for i in np.random.permutation(len(C)):
#         max_value = (Cl-np.diag(np.diagonal(Cl))).loc[i,:].max()
#         if max_value>0:            
#             max_value_col = np.where(((Cl-np.diag(np.diagonal(Cl))).loc[i,:]).values == max_value)
#             max_value_col = Cl.columns[max_value_col][0]
#             Cl.loc[max_value_col,:] += Cl.loc[i,:]
#             Cl.loc[:,max_value_col] += Cl.loc[:,i]
#             Cl.loc[max_value_col,max_value_col] += Cl.loc[i,i]
#             Cl = Cl.drop(i,axis=0)
#             Cl = Cl.drop(i,axis=1)
#             Q = np.diagonal(Cl).sum()/Cnorm
#             partition[partition==i] = max_value_col
  
#     num_of_clusters = len(np.unique(partition))
#     return partition,num_of_clusters,Q,Cl
        