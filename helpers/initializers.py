import numpy as np
from numpy.matlib import repmat

def FurthestSum(K, noc, i, exclude=[]):
    """
    Python implementation of Morten Mørup's Furthest Sum algorithm

    FurthestSum algorithm to efficiently generate initial seeds/archetypes

    Input:
    K           Either a data matrix or a kernel matrix
    noc         number of candidate archetypes to extract
    i           inital observation used for to generate the FurthestSum
    exclude     Entries in K that can not be used as candidates

    Output:
    i           The extracted candidate archetypes

    """
    I, J = K.shape
    index = np.ones(J, dtype=bool)
    index[exclude] = False
    index[i] = False
    ind_t = i
    sum_dist = np.zeros(J)
    noc = noc-1 if noc > 1 else 1 # prevent error when noc = 1, subscript of int
    if J > noc * I:
        # Fast implementation for large scale number of observations. Can be improved by reusing calculations
        Kt = K.copy()
        Kt2 = np.sum(Kt**2, axis=0)
        for k in range(noc + 10):
            if k > noc - 1: #remove initial seed
                Kq = Kt[:, i[0]].conj().T @ Kt
                sum_dist = sum_dist - np.emath.sqrt(Kt2 - 2*Kq + Kt2[i[0]])
                index[i[0]] = True
                i = i[1:]
            t = np.where(index)[0]
            Kq = Kt[:, ind_t].conj().T @ Kt
            sum_dist = sum_dist + np.emath.sqrt(Kt2 - 2*Kq + Kt2[ind_t])
            ind = np.argmax(sum_dist[t])
            ind_t = t[ind]
            i = np.append(i, ind_t)
            index[ind_t] = False
    else:
        # Generate kernel if K not a kernel matrix
        if I != J or np.sum(K - K.conj().T) != 0:
            Kt = K.copy()
            K = Kt.conj().T @ Kt
            K = np.lib.scimath.sqrt(
                repmat(np.diag(K), J, 1) - 2 * K + \
                repmat(np.mat(np.diag(K)).T, 1, J)
            )
        Kt2 = np.diag(K).conj().T
        for k in range(noc + 11):
            if k > noc - 1:
                sum_dist = sum_dist -np.lib.scimath.sqrt(Kt2 - 2*K[i[0], :] + Kt2[i[0]])
                index[i[0]] = True
                i = i[1:]
            t = np.where(index)[0]
            sum_dist = sum_dist + np.lib.scimath.sqrt(Kt2 - 2*K[ind_t, :] + Kt2[ind_t])
            ind = np.argmax(sum_dist[t])
            ind_t = t[ind]
            i = np.append(i, ind_t)
            index[ind_t] = False
    return i

def init_s(X, C, noc):
     
    U = range(X.shape[0])
    
    XC = np.matmul(X, C)
    XCtX = np.dot(XC.T, X[:, U])
    CtXtXC = np.dot(XC.T, XC)
    S = -np.log(np.random.rand(noc, len(U)))
    S = S / (np.ones((noc, 0)) * np.sum(S, 0))
    SSt = np.dot(S, S.T)

    SST = np.sum(np.sum(X[:, U] * X[:, U]))

    SSE = SST - 2 * np.sum(XCtX * S) + np.sum(CtXtXC * SSt)
    # S, SSE, muS, SSt = Supdate(S, XCtX, CtXtXC, muS, SST, SSE, 25)



if __name__ == "__main__":
    print("This is a helper file, import it to use it.")
    import pandas as pd
    import scipy.io
    mat = scipy.io.loadmat('data/NMR_mix_DoE.mat')
    X = mat.get('xData')
    
    print(FurthestSum(X.T, 3, 0))
    
    n_col = X.shape[1]
    cols = FurthestSum(X, 3, 0)
    C = np.zeros((n_col, 3))
    for i in cols:
        C[i] = 10
    print(init_s(X, C, 3))