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

if __name__ == "__main__":
    print("This is a helper file, import it to use it.")
    import pandas as pd
    import scipy.io
    mat = scipy.io.loadmat('data/NMR_mix_DoE.mat')
    X = mat.get('xData')
    
    print(FurthestSum(X.T, 3, 0))