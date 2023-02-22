import numpy as np


from data import X


"""
Python implementation of Morten Mørup's Furthest Sum algorithm

-K           Either a data matrix or a kernel matrix
-noc         number of candidate archetypes to extract
-i           inital observation used for to generate the FurthestSum - Initial rows of data for algo
-exclude     Entries in K that can not be used as candidates

"""

def FurthestSum(K, noc, i, exclude=[]):
    # FurthestSum algorithm to efficiently generate initial seeds/archetypes
    #
    #
    # Input:
    #   K           Either a data matrix or a kernel matrix
    #   noc         number of candidate archetypes to extract
    #   i           inital observation used for to generate the FurthestSum
    #   exclude     Entries in K that can not be used as candidates
    #
    # Output:
    #   i           The extracted candidate archetypes
    
    I, J = K.shape
    index = np.ones(J, dtype=bool)
    index[exclude] = False
    index[i] = False
    ind_t = i
    sum_dist = np.zeros(J)
    if J > noc * I:
        # Fast implementation for large scale number of observations. Can be improved by reusing calculations
        Kt = K.copy()
        Kt2 = np.sum(Kt**2, axis=0)
        for k in range(noc + 10):
            if k > noc - 1:
                Kq = Kt[:, i[0]].T @ Kt
                sum_dist -= np.sqrt(Kt2 - 2*Kq + Kt2[i[0]])
                index[i[0]] = True
                i = i[1:]
            t = np.where(index)[0]
            Kq = Kt[:, ind_t].T @ Kt
            sum_dist += np.sqrt(Kt2 - 2*Kq + Kt2[ind_t])
            ind = np.argmax(sum_dist[t])
            ind_t = t[ind]
            i = np.append(i, ind_t)
            index[ind_t] = False
    else:
        # Generate kernel if K not a kernel matrix
        if I != J or np.sum(K - K.T) != 0:
            Kt = K.copy()
            K = Kt.T @ Kt
            K = np.sqrt(np.tile(np.diag(K).T, (J, 1)) - 2*K + np.tile(np.diag(K), (1, J)))
        Kt2 = np.diag(K)
        for k in range(noc + 10):
            if k > noc - 1:
                sum_dist -= np.sqrt(Kt2 - 2*K[i[0], :] + Kt2[i[0]])
                index[i[0]] = True
                i = i[1:]
            t = np.where(index)[0]
            sum_dist += np.sqrt(Kt2 - 2*K[ind_t, :] + Kt2[ind_t])
            ind = np.argmax(sum_dist[t])
            ind_t = t[ind]
            i = np.append(i, ind_t)
            index[ind_t] = False
    return i
            
print(FurthestSum(X, 3, 3))