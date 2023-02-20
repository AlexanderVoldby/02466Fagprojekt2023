import numpy as np


from data import X


"""
Python implementation of Morten Mørup's Furthest Sum algorithm

-K           Either a data matrix or a kernel matrix
-noc         number of candidate archetypes to extract
-i           inital observation used for to generate the FurthestSum - Initial rows of data for algo
-exclude     Entries in K that can not be used as candidates

"""

def furthest_sum(K, noc, i, exclude = None):
    
    I, J = K.shape
    
    index = np.array(range(1,J))
    
    if exclude != None:
        index[exclude] = 0
    
    # print(np.where(K == i)[0])
    index[[i]] = 0
    ind_t = i
    
    # i=K[i]
    
    sum_dist = np.zeros(shape = (1,J))
    

    
    # print(i.shape)
    # exit()
    if J > noc*I: #Remove initial seed
        Kt = K
        Kt2 = np.sum(np.power(Kt, 2), axis=0)
        for _ in range(noc+10):
            
            #Originally Kq=Kt(:,i(1)), but im way too tired to figure out what that means
            #maybe i is a list???, and i think matlabs accesses by (), and uses 1-indexing
            Kq=np.dot(Kt[:,i[0]].conj().T, Kt)
            
            print(Kt2.shape)
            print(Kq.shape)
            print(Kt2[i[0]])

            # print(Kt2[i[0]])

            sum_dist = sum_dist-np.sqrt(Kt2-2*Kq + Kt2[i[0]])
            index[i[0]] = i[0]
            #originally i(1)=[];  i think it should just remove such that index 0 is the next element
            #how fun, now we get an out of bounds error
            i = np.delete(i, 0)
        
        
        t = np.nonzero(index)
        #Kq = np.dot(Kt[:,ind_t].conj().T, Kt)
        Kq = np.dot(Kt[ind_t].conj().T, Kt)
        sum_dist = sum_dist + np.sqrt(Kt2 - 2*Kq + Kt2[ind_t])
        
        val = max(sum_dist[t])
        ind = sum_dist[t].index(val)
        
        ind_t = t[ind[0]]
        # i=[i t(ind(1))]; N/A or space character in matlab opertors
        i = [i, t[ind[0]]]
        index[t[ind[0]]] = 0
    
    else:
        if I != J or sum(sum(K-K.T)) != 0:
            Kt=K
            K = np.dot(Kt.H, Kt)
            K=np.sqrt(np.tile(
                np.diag(K)
            ))
        
        Kt2 = np.diag(K)
        
        for k in range(noc+10):
            if k > noc-1:
                sum_dist=sum_dist-np.sqrt(Kt2-2*K[i[0],:])+Kt2[i[0]]
                index[i[0]] = i[0]
                i[0] = []
            t = np.nonzero(index)
            sum_dist=sum_dist+np.sqrt(Kt2-2*K[ind_t,:] + Kt2[ind_t])
            
            val = max(sum_dist[t])
            ind = sum_dist[t].index(val)
            
            ind_t = t[ind[0]]
            i = [i, ind_t]
            index[t[ind[0]]] = 0
            
furthest_sum(X, 3, [i for i in range(20)])