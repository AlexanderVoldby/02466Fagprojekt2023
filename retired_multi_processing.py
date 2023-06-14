
import torch.multiprocessing as mp
from torchAA import torchAA
from torchNMF import NMF
from Artificial_no_shift import X
import numpy as np
import torch
import time

torch.device('cpu')

#https://discuss.pytorch.org/t/segfault-with-multiprocessing-queue/81292
mp_spawn = mp.get_context('spawn')

def AA_train(i, X, D, results, kwargs):
    print("starting:"+str(i))
    AA = torchAA(X, D, **kwargs)
    C, S, loss = AA.fit(verbose=False, return_loss=True)
    torch.save(AA.state_dict(), "./train_weights/AA_weights_"+str(i))
    results.put([i, C, S, loss[-1]])

def NMF_train(i, X, D, results, kwargs):
    print("starting:"+str(i))
    nmf = NMF(X, D, **kwargs)
    W, H, loss = nmf.fit(verbose=False, return_loss=True)
    torch.save(nmf.state_dict(), "./train_weights/NMF_weights_"+str(i))
    results.put([i, W, H, loss[-1]])
    


if __name__ == '__main__':
    print("STARTTIME:")
    print(time.localtime())
    num_processes = 100
    results = mp_spawn.Queue()
    kwargs ={"lr":0.3, "alpha":1/10000, "factor":0.8, "patience":10}
    #number of components
    D = 3
    process = mp.spawn(NMF_train,args=(X,D,results,kwargs,), nprocs=num_processes , join=False)
    #timeout is number of seconds for each process to finish
    print("")
    if not process.join(timeout=None):
        print("Process timeout or failure")
    print("pre Q")
    print(results.qsize())
    print("Done")
    best_loss = np.inf
    for _ in range(results.qsize()):
        i, C, S, loss = results.get()
        print(str(i)+" loss:"+str(loss))
        if loss<best_loss:
           best_run = i
           best_loss = loss

    print("")
    print("Best index:"+str(best_run))
    print("")
    print("ENDTIME:")
    print(time.localtime())


