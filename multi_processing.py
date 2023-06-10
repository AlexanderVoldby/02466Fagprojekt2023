
import torch.multiprocessing as mp
from torchAA import torchAA
from Artificial_no_shift import X
import numpy as np
import torch

def AA_train(i, X, D, results, kwargs):
    AA = torchAA(X, D, **kwargs)
    C, S, loss = AA.fit(verbose=True, return_loss=True)
    results.put((i, AA, loss[-1]))
    



if __name__ == '__main__':
    num_processes  = 4
    return_dict = {}
    results = mp.Queue()
    kwargs ={"lr":0.3, "alpha":1/10000, "factor":0.8, "patience":10}
    #number of components
    D = 3
    process = mp.spawn(AA_train,args=(X,D,results,kwargs,), nprocs=num_processes , join=False)
    #timeout is number of seconds for each process to finish
    if not process.join(timeout=2):
        print("\n Process timeout or failure")
    best_loss = np.inf
    for _ in range(num_processes):
        i, AA, loss = results.get()
        if loss<best_loss:
            best_AA = AA

    torch.save(best_AA.state_dict(), "AA_weights")



