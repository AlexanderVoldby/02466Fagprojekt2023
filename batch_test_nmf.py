import torch
from Artificial_no_shift import X
from torchNMF import NMF
import numpy as np

best_loss = np.inf
print("starting")
for i in range(100):
    print(i)
    nmf = NMF(X, 3, lr=0.3, factor=0.8, patience=10)
    C, S, loss = nmf.fit(verbose=False, return_loss=True)
    if loss[-1] < best_loss:
        bestNMF = nmf
        best_loss = loss[-1]
print("Best loss:")
print(best_loss)
torch.save(bestNMF.state_dict(), "NMF_best_weights")








