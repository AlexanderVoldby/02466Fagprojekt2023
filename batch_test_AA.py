import torch
from Artificial_no_shift import X
from torchAA import torchAA
import numpy as np

best_loss = np.inf
for i in range(100):
    print(i)
    AA = torchAA(X, 3, lr=0.3, alpha=1/10000, factor=0.8, patience=10)
    C, S, loss = AA.fit(verbose=False, return_loss=True)
    if loss[-1] < best_loss:
        bestAA = AA
        best_loss = loss[-1]
print("Best loss:")
print(best_loss)
torch.save(bestAA.state_dict(), "AA_best_weights")








