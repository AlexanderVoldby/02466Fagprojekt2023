import torch
from Artificial_no_shift import X
from torchAA import torchAA

AA = torchAA(X, 3, lr=0.3, alpha=1/10000, factor=0.8, patience=10)
C, S = AA.fit(verbose=True)

torch.save(AA.state_dict(), "AA_weights")








