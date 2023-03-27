import torch
import numpy as np

#Defining frobenius Loss
class frobeniusLoss(torch.nn.Module):
    def __init__(self, x: torch.tensor):
        super(frobeniusLoss, self).__init__()
        self.loss = torch.linalg.matrix_norm
        self.X = x
    
    def forward(self, input):
        return self.loss(self.X - input, ord='fro')
    
class VolLoss(torch.nn.Module):
    def __init__(self, x: torch.tensor):
        super(VolLoss, self).__init__()
        self.loss = torch.linalg.det
        self.X = x

        

    def forward(self, w, h, x):
        return self.loss((w.T@w))+torch.linalg.matrix_norm(self.X - x, ord='fro')

# Sparseness measure of the H-matrix
def sparseness(h):
    r, N = h.shape
    nonzero = np.count_nonzero(h)
    return 1 - nonzero/(r*N)
