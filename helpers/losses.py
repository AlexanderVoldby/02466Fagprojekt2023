import torch
import numpy as np

#Defining frobenius Loss
class frobeniusLoss(torch.nn.Module):
    def __init__(self, x: torch.tensor):
        super(frobeniusLoss, self).__init__()
        self.X = x
    
    def forward(self, input):
        return torch.linalg.matrix_norm(self.X - input, ord='fro')**2


class ShiftNMFLoss(torch.nn.Module):
    def __init__(self, x: torch.tensor):
        super().__init__()
        self.N, self.M = x.shape
        self.X = torch.fft.fft(x)
        # TODO: Definer over (f-1)/2 frekvenser
    def forward(self, input):
        # TODO: Probably each of these loss functions work fine (the outcommented code runs as well)
        # TODO: Find which has the best performance
        # loss = 0
        # for f in range(self.M):
            #loss += torch.matmul(torch.conj(self.X[:, f] - input[:, f]), self.X[:, f] - input[:, f])

        loss = 1/(2*self.M) * torch.linalg.matrix_norm(self.X - input, ord='fro')
        return loss.real
    
class VolLoss(torch.nn.Module):
    def __init__(self, x: torch.tensor, alpha=0.1):
        super(VolLoss, self).__init__()
        self.X = x
        self.alpha = alpha

        

    def forward(self, w, h, x):
        # TODO: We might have to change to constraining on H since it should denote the source.
        return torch.linalg.det(self.alpha*(h.T@h))+torch.linalg.matrix_norm(self.X - x, ord='fro')**2

# Sparseness measure of the H-matrix
def sparseness(h):
    r, N = h.shape
    nonzero = np.count_nonzero(h)
    return 1 - nonzero/(r*N)
