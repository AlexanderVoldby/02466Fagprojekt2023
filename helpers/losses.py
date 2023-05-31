import torch

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
        _, self.M = x.shape
        self.X = x

    def forward(self, input):
        loss = 1/(2*self.M) * torch.linalg.matrix_norm(self.X - input, ord='fro')**2
        return loss


class VolLoss(torch.nn.Module):
    def __init__(self, x: torch.tensor, alpha=0.1):
        super().__init__()
        self.X = x
        self.alpha = alpha

    def forward(self, x, h):
        return self.alpha*torch.linalg.det(torch.matmul(h, h.T)+1e-9)+torch.linalg.matrix_norm(self.X - x, ord='fro')**2

# Sparseness measure of the H-matrix

class MVR_ShiftNMF_Loss(torch.nn.Module):
    def __init__(self, x: torch.tensor, lamb=0.01):
        super().__init__()
        self.N, self.M = x.shape
        self.Xf = x
        self.lamb = lamb
        self.eps = 1e-9

    def forward(self, inp, H): # Loss function must take the reconstruction and H.
        loss = 1 / (2 * self.M) * torch.linalg.matrix_norm(self.Xf - inp, ord='fro')**2
        vol_H = torch.det(torch.matmul(H, H.T) + self.eps)
        reg = self.lamb * vol_H
        return loss + reg