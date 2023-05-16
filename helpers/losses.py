import torch
import numpy as np

#Defining frobenius Loss
class frobeniusLoss(torch.nn.Module):
    def __init__(self, x: torch.tensor):
        super(frobeniusLoss, self).__init__()
        self.X = x
    
    def forward(self, input):
        return torch.linalg.matrix_norm(self.X - input, ord='fro')


class ShiftNMFLoss(torch.nn.Module):
    def __init__(self, x: torch.tensor):
        super().__init__()
        _, self.M = x.shape
        self.X = x

    def forward(self, input):
        # TODO: Add regularization with the determinant of the W matrix
        loss = 1/(2*self.M) * torch.linalg.matrix_norm(self.X - input, ord='fro')**2
        return loss


class VolLoss(torch.nn.Module):
    def __init__(self, x: torch.tensor, alpha=0.1):
        super(VolLoss, self).__init__()
        self.X = x
        self.alpha = alpha


    def forward(self, w, h, x):
        # TODO: We might have to change to constraining on H since it should denote the source.
        return torch.linalg.det(self.alpha*(h.T@h))+torch.linalg.matrix_norm(self.X - x, ord='fro')**2

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
        vol_W = torch.det(torch.matmul(H, H.T) + self.eps)
        reg = self.lamb * vol_W
        # print(f"Loss: {loss}, Regularization: {reg}")
        return loss + reg