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
        #TODO: Add regularization with the determinant of the W matrix
        loss = 1/(2*self.M) * torch.linalg.matrix_norm(self.X - input, ord='fro')
        return loss.real

class ShiftNMFLoss_halff(torch.nn.Module):
    """
    This class is equivalent to SHifNMFLoss, but here the fourier transform is just calculated over half of
    the frequency space and then rebuilt using symmetry
    """
    def __init__(self, x: torch.tensor):
        super().__init__()
        self.N, self.M = x.shape
        self.X = x
        Xf = torch.fft.fft(self.X, dim=1)
        # Keep only the first half of the Fourier transform (due to symmetry)
        Xf = Xf[:, :(Xf.shape[1] // 2) + 1]
        # Get the size of Xf
        Nf = Xf.shape
        Xf_reverse = torch.flip(Xf[:, 1:Nf[1] - 1], dims=[1])
        # Concatenate the original columns of Xf with the reversed columns along the second dimension
        self.Xft = torch.cat((Xf, torch.conj(Xf_reverse)), dim=1)

    def forward(self, input):
        #TODO: Add regularization with the determinant of the W matrix
        loss = 1/(2*self.M) * torch.linalg.matrix_norm(self.Xft - input, ord='fro')
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

class MVR_ShiftNMF_Loss(torch.nn.Module):
    def __init__(self, x: torch.tensor, lamb=0.01):
        super().__init__()
        self.N, self.M = x.shape
        self.Xf = torch.fft.fft(x)
        self.lamb = lamb
        self.eps = 1e-6

    def forward(self, inp, H): # Loss function must take the reconstruction and H.
        loss = 1 / (2 * self.M) * torch.linalg.matrix_norm(self.Xf - inp, ord='fro')
        vol_H = torch.log(torch.det(torch.matmul(H, H.T) + self.eps))
        reg = self.lamb/2 * vol_H
        print(f"Loss: {loss}, Regularization: {reg}")
        return loss + reg