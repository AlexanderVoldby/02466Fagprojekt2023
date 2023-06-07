import torch
from torch.optim import Adam, lr_scheduler
from helpers.callbacks import ChangeStopper
from helpers.losses import frobeniusLoss
from helpers.initializers import FurthestSum

import scipy.io

class torchAA(torch.nn.Module):
    def __init__(self, X, rank, alpha=1e-9):
        super(torchAA, self).__init__()

        # Shape of Matrix for reproduction
        N, M = X.shape
        self.X = torch.tensor(X, dtype=torch.double)

        self.softmax = torch.nn.Softmax(dim=1)
        self.lossfn = frobeniusLoss(self.X)
        
        self.C = torch.nn.Parameter(torch.randn(rank, N, requires_grad=True, dtype=torch.double)*3)
        self.S = torch.nn.Parameter(torch.randn(N, rank, requires_grad=True, dtype=torch.double)*3)
        
        # mat = scipy.io.loadmat('helpers/PCHA/C.mat')
        # self.C = mat.get('c')
        # self.C = torch.tensor(self.C, requires_grad=True, dtype=torch.double)
        # self.C = torch.nn.Parameter(self.C.T)
        
        # mat = scipy.io.loadmat('helpers/PCHA/S.mat')
        # self.S = mat.get('s')
        # self.S = torch.tensor(self.S, requires_grad=True, dtype=torch.double)
        # self.S = torch.nn.Parameter(self.S.T)
        
        
        self.optimizer = Adam(self.parameters(), lr=0.2)
        self.stopper = ChangeStopper(alpha=alpha)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.9, patience=5)

 
    def forward(self):

        # first matrix Multiplication with softmax
        CX = torch.matmul(self.softmax(self.C), self.X)

        # Second matrix multiplication with softmax
        SCX = torch.matmul(self.softmax(self.S), CX)

        return SCX

    def fit(self, verbose=False, return_loss=False):

        # Convergence criteria
        running_loss = []

        while not self.stopper.trigger():
            # zero optimizer gradient
            self.optimizer.zero_grad()

            # forward
            output = self.forward()

            # backward
            loss = self.lossfn.forward(output)
            loss.backward()

            # Update A and B
            self.optimizer.step()
            self.scheduler.step(loss)
            # append loss for graphing
            running_loss.append(loss.item())

            # count with early stopping
            self.stopper.track_loss(loss)

            # print loss
            if verbose:
                print(f"epoch: {len(running_loss)}, Loss: {1-loss.item()}", end='\r')

        C = self.softmax(self.C)
        S = self.softmax(self.S)

        C = C.detach().numpy()
        S = S.detach().numpy()

        if return_loss:
            return C, S, running_loss
        else:
            return C, S


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.io
    mat = scipy.io.loadmat('helpers/data/NMR_mix_DoE.mat')

    # Get X and Labels. Probably different for the other dataset, but i didn't check :)
    X = mat.get('xData')
    
    AA = torchAA(X, 3)
    C, S = AA.fit(verbose=True)
    CX = np.matmul(C, X)
    SCX = np.matmul(S, CX)
    plt.figure()
    for vec in CX:
        plt.plot(vec)
    plt.title("Archetypes")
    plt.show()
    plt.figure()
    plt.plot(X[1], label="First signal of X")
    plt.plot(SCX[1], label="Reconstructed signal with AA")
    plt.legend()
    plt.show()

    plt.plot(X[2])
    plt.plot(SCX[2])
    plt.show()
    