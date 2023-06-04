import torch
from torch.optim import Adam, lr_scheduler

import sys
sys.path.append("../..")
from helpers.callbacks import ChangeStopper
from helpers.losses import frobeniusLoss
from helpers.initializers import FurthestSum

print("torchAA.py")

class torchAA(torch.nn.Module):
    def __init__(self, X, rank, initializer = None, noc = 10, power = 1, initial = 0, exclude = []):
        super(torchAA, self).__init__()

        # Shape of Matrix for reproduction
        n_row, n_col = X.shape
        self.X = torch.tensor(X)

        self.softmax = torch.nn.Softmax(dim=0)
        self.lossfn = frobeniusLoss(self.X)
        
        
        if initializer is not None:
            cols = FurthestSum(X, noc, initial, exclude)
            self.C = torch.zeros(n_col, rank)
            for i in cols:
                self.C[i] = power
                
            self.C = torch.tensor(self.C, requires_grad=True)
            self.C = torch.nn.Parameter(self.C)
        else:
            self.C = torch.nn.Parameter(torch.rand(n_col, rank, requires_grad=True))
        # print(self.C)
        
        self.S = torch.nn.Parameter(torch.rand(rank, n_col, requires_grad=True))


    def forward(self):

        # first matrix Multiplication with softmax
        XC = torch.matmul(self.X, self.softmax(self.C).double())

        # Second matrix multiplication with softmax
        XCS = torch.matmul(XC, self.softmax(self.S).double())

        return XCS

    def fit(self, verbose=False, return_loss=False, stopper = ChangeStopper()):
        optimizer = Adam(self.parameters(), lr=0.5)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        # Convergence criteria
        running_loss = []

        while not stopper.trigger():
            # zero optimizer gradient
            optimizer.zero_grad()

            # forward
            output = self.forward()

            # backward
            loss = self.lossfn.forward(output)
            loss.backward()

            # Update A and B
            optimizer.step()
            scheduler.step(loss)
            # append loss for graphing
            running_loss.append(loss.item())

            # count with early stopping
            stopper.track_loss(loss)

            # print loss
            if verbose:
                print(f"epoch: {len(running_loss)}, Loss: {loss.item()}", end='\r')

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
    