import torch
from torch.optim import Adam, lr_scheduler

import sys
sys.path.append("../..")
from helpers.callbacks import ChangeStopper
from helpers.losses import frobeniusLoss

print("torchAA.py")
#this is a version of AA thats exactly written like the 
class torchAA(torch.nn.Module):
    def __init__(self, X, rank):
        super(torchAA, self).__init__()

        # Shape of Matrix for reproduction
        n_row, n_col = X.shape
        self.X = torch.tensor(X)

        self.softmax = torch.nn.Softmax(dim=0)
        self.lossfn = frobeniusLoss(self.X)
        
        self.C = torch.nn.Parameter(torch.rand(n_col, rank, requires_grad=True))
        self.S = torch.nn.Parameter(torch.rand(rank, n_col, requires_grad=True))


    def forward(self):

        # first matrix Multiplication with softmax
        XC = torch.matmul(self.X.double(), self.softmax(self.C).double())

        # Second matrix multiplication with softmax
        XCS = torch.matmul(XC, self.softmax(self.S).double())

        return XCS

    def fit(self, verbose=False, return_loss=False, stopper = ChangeStopper()):
        stopper.reset()
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
    #mat = scipy.io.loadmat('helpers/data/NMR_mix_DoE.mat')

    # Get X and Labels. Probably different for the other dataset, but i didn't check :)
    #X = mat.get('xData')
    from Artificial_no_shift import X
    X = X.T
    AA = torchAA(X, 3)
    C, S, loss = AA.fit(verbose=True, return_loss=True)
    XC = np.matmul(X, C)
    print("loss2:")
    print(loss[-1])
    # Second matrix multiplication with softmax
    XCS = np.matmul(XC, S)
    plt.figure()
    for vec in XC.T:
        plt.plot(vec)
    plt.title("Archetypes of AA2")
    plt.show()
    plt.figure()
    plt.plot(X.T[1], label="First signal of X")
    plt.plot(XCS.T[1], label="Reconstructed signal with AA2")
    plt.legend()
    plt.show()

  