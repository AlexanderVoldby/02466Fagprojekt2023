import torch
from torch.optim import Adam, lr_scheduler
from helpers.callbacks import ChangeStopper
from helpers.losses import frobeniusLoss



class torchAA(torch.nn.Module):
    def __init__(self, X, rank):
        super(torchAA, self).__init__()

        # Shape of Matrix for reproduction
        N, M = X.shape
        self.X = torch.tensor(X)

        self.softmax = torch.nn.Softmax(dim=1)
        self.lossfn = frobeniusLoss(self.X)
        
        self.C = torch.nn.Parameter(torch.rand(rank, N, requires_grad=True))
        self.S = torch.nn.Parameter(torch.rand(N, rank, requires_grad=True))


    def forward(self):

        # first matrix Multiplication with softmax
        CX = torch.matmul(self.softmax(self.C).double(), self.X.double())

        # Second matrix multiplication with softmax
        SCX = torch.matmul(self.softmax(self.S).double(), CX.double())

        return SCX

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
    