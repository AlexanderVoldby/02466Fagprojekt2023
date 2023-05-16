import torch

from helpers.callbacks import earlyStop
from helpers.losses import frobeniusLoss


class torchAA(torch.nn.Module):
    def __init__(self, X, rank):
        super(torchAA, self).__init__()

        # Shape of Matrix for reproduction
        n_row, n_col = X.shape
        self.X = torch.tensor(X)

        # softmax layer
        self.softmax = torch.nn.Softmax(dim=0)
        self.lossfn = frobeniusLoss(self.X)
        # Initialization of Tensors/Matrices S and C with size Col x Rank and Rank x Col
        # DxN (C) * NxM (X) =  DxM (CX)
        # NxD (S) *  DxM (CX) = NxM (SCX)    
        
        self.C_tilde = torch.nn.Parameter(torch.rand(rank, n_row, requires_grad=True))
        self.S_tilde = torch.nn.Parameter(torch.rand(n_row, rank, requires_grad=True))
        self.C = lambda: self.softmax(self.C_tilde)
        self.S = lambda: self.softmax(self.S_tilde)

    def forward(self):
        # Implementation of AA - F(C, S) = ||X - XCS||^2

        # first matrix Multiplication with softmax
        CX = torch.matmul(self.C().double(), self.X.double())

        # Second matrix multiplication with softmax
        SCX = torch.matmul(self.S().double(), CX.double())

        return SCX

    def fit(self, verbose=False):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.3)

        # early stopping
        es = earlyStop(patience=5, offset=-0.001)

        running_loss = []

        while (not es.trigger()):
            # zero optimizer gradient
            optimizer.zero_grad()

            # forward
            output = self.forward()

            # backward
            loss = self.lossfn.forward(output)
            loss.backward()

            # Update A and B
            optimizer.step()

            # append loss for graphing
            running_loss.append(loss.item())

            # count with early stopping
            es.count(loss.item())

            # print loss
            if verbose and len(running_loss) % 50 == 0:
                print(f"epoch: {len(running_loss)}, Loss: {loss.item()}", end='\r')

        C, S = list(self.parameters())

        C = self.softmax(C)
        S = self.softmax(S)

        C = C.detach().numpy()
        S = S.detach().numpy()

        return C, S
    


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.io
    mat = scipy.io.loadmat('helpers/data/NMR_mix_DoE.mat')

    #Get X and Labels. Probably different for the other dataset, but i didn't check :)
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
    