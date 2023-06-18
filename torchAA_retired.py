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
        self.softmax = torch.nn.Softmax(dim=1)
        self.lossfn = frobeniusLoss(self.X)
        # Initialization of Tensors/Matrices S and C with size Col x Rank and Rank x Col
        # NxM (X) * MxD (C) = NxD (XC)
        # NxD (XC) * DxM (S) = NxM (XCS)    
        
        self.C = torch.nn.Parameter(torch.rand(n_col, rank, requires_grad=True))
        self.S = torch.nn.Parameter(torch.rand(rank, n_col, requires_grad=True))


    def forward(self):
        # Implementation of AA - F(C, S) = ||X - XCS||^2

        # first matrix Multiplication with softmax
        self.XC = torch.matmul(self.X.double(),
                               self.softmax(
                                   self.C.double()))

        # Second matrix multiplication with softmax
        self.XCS = torch.matmul(self.XC.double(),
                                self.softmax(
                                    self.S.double()))

        x = self.X - self.XCS

        return x

    def fit(self, verbose=False):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.3)

        # early stopping
        es = earlyStop(patience=5, offset=-0.3)

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
            if verbose:
                print(f"epoch: {len(running_loss)}, Loss: {loss.item()}", end='\r')

        C, S = list(self.parameters())

        C = self.softmax(C)
        S = self.softmax(S)

        C = C.detach().numpy()
        S = S.detach().numpy()

        return C, S
    


if __name__ == "__main__":
    print("hello")
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.io
    mat = scipy.io.loadmat('helpers/data/NMR_mix_DoE.mat')

    #Get X and Labels. Probably different for the other dataset, but i didn't check :)
    X = mat.get('xData')

    plt.plot(X[1])
    AA = torchAA(X, 3)
    C,S = AA.fit(verbose=True)
    XC = np.matmul(X,C)

    # Second matrix multiplication with softmax
    XCS = np.matmul(C,S)

    plt.plot(XCS[1])
    plt.show()
    
    
