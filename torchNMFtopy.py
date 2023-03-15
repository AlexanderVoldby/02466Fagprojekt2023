import torch

from helpers.data import X
from helpers.callbacks import earlyStop
from helpers.losses import frobeniusLoss


class torchNMF(torch.nn.Module):
    def __init__(self, X, rank):
        super().__init__()

        # Shape of Matrix for reproduction
        n_row, n_col = X.shape
        self.X = torch.tensor(X)
        self.softplus = torch.nn.Softplus()
        self.optim = torch.optim.Adam(self.parameters(), lr=0.3)
        self.lossfn = frobeniusLoss(self.X)

        # Initialization of Tensors/Matrices a and b with size NxR and RxM
        self.A = torch.nn.Parameter(torch.rand(n_row, rank, requires_grad=True))
        self.B = torch.nn.Parameter(torch.rand(rank, n_col, requires_grad=True))

    def forward(self):
        # Implementation of NMF - F(A, B) = ||X - AB||^2
        AB = torch.matmul(self.softplus(self.A), self.softplus(self.B))
        return AB

    def run(self, verbose=False):
        es = earlyStop(patience=5, offset=-0.1)
        running_loss = []
        while not es.trigger():
            # zero optimizer gradient
            self.optim.zero_grad()

            # forward
            output = self.forward()

            # backward

            loss = self.lossfn.forward(output)
            loss.backward()

            # Update A and B
            self.optim.step()

            running_loss.append(loss.item())
            es.count(loss.item())

            # print loss
            if verbose:
                print(f"epoch: {len(running_loss)}, Loss: {loss.item()}", end='\r')

        A, B = list(self.parameters())

        A = self.softplus(A)
        B = self.softplus(B)

        return A, B