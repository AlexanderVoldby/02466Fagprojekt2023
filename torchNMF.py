import torch

from helpers.data import X
from helpers.callbacks import earlyStop
from helpers.losses import frobeniusLoss


class NMF(torch.nn.Module):
    def __init__(self, X, rank):
        super().__init__()

        n_row, n_col = X.shape
        self.X = torch.tensor(X)
        self.softplus = torch.nn.Softplus()
        self.lossfn = frobeniusLoss(self.X)

        # Initialization of Tensors/Matrices a and b with size NxR and RxM
        self.W = torch.nn.Parameter(torch.rand(n_row, rank, requires_grad=True))
        self.H = torch.nn.Parameter(torch.rand(rank, n_col, requires_grad=True))

        self.optim = torch.optim.Adam(self.parameters(), lr=0.3)

    def forward(self):
        WH = torch.matmul(self.softplus(self.W), self.softplus(self.H))
        return WH

    def fit(self, verbose=False):
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

            # Update W and H
            self.optim.step()

            running_loss.append(loss.item())
            es.count(loss.item())

            # print loss
            if verbose and len(running_loss) % 50 == 0:
                print(f"epoch: {len(running_loss)}, Loss: {loss.item()}")

        W, H = list(self.parameters())

        W = self.softplus(W).detach().numpy()
        H = self.softplus(H).detach().numpy()

        return W, H