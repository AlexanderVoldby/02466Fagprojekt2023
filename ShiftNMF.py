import torch

from helpers.data import X
from helpers.callbacks import earlyStop
from helpers.losses import frobeniusLoss


class ShiftNMF(torch.nn.Module):
    def __init__(self, X, rank):
        super().__init__()

        # Shape of Matrix for reproduction
        self.rank = rank
        self.n_row, self.n_col = X.shape
        self.X = torch.tensor(X)
        self.softplus = torch.nn.Softplus()
        self.lossfn = frobeniusLoss(self.X)

        # Initialization of Tensors/Matrices a and b with size NxR and RxM
        self.W = torch.nn.Parameter(torch.rand(self.n_row, rank, requires_grad=True))
        self.H = torch.nn.Parameter(torch.rand(rank, self.n_col, requires_grad=True))
        self.tau = torch.nn.Parameter(torch.rand(self.n_row, rank, requires_grad=True))

        self.optim = torch.optim.Adam(self.parameters(), lr=0.3)

    def forward(self):
        # Implementation of NMF - F(A, B) = ||X - AB||^2
        Ht = torch.fft.fft(self.softplus(self.H))
        omega = torch.exp(torch.tensor(-1j*2.*torch.pi/self.n_row))
        shifts = torch.tensor([[torch.pow(omega, i*j) for i in range(self.rank)]
                              for j in range(self.n_row)])*self.tau
        Wt = self.softplus(self.W)*shifts
        WtHt = torch.matmul(Wt, Ht)
        return WtHt

    def run(self, verbose=False):
        es = earlyStop(patience=5, offset=-0.1)
        running_loss = []
        while not es.trigger():
            # zero optimizer gradient
            self.optim.zero_grad()

            # forward
            output = self.forward()

            # backward
            loss = 1/(2*self.n_col)*self.lossfn.forward(output)
            loss.backward()

            # Update W and H
            self.optim.step()

            running_loss.append(loss.item())
            es.count(loss.item())

            # print loss
            if verbose and len(running_loss) % 50 == 0:
                print(f"epoch: {len(running_loss)}, Loss: {loss.item()}")

        if verbose:
            print(f"Final loss: {running_loss[-1]}")

        W, H, tau = list(self.parameters())

        W = self.softplus(W).detach().numpy()
        H = self.softplus(H).detach().numpy()

        return W, H, tau


if __name__ == "__main__":
    nmf = ShiftNMF(X, 4)
    W, H, tau = nmf.run(verbose=True)
