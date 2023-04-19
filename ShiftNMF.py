import torch

from helpers.data import X
from helpers.callbacks import earlyStop
from helpers.losses import ShiftNMFLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class ShiftNMF(torch.nn.Module):
    def __init__(self, X, rank):
        super().__init__()

        # Shape of Matrix for reproduction
        self.rank = rank
        self.N, self.M = X.shape
        self.X = torch.tensor(X)
        self.softplus = torch.nn.Softplus()
        self.lossfn = ShiftNMFLoss(self.X)

        # Initialization of Tensors/Matrices a and b with size NxR and RxM
        self.W = torch.nn.Parameter(torch.rand(self.N, rank, requires_grad=True))
        self.H = torch.nn.Parameter(torch.rand(rank, self.M, requires_grad=True))
        self.tau = torch.nn.Parameter(torch.rand(self.N, rank, requires_grad=True))

        self.optim = torch.optim.Adam(self.parameters(), lr=0.3)

    def forward(self):
        # The underlying signals in the frequency space
        Hf = torch.fft.fft(self.softplus(self.H))
        # The matrix that approximates the observations
        WHt = torch.empty((self.N, self.M), dtype=torch.cfloat)

        for f in range(self.M):
            omega = torch.ones((self.N, self.rank)) * 2 * torch.pi * f / self.M
            exp_tau = torch.exp(-1j * omega * self.tau)
            Wf = self.softplus(self.W) * exp_tau
            WHt[:, f] = torch.matmul(Wf, Hf[:, f])
        return WHt

    def fit(self, verbose=False):
        es = earlyStop(patience=5, offset=-0.1)
        running_loss = []
        while not es.trigger():
            # zero optimizer gradient
            self.optim.zero_grad()

            # forward
            print("Forward pass...")
            output = self.forward()

            # backward
            loss = self.lossfn(output)
            print("Backward pass...")
            loss.backward()

            # Update W, H and tau
            print("Updating W, H, and tau")
            print()
            self.optim.step()

            running_loss.append(loss.item())
            es.count(loss.item())

            # print loss
            print(f"epoch: {len(running_loss)}, Loss: {loss.item()}")

        W, H, tau = list(self.parameters())

        W = self.softplus(W).detach().numpy()
        H = self.softplus(H).detach().numpy()

        return W, H, tau


if __name__ == "__main__":
    nmf = ShiftNMF(X, 4)
    nmf.to(device)
    W, H, tau = nmf.fit(verbose=True)
