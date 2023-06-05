import torch
from torch.optim import Adam, lr_scheduler
from helpers.callbacks import ChangeStopper
from helpers.losses import frobeniusLoss, VolLoss
from helpers.data import X_clean
import matplotlib.pyplot as plt

class NMF(torch.nn.Module):
    def __init__(self, X, rank, alpha=1e-9):
        super().__init__()

        n_row, n_col = X.shape
        self.softplus = torch.nn.Softplus()
        self.lossfn = frobeniusLoss(torch.tensor(X))
        self.stopper = ChangeStopper(alpha=alpha)

        # Initialization of Tensors/Matrices a and b with size NxR and RxM
        self.W = torch.nn.Parameter(torch.rand(n_row, rank, requires_grad=True))
        self.H = torch.nn.Parameter(torch.rand(rank, n_col, requires_grad=True))

        self.optim = Adam(self.parameters(), lr=0.4)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', factor=0.1, patience=5)

    def forward(self):
        WH = torch.matmul(self.softplus(self.W), self.softplus(self.H))
        return WH

    def fit(self, verbose=False, return_loss=False,  stopper=ChangeStopper()):
        running_loss = []
        while not stopper.trigger():
            # zero optimizer gradient
            self.optim.zero_grad()

            # forward
            output = self.forward()

            # backward

            loss = self.lossfn.forward(output)
            loss.backward()

            # Update W and H
            self.optim.step()
            self.scheduler.step(loss)

            running_loss.append(loss.item())
            stopper.track_loss(loss)

            # print loss
            if verbose:
                print(f"epoch: {len(running_loss)}, Loss: {loss.item()}", end='\r')

        W = self.softplus(self.W).detach().numpy()
        H = self.softplus(self.H).detach().numpy()

        if return_loss:
            return W, H, running_loss
        else:
            return W, H


class MVR_NMF(torch.nn.Module):
    def __init__(self, X, rank):
        super().__init__()

        n_row, n_col = X.shape
        self.rank = rank
        self.softmax = torch.nn.Softmax(dim=1) # dim = 1 is on the rows
        self.lossfn = VolLoss(torch.tensor(X))
        self.softplus = torch.nn.Softplus()

        # Initialization of Tensors/Matrices a and b with size NxR and RxM
        self.W = torch.nn.Parameter(torch.rand(n_row, rank, requires_grad=True))
        self.H = torch.nn.Parameter(torch.rand(rank, n_col, requires_grad=True))

        self.optim = Adam(self.parameters(), lr=0.3)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', factor=0.1, patience=5)

    def forward(self):
        WH = torch.matmul(self.softmax(self.W), self.softplus(self.H))
        return WH

    def fit(self, verbose=False, return_loss=False):
        running_loss = []
        while not self.stopper.trigger():
            # zero optimizer gradient
            self.optim.zero_grad()

            # forward
            output = self.forward()

            # backward
            loss = self.lossfn.forward(output, self.softplus(self.H))
            loss.backward()

            # Update W and H
            self.optim.step()
            self.scheduler.step(loss)

            running_loss.append(loss.item())
            self.stopper.track_loss(loss)

            # print loss
            if verbose:
                print(f"epoch: {len(running_loss)}, Loss: {loss.item()}", end='\r')

        W = self.softmax(self.W).detach().numpy()
        H = self.softplus(self.H).detach().numpy()

        if return_loss:
            return W, H, running_loss
        else:
            return W, H


if __name__ == "__main__":
    from helpers.callbacks import explained_variance
    mvr_nmf = MVR_NMF(X_clean, 6)
    W, H = mvr_nmf.fit(verbose=True)
    print(f"Explained variance MVR_NMF: {explained_variance(X_clean, mvr_nmf.forward().detach().numpy())}")
    plt.figure()
    for vec in H:
        plt.plot(vec)
    plt.title("H")
    plt.show()