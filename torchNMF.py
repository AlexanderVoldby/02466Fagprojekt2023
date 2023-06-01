import torch
from torch.optim import Adam, lr_scheduler
from helpers.callbacks import ChangeStopper
from helpers.losses import frobeniusLoss, VolLoss
from helpers.data import X_clean


class NMF(torch.nn.Module):
    def __init__(self, X, rank):
        super().__init__()

        n_row, n_col = X.shape
        self.softplus = torch.nn.Softplus()
        self.lossfn = frobeniusLoss(torch.tensor(X))

        # Initialization of Tensors/Matrices a and b with size NxR and RxM
        self.W = torch.nn.Parameter(torch.rand(n_row, rank, requires_grad=True))
        self.H = torch.nn.Parameter(torch.rand(rank, n_col, requires_grad=True))

        self.optim = Adam(self.parameters(), lr=0.5)
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
                print(f"epoch: {len(running_loss)}, Loss: {loss.item()}")

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
        self.softmax = torch.nn.Softmax(dim=1)
        self.lossfn = VolLoss(torch.tensor(X))

        # Initialization of Tensors/Matrices a and b with size NxR and RxM
        self.W = torch.nn.Parameter(torch.rand(n_row, rank, requires_grad=True))
        self.H = torch.nn.Parameter(torch.rand(rank, n_col, requires_grad=True))

        self.optim = Adam(self.parameters(), lr=0.5)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', factor=0.1, patience=5)

    def forward(self):
        WH = torch.matmul(self.softmax(self.W), self.softmax(self.H))
        return WH

    def fit(self, verbose=False, return_loss=False, stopper = ChangeStopper()):
        running_loss = []
        while not stopper.trigger():
            # zero optimizer gradient
            self.optim.zero_grad()

            # forward
            output = self.forward()

            # backward
            loss = self.lossfn.forward(output, self.softmax(self.H))
            loss.backward()

            # Update W and H
            self.optim.step()
            self.scheduler.step(loss)

            running_loss.append(loss.item())
            stopper.track_loss(loss)

            # print loss
            if verbose:
                print(f"epoch: {len(running_loss)}, Loss: {loss.item()}")

        W = self.softmax(self.W).detach().numpy()
        H = self.softmax(self.H).detach().numpy()

        if return_loss:
            return W, H, running_loss
        else:
            return W, H


if __name__ == "__main__":
    from helpers.callbacks import explained_variance
    nmf = NMF(X_clean, 4)
    mvr_nmf = MVR_NMF(X_clean, 4)

    nmf.fit(verbose=True)
    mvr_nmf.fit(verbose=True)

    print(f"Explained variance NMF: {explained_variance(X_clean, nmf.forward().detach().numpy())}")
    print(f"Explained variance MVR_NMF: {explained_variance(X_clean, mvr_nmf.forward().detach().numpy())}")