import torch
from torch.optim import Adam, lr_scheduler
from helpers.callbacks import ChangeStopper
from helpers.losses import frobeniusLoss, VolLoss


class NMF(torch.nn.Module):
    def __init__(self, X, rank, alpha=1e-9, lr=0.5, patience=5, factor=0.9):
        super().__init__()

        n_row, n_col = X.shape
        self.softplus = torch.nn.Softplus()
        self.lossfn = frobeniusLoss(torch.tensor(X))

        # Initialization of Tensors/Matrices a and b with size NxR and RxM
        self.W = torch.nn.Parameter(torch.randn(n_row, rank, requires_grad=True)*3)
        self.H = torch.nn.Parameter(torch.randn(rank, n_col, requires_grad=True)*3)

        self.optimizer = Adam(self.parameters(), lr=lr)
        self.stopper = ChangeStopper(alpha=alpha, patience=patience+5)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=factor, patience=patience)

    def forward(self):
        WH = torch.matmul(self.softplus(self.W), self.softplus(self.H))
        return WH

    def fit(self, verbose=False, return_loss=False):
        running_loss = []
        while not self.stopper.trigger():
            # zero optimizer gradient
            self.optimizer.zero_grad()

            # forward
            output = self.forward()

            # backward

            loss = self.lossfn.forward(output)
            loss.backward()

            # Update W and H
            self.optimizer.step()
            self.scheduler.step(loss)

            running_loss.append(loss.item())
            self.stopper.track_loss(loss)

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
    def __init__(self, X, rank, regularization=1e-9, normalization=2, lr=20, alpha=1e-8, patience=5, factor=0.9):
        super().__init__()

        n_row, n_col = X.shape
        self.normalization = normalization
        self.rank = rank
        self.softmax = torch.nn.Softmax(dim=1) # dim = 1 is on the rows
        self.lossfn = VolLoss(torch.tensor(X), regularization)
        self.softplus = torch.nn.Softplus()

        # Initialization of Tensors/Matrices a and b with size NxR and RxM
        self.W = torch.nn.Parameter(torch.rand(n_row, rank, requires_grad=True))
        self.H = torch.nn.Parameter(torch.rand(rank, n_col, requires_grad=True))

        self.optim = Adam(self.parameters(), lr=lr)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', factor=factor, patience=patience)
        self.stopper = ChangeStopper(alpha=alpha, patience=patience+5)

    def forward(self):
        if self.normalization == 2:
            norm = torch.linalg.vector_norm(self.W, dim=1)
            exp_norm = norm.unsqueeze(0).expand(self.W.size(1), -1).T
            WH = torch.matmul(self.W / exp_norm, self.softplus(self.H))
        elif self.normalization == 1:
            WH = torch.matmul(self.softmax(self.W), self.softplus(self.H))
        else:
            raise ValueError(f"{self.normalization} is not a currently supported normalization technique (must be 1 or 2)")
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
                print(f"epoch: {len(running_loss)}, Loss: {loss.item()}")

        W = self.softmax(self.W).detach().numpy()
        H = self.softplus(self.H).detach().numpy()

        if return_loss:
            return W, H, running_loss
        else:
            return W, H


if __name__ == "__main__":
    from helpers.callbacks import explained_variance
    from helpers.data import X_clean
    import matplotlib.pyplot as plt
    mvr_nmf = MVR_NMF(X_clean, 6, regularization=1e-10, normalization=2)
    W, H = mvr_nmf.fit()
    print(f"Explained variance MVR_NMF: {explained_variance(X_clean, mvr_nmf.forward().detach().numpy())}")
    plt.figure()
    for vec in H:
        plt.plot(vec)
    plt.title("H")
    plt.show()