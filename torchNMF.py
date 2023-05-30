import torch
from torch.optim import Adam, lr_scheduler
from helpers.callbacks import RelativeStopper, ChangeStopper
from helpers.losses import frobeniusLoss


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

    def fit(self, verbose=False, return_loss=False):
        stopper = ChangeStopper()
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