import torch
import numpy as np
import pandas as pd

from helpers.data import X
from helpers.callbacks import earlyStop
from helpers.losses import VolLoss


class TorchNMF_MinVol(torch.nn.Module):
    def __init__(self, X, rank):
        super(TorchNMF_MinVol, self).__init__()

        # Shape of Matrix for reproduction
        n_row, n_col = X.shape
        self.X = torch.tensor(X)

        self.softmax = torch.nn.Softmax(dim=0)
        self.softplus = torch.nn.Softplus()

        # Initialization of Tensors/Matrices a and b with size NxR and RxM
        # W is the basis matrix
        self.W = torch.nn.Parameter(torch.rand(n_row, rank, requires_grad=True))
        # H is the encoding matrix
        self.H = torch.nn.Parameter(torch.rand(rank, n_col, requires_grad=True))

    def forward(self):
        # Implementation of NMF - F(W, H) = ||X - WH||^2
        self.WH = torch.matmul(self.softmax(self.W), self.softplus(self.H))
        x = self.WH

        return self.softmax(self.W), self.softplus(self.H), x

    def run(self, verbose=False):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.03)

        # early stopping
        es = earlyStop(patience=10, offset=-0.0001)

        running_loss = []

        while (not es.trigger()):
            # zero optimizer gradient
            optimizer.zero_grad()

            # forward
            w_out, h_out, x_out = self.forward()
            # backward
            loss = VolLoss(self.X)
            loss = loss.forward(w_out, h_out, x_out)
            loss.backward()

            # Update A and B
            optimizer.step()

            running_loss.append(loss.item())
            es.count(loss.item())

            # print loss
            if verbose:
                print(f"epoch: {len(running_loss)}, Loss: {loss.item()}", end='\r')

        if verbose:
            print(f"Final loss: {loss.item()}")

        W = self.softmax(self.W)
        H = self.softplus(self.H)

        return W.detach().numpy(), H.detach().numpy()