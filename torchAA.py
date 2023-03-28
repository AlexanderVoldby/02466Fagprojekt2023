import torch
import numpy as np
import pandas as pd

from helpers.data import X
from helpers.callbacks import earlyStop
from helpers.losses import frobeniusLoss

import matplotlib.pyplot as plt


class torchAA(torch.nn.Module):
    def __init__(self, X, rank):
        super(torchAA, self).__init__()

        # Shape of Matrix for reproduction
        n_row, n_col = X.shape
        self.X = torch.tensor(X)

        # softmax layer
        self.softmax = torch.nn.Softmax(dim=1)

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


aa = torchAA(X, 3)

# optimizer for modifying learning rate, ADAM chosen because of https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
optimizer = torch.optim.Adam(aa.parameters(), lr=0.3)

# early stopping
es = earlyStop(patience=5, offset=-0.1)

running_loss = []

while (not es.trigger()):
    # zero optimizer gradient
    optimizer.zero_grad()

    # forward
    output = aa()

    # backward
    loss = frobeniusLoss()
    loss = loss.forward(output)
    loss.backward()

    # Update A and B
    optimizer.step()

    # append loss for graphing
    running_loss.append(loss.item())

    # count with early stopping
    es.count(loss.item())

    # print loss
    print(f"epoch: {len(running_loss)}, Loss: {loss.item()}", end='\r')

plt.plot(running_loss)
plt.show()
# print(list(aa.parameters())[1].shape)

C, S = list(aa.parameters())

C = aa.softmax(C)
S = aa.softmax(S)

C = C.detach().numpy()
S = S.detach().numpy()

rec = np.dot(np.dot(X, C),
             S)

rec = rec.T

rec_frame = pd.DataFrame(rec)
rec_frame.columns = rec_frame.columns.astype(str)

rec_frame.to_parquet("recons_x.parquet",
                     engine='fastparquet')