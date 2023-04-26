import torch

from helpers.data import X
from helpers.callbacks import earlyStop
from helpers.losses import ShiftNMFLoss
import matplotlib.pyplot as plt

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
        # Introduce regularization on W with min volume by making W have unit norm by dividing through
        # with the norm of W
        self.W = torch.nn.Parameter(torch.rand(self.N, rank, requires_grad=True))
        self.H = torch.nn.Parameter(torch.rand(rank, self.M, requires_grad=True))
        # TODO: Constrain tau by using tanh and multiplying with a max/min value
        # self.tau = torch.nn.Parameter(torch.tanh(torch.rand(self.N, self.rank) * 2000 - 1000), requires_grad=True)
        self.tau = torch.nn.Parameter(torch.rand(self.N, self.rank)-0.5, requires_grad=True)
        self.optim = torch.optim.Adam(self.parameters(), lr=0.1)

    def forward(self):
        Hf = torch.fft.fft(self.softplus(self.H)) # DFT on H
        f = torch.arange(0, self.M) / self.M # The frequencies
        
        tau_cons = torch.tanh(self.tau) * 5 # Constrain tau to be within +/- 100
        omega = torch.exp(-1j*2*torch.pi*torch.einsum('ab,c->abc',tau_cons, f)) # The shift operator
        Wf = torch.einsum('ab,abc->abc', self.softplus(self.W), omega) # Multiply shift operator into W
        V = torch.einsum('abc,bc->ac', Wf, Hf) # Multiply Wf and Hf
        return V

    def fit(self, verbose=False):
        es = earlyStop(patience=5, offset=-0.01)
        running_loss = []
        while not es.trigger():
            # zero optimizer gradient
            self.optim.zero_grad()

            # forward
            output = self.forward()

            # backward
            loss = self.lossfn(output)
            loss.backward()

            # Update W, H and tau
            print()
            # TODO: Or instead constrain tau to make shifts within +/- 1000 when we take the step
            self.optim.step()

            running_loss.append(loss.item())
            es.count(loss.item())

            # print loss
            if verbose:
                print(f"epoch: {len(running_loss)}, Loss: {loss.item()}")

        W, H, tau = list(self.parameters())

        W = self.softplus(W).detach().numpy()
        H = self.softplus(H).detach().numpy()
        tau = torch.tanh(tau.detach()).numpy() * 5
        # tau = tau.detach().numpy()

        return W, H, tau


if __name__ == "__main__":
    nmf = ShiftNMF(X, 2)
    nmf.to(device)
    W, H, tau = nmf.fit(verbose=True)

# plt.figure()
# for signal in H:
#     plt.plot(signal)
# plt.title("H - the latent variables")
# plt.show()


plt.figure()
for signal in X:
    plt.plot(signal)
# plt.show()


# import numpy as np
X_rec = W @ H
plt.figure()
for signal in X_rec:
    plt.plot(signal)
plt.show()


# plt.figure()
# plt.imshow(W)
# plt.colorbar()
# plt.title("W - The mixings")
# plt.show()

plt.figure()
plt.imshow(tau)
plt.colorbar()
plt.title("Tau")
plt.show()