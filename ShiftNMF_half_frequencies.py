import torch

from helpers.data import X, X_clean
from helpers.callbacks import earlyStop
from helpers.losses import ShiftNMFLoss_halff
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class ShiftNMF(torch.nn.Module):
    def __init__(self, X, rank, shift_constraint = 5):
        super().__init__()

        # Shape of Matrix for reproduction
        self.rank = rank
        self.X = torch.tensor(X)
        self.shift_constraint = shift_constraint
        
        self.N, self.M = X.shape
        self.softplus = torch.nn.Softplus()
        self.lossfn = ShiftNMFLoss_halff(self.X)
        
        # Initialization of Tensors/Matrices a and b with size NxR and RxM
        # Introduce regularization on W with min volume by making W have unit norm by dividing through
        # with the norm of W
        self.W = torch.nn.Parameter(torch.rand(self.N, rank, requires_grad=True))
        self.H = torch.nn.Parameter(torch.rand(rank, self.M, requires_grad=True))
        # TODO: Constrain tau by using tanh and multiplying with a max/min value
        # self.tau = torch.nn.Parameter(torch.tanh(torch.rand(self.N, self.rank) * 2000 - 1000), requires_grad=True)
        self.tau = torch.nn.Parameter(torch.rand(self.N, self.rank)*2*self.shift_constraint-self.shift_constraint, requires_grad=True)
        self.optim = torch.optim.Adam(self.parameters(), lr=0.2)

    def forward(self):
        Xf = torch.fft.fft(self.X, dim=1)
        # Keep only the first half of the Fourier transform (due to symmetry)
        Xf = Xf[:, :(Xf.shape[1] // 2) + 1]
        # Get the size of Xf
        Nf = Xf.shape
        # Fourier transform of H along the second dimension
        Hf = torch.fft.fft(self.H, dim=1)
        # Keep only the first Nf[1] elements of the Fourier transform of H
        Hf = Hf[:, :Nf[1]]
        # Construct the shifted Fourier transform of H
        Hf_reverse = torch.flip(Hf[:, 1:Nf[1]-1], dims=[1])
        # Concatenate the original columns with the reversed columns along the second dimension
        Hft = torch.cat((Hf, torch.conj(Hf_reverse)), dim=1)
        f = torch.arange(0, self.M) / self.M
        omega = torch.exp(-1j*2 * torch.pi*torch.einsum('Nd,M->NdM', self.tau, f))
        Wf = torch.einsum('Nd,NdM->NdM', self.softplus(self.W), omega)
        # Broadcast Wf and H together
        V = torch.einsum('NdM,dM->NM', Wf, Hft)
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
            self.optim.step()

            running_loss.append(loss.item())
            es.count(loss.item())

            # print loss
            if verbose:
                print(f"epoch: {len(running_loss)}, Loss: {loss.item()}")

        W, H, tau = list(self.parameters())

        W = self.softplus(W).detach().numpy()
        H = self.softplus(H).detach().numpy()
        tau = torch.tanh(tau.detach()).numpy() * self.shift_constraint
        # tau = tau.detach().numpy()

        return W, H, tau


if __name__ == "__main__":
    nmf = ShiftNMF(X_clean, 4, shift_constraint=5)
    nmf.to(device)
    W, H, tau = nmf.fit(verbose=True)

    plt.figure()
    for signal in H:
        plt.plot(signal)
    plt.title("H - the latent variables")
    plt.show()


    plt.figure()
    plt.imshow(W)
    plt.colorbar()
    plt.title("W - The mixings")
    plt.show()

    plt.figure()
    plt.imshow(tau)
    plt.colorbar()
    plt.title("Tau")
    plt.show()