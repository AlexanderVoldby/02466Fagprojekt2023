import torch
from torch.optim import Adam, lr_scheduler
from helpers.data import X, X_clean
from helpers.callbacks import ChangeStopper
from helpers.losses import ShiftNMFLoss
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class ShiftNMF(torch.nn.Module):
    def __init__(self, X, rank, alpha=1e-8, shift_init = 5000):
        super().__init__()

        # Shape of Matrix for reproduction
        self.rank = rank
        self.X = torch.tensor(X)
        self.shift_init = shift_init
        
        self.N, self.M = X.shape
        self.softplus = torch.nn.Softplus()
        self.lossfn = ShiftNMFLoss(torch.fft.fft(self.X))
        self.stopper = ChangeStopper(alpha=alpha)
        
        # Initialization of Tensors/Matrices a and b with size NxR and RxM
        # Introduce regularization on W with min volume by making W have unit norm by dividing through
        # with the norm of W
        self.W = torch.nn.Parameter(torch.rand(self.N, rank, requires_grad=True))
        self.H = torch.nn.Parameter(torch.rand(rank, self.M, requires_grad=True))
        # Init tau between -1 and 1
        self.tau = torch.nn.Parameter(-2*self.shift_init * torch.rand(self.N, self.rank)+self.shift_init, requires_grad=True)
        # Tau is then cast to [-shift_constraint, shift_constraint]
        # self.tau = lambda: torch.tanh(self.tau_tilde) * self.shift_constraint
        self.optim = Adam(self.parameters(), lr=0.3)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', factor=0.1, patience=5)

    def forward(self):
        # Get half of the frequencies
        # TODO: Sørg for at det virker for både lige og ulige data
        Nf = self.M // 2 + 1
        # Fourier transform of H along the second dimension
        Hf = torch.fft.fft(self.softplus(self.H), dim=1)
        # Keep only the first Nf[1] elements of the Fourier transform of H
        Hf = Hf[:, :Nf]
        # Construct the shifted Fourier transform of H
        Hf_reverse = torch.flip(Hf[:, 1:Nf-1], dims=[1])
        # Concatenate the original columns with the reversed columns along the second dimension
        Hft = torch.cat((Hf, torch.conj(Hf_reverse)), dim=1)
        f = torch.arange(0, self.M) / self.M
        omega = torch.exp(-1j * 2 * torch.pi * torch.einsum('Nd,M->NdM', self.tau, f))
        Wf = torch.einsum('Nd,NdM->NdM', self.softplus(self.W), omega)
        # Broadcast Wf and H together
        V = torch.einsum('NdM,dM->NM', Wf, Hft)
        return V

    def fit(self, verbose=False, return_loss=False):
        running_loss = []
        while not self.stopper.trigger():
            # zero optimizer gradient
            self.optim.zero_grad()

            # forward
            output = self.forward()

            # backward
            loss = self.lossfn(output)
            loss.backward()

            # Update W, H and tau
            self.optim.step()
            self.scheduler.step(loss)
            running_loss.append(loss.item())
            self.stopper.track_loss(loss)

            # print loss
            if verbose:
                print(f"epoch: {len(running_loss)}, Loss: {loss.item()}")

        W = self.softplus(self.W).detach().numpy()
        H = self.softplus(self.H).detach().numpy()
        tau = self.tau.detach().numpy()

        output = self.forward()
        self.recon = torch.fft.ifft(output)

        if return_loss:
            return W, H, tau, running_loss
        else:
            return W, H, tau


if __name__ == "__main__":
    nmf = ShiftNMF(X_clean, 4, shift_init=5)
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