import torch

from helpers.data import X, X_clean
from helpers.callbacks import ChangeStopper
from helpers.losses import ShiftNMFLoss
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
        self.lossfn = ShiftNMFLoss(self.X)
        
        # Initialization of Tensors/Matrices a and b with size NxR and RxM
        # Introduce regularization on W with min volume by making W have unit norm by dividing through
        # with the norm of W
        self.W = torch.nn.Parameter(torch.rand(self.N, rank, requires_grad=True))
        self.H = torch.nn.Parameter(torch.rand(rank, self.M, requires_grad=True))
        # TODO: Constrain tau by using tanh and multiplying with a max/min value
        # self.tau = torch.nn.Parameter(torch.tanh(torch.rand(self.N, self.rank) * 2000 - 1000), requires_grad=True)
        self.tau = torch.nn.Parameter(torch.rand(self.N, self.rank, requires_grad=True))
        self.optim = torch.optim.Adam(self.parameters(), lr=0.2)

    def forward(self):
        # The underlying signals in the frequency space
        Hf = torch.fft.fft(self.softplus(self.H))
        # The matrix that approximates the observations
        # Needs to be N x d x M
        f = torch.arange(0, self.M) / self.M
        omega = torch.exp(-1j*2 * torch.pi*torch.einsum('Nd,M->NdM', self.tau, f))
        Wf = torch.einsum('Nd,NdM->NdM', self.softplus(self.W), omega) 
        #W is encoding, so we put on the shift of each component.
        
        # Broadcast Wf and H together
        V = torch.einsum('NdM,dM->NM', Wf, Hf)
        self.recon = torch.fft.ifft(V)
        return V

    def fit(self, verbose=False, stopper = ChangeStopper(alpha=1e-6)):
        running_loss = []
        while not stopper.trigger():
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
            stopper.track_loss(loss.item())

            # print loss
            if verbose:
                print(f"epoch: {len(running_loss)}, Loss: {loss.item()}")

        W, H, tau = list(self.parameters())

        W = self.softplus(W).detach().numpy()
        H = self.softplus(H).detach().numpy()
        #tau = torch.tanh(tau.detach()).numpy() * self.shift_constraint
        tau = tau.detach().numpy()

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