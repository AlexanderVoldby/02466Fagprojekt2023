import torch
from torch.optim import Adam, lr_scheduler
from helpers.callbacks import ChangeStopper
from helpers.losses import frobeniusLoss
from helpers.losses import ShiftNMFLoss


class torchShiftAA(torch.nn.Module):
    def __init__(self, X, rank, shift_constraint = 100):
        super(torchShiftAA, self).__init__()

        self.shift_constraint = shift_constraint
        # Shape of Matrix for reproduction
        N, M = X.shape
        self.N, self.M = N, M
        self.X = torch.tensor(X)

        # softmax layer
        self.softmax = torch.nn.Softmax(dim=1)
        self.softplus = torch.nn.Softplus()


        # self.lossfn = frobeniusLoss(torch.fft.fft(self.X))
        # Should be the same as NMF?
        self.lossfn = ShiftNMFLoss(torch.fft.fft(self.X))


        # Initialization of Tensors/Matrices S and C with size Col x Rank and Rank x Col
        # DxN (C) * NxM (X) =  DxM (A)
        # NxD (S) *  DxM (A) = NxM (SA)    
        
        self.C_tilde = torch.nn.Parameter(torch.rand(rank, N, requires_grad=True))
        self.S_tilde = torch.nn.Parameter(torch.rand(N, rank, requires_grad=True))
        self.tau_tilde = torch.nn.Parameter(torch.zeros(N, rank), requires_grad=True)

        self.C = lambda:self.softmax(self.C_tilde)
        self.S = lambda:self.softmax(self.S_tilde)
        self.tau = lambda:torch.tanh(self.tau_tilde)*self.shift_constraint

    # def tau(self):
    #     return torch.zeros(N, rank)

    def forward(self):
        # Implementation of shift AA.
        f = torch.arange(0, self.M) / self.M
        # first matrix Multiplication
        omega = torch.exp(-1j * 2 * torch.pi*torch.einsum('Nd,M->NdM', self.tau(), f))
        # omega_neg = torch.exp(-1j*2 * torch.pi*torch.einsum('Nd,M->NdM', self.tau()*(-1), f))
        omega_neg = torch.conj(omega)

        #data to frequency domain
        X_F = torch.fft.fft(self.X)

        #Aligned data (per component)
        X_F_align = torch.einsum('NM,NdM->NdM',X_F, omega_neg)
        X_align = torch.fft.ifft(X_F_align)
        #The A matrix, (d,M) A, in frequency domain
        self.A = torch.einsum('dN,NdM->dM', self.C().double(), X_align.double())
        A_F = torch.fft.fft(self.A)
        #self.A_F = torch.einsum('dN,NdM->dM',self.C().double(), X_F_align.double())
        #S_F = torch.einsum('Nd,NdM->NdM', self.S().double(), omega)

        # archetypes back shifted
        #A_shift = torch.einsum('dM,NdM->NdM', self.A_F.double(), omega.double())
        S_shift = torch.einsum('Nd,NdM->NdM', self.S(), omega) 

        # Reconstruction
        x = torch.einsum('NdM,dM->NM', S_shift.double(), A_F.double())
        self.recon = torch.fft.ifft(x)
        return x

    def fit(self, verbose=False, return_loss=False, stopper = ChangeStopper(alpha=1/1000)):
        stopper.reset()
        optimizer = Adam(self.parameters(), lr=0.4)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        # Convergence criteria
        running_loss = []

        while not stopper.trigger():
            # zero optimizer gradient
            optimizer.zero_grad()

            # forward
            output = self.forward()

            # backward
            loss = self.lossfn.forward(output)
            loss.backward()

            # Update A and B
            optimizer.step()
            scheduler.step(loss)
            # append loss for graphing
            running_loss.append(loss.item())

            # count with early stopping
            stopper.track_loss(loss)

            # print loss
            if verbose:
                print(f"epoch: {len(running_loss)}, Loss: {loss.item()}")

        C = self.softmax(self.C_tilde)
        S = self.softmax(self.S_tilde)
        tau = torch.tanh(self.tau_tilde.detach()).numpy() * self.shift_constraint

        C = C.detach().numpy()
        S = S.detach().numpy()

        return C, S, tau

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.io
    mat = scipy.io.loadmat('helpers/data/NMR_mix_DoE.mat')

    #Get X and Labels. Probably different for the other dataset, but i didn't check :)
    X = mat.get('xData')
    N, M = X.shape
    rank = 3
    D = rank
    AA = torchShiftAA(X, rank)
    print("test")
    C,S, tau = AA.fit(verbose=True)

    recon = AA.recon.detach().resolve_conj().numpy()
    A = AA.A.detach().numpy()

    # For visualizing the aligned data
    X_torch = torch.Tensor(X)
    f = torch.arange(0, M) / M
    omega = torch.exp(-1j * 2 * torch.pi * torch.einsum('Nd,M->NdM', AA.tau(), f))
    omega_neg = torch.conj(omega)
    X_F = torch.fft.fft(X_torch)
    X_F_align = torch.einsum('NM,NdM->NdM', X_F, omega_neg)
    X_align = torch.fft.ifft(X_F_align).detach().numpy()

    plt.figure()
    plt.plot(X[0], label="First signal")
    plt.plot(X_align[:,0,:][0], label="Data aligned with 1st archetype (1st signal)")
    plt.legend()
    plt.show()

    
    plt.figure()
    for arc in A:
        plt.plot(arc)
    plt.title("Archetypes")
    plt.show()
    plt.figure()
    plt.plot(X[1], label="First signal of X")
    plt.plot(recon[1], label="Reconstructed signal with shift AA")
    plt.legend()
    plt.show()

    plt.figure()
    plt.imshow(tau, aspect='auto', interpolation="none")
    ax = plt.gca()
    ax.set_xticks(np.arange(0, D, 1))
    plt.colorbar()
    plt.title("Tau")
    plt.show()
    #print(tau)