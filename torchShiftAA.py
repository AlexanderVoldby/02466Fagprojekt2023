import torch

from helpers.callbacks import earlyStop
from helpers.losses import frobeniusLoss
from helpers.losses import ShiftNMFLoss


class torchShiftAA(torch.nn.Module):
    def __init__(self, X, rank):
        super(torchShiftAA, self).__init__()

        # Shape of Matrix for reproduction
        N, M = X.shape
        self.N, self.M = N, M
        self.X = torch.tensor(X)

        # softmax layer
        self.softmax = torch.nn.Softmax(dim=0)
        self.softmax1 = torch.nn.Softmax(dim=1)
        self.softplus = torch.nn.Softplus()


        #self.lossfn = frobeniusLoss(self.X)
        #Should be the same as NMF?
        self.lossfn = ShiftNMFLoss(self.X)


        # Initialization of Tensors/Matrices S and C with size Col x Rank and Rank x Col
        # DxN (C) * NxM (X) =  DxM (CX)
        # NxD (S) *  DxM (CX) = NxM (SCX)    
        
        self.C_tilde = torch.nn.Parameter(torch.rand(rank, N, requires_grad=True))
        self.S_tilde = torch.nn.Parameter(torch.rand(N, rank, requires_grad=True))
        self.tau_tilde = torch.nn.Parameter(torch.zeros(N, rank), requires_grad=True)

        self.shift_constraint = 1000

        self.C = lambda:self.softmax(self.C_tilde)
        #self.S = lambda:self.softmax(self.S_tilde)
        self.S = lambda:self.softmax1(self.S_tilde)
        self.tau = lambda:torch.tanh(self.tau_tilde)*self.shift_constraint

    def forward(self):
        # Implementation of shift AA.
        f = torch.arange(0, self.M) / self.M
        # first matrix Multiplication
        omega = torch.exp(-1j*2 * torch.pi*torch.einsum('Nd,M->NdM', self.tau(), f))
        omega_neg = torch.exp(-1j*2 * torch.pi*torch.einsum('Nd,M->NdM', self.tau()*(-1), f))


        #data to frequency domain
        X_F = torch.fft.fft(self.X)

        #Aligned data (per component)
        X_align = torch.einsum('NM,NdM->NdM',X_F,omega_neg)

        #The A matrix, (d,M) CX, in frequency domain
        self.A = torch.einsum('dN,NdM->dM',self.C().double(), X_align.double())

        #A_F = torch.fft.fft(self.A)
        
        #S_F = torch.einsum('Nd,NdM->NdM', self.S().double(), omega)

        # archetypes back shifted
        A_shift = torch.einsum('dM,NdM->NdM', self.A.double(), omega.double())

        # Reconstruction
        self.recon= torch.einsum('Nd,NdM->NM', self.S().double(), A_shift.double())

        x = self.X - self.recon

        return x

    def fit(self, verbose=False):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.4)

        # early stopping
        es = earlyStop(patience=5, offset=-0.001)

        running_loss = []

        while (not es.trigger()):
            # zero optimizer gradient
            optimizer.zero_grad()

            # forward
            output = self.forward()

            # backward
            loss = self.lossfn.forward(output)
            loss.backward()

            # Update A and B
            optimizer.step()

            # append loss for graphing
            running_loss.append(loss.item())

            # count with early stopping
            es.count(loss.item())

            # print loss
            if verbose:
                print(f"epoch: {len(running_loss)}, Loss: {loss.item()}", end='\r')

        C, S, tau = list(self.parameters())

        C = self.softmax(C)
        S = self.softmax1(S)
        tau = torch.tanh(tau.detach()).numpy() * self.shift_constraint

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

    f = np.arange(0, M) / M
    omega = np.exp(-1j*2 * np.pi*np.einsum('Nd,M->NdM', tau, f))
    omega_neg = np.exp(-1j*2 * np.pi*np.einsum('Nd,M->NdM', tau*(-1), f))
    #Aligned data (per component)
    X_F = np.fft.fft(X)
    X_align = np.einsum('NM,NdM->NdM',X_F,omega_neg)

    #The A matrix, (d,M) CX, in frequency domain
    A = np.einsum('dN,NdM->dM',C, X_align)
    A_shift = np.einsum('dM,NdM->NdM', A, omega)

    # Reconstruction
    recon = np.einsum('Nd,NdM->NM', S, A_shift)
    
    CX = np.fft.ifft(A)
    SCX = np.fft.ifft(recon)
    print()
    plt.plot(CX[0])
    plt.plot(CX[1])
    plt.plot(CX[2])
    plt.show()

    plt.plot(X[1], color='blue')
    plt.plot(SCX[1], color='red')
    plt.show()

    # plt.plot(X[2])
    # plt.plot(SCX[2])
    # plt.show()
    
plt.figure()
plt.imshow(tau, aspect='auto', interpolation="none")
ax = plt.gca()
ax.set_xticks(np.arange(0, D, 1))
plt.colorbar()
plt.title("Tau")
plt.show()
#print(tau)