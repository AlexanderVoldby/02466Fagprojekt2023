import torch
from torch.optim import Adam, lr_scheduler, SGD
from helpers.callbacks import ChangeStopper
from helpers.losses import frobeniusLoss
from helpers.losses import ShiftNMFLoss
import helpers.initializers as init
import numpy as np

import scipy.io
import time

class torchShiftAADisc(torch.nn.Module):
    def __init__(self, X, rank, alpha=1e-9, lr = 10, factor = 0.9, patience = 5, fs_init = False):
        super(torchShiftAADisc, self).__init__()

        # Shape of Matrix for reproduction
        N, M = X.shape
        self.N, self.M = N, M
        self.X = torch.tensor(X)

        # softmax layer
        self.softmax = torch.nn.Softmax(dim=1)
        self.softplus = torch.nn.Softplus()


        # self.lossfn = frobeniusLoss(torch.fft.fft(self.X))
        # Should be the same as NMF?
        self.lossfn = frobeniusLoss(torch.fft.fft(self.X))

        # Initialization of Tensors/Matrices S and C with size Col x Rank and Rank x Col
        # DxN (C) * NxM (X) =  DxM (A)
        # NxD (S) *  DxM (A) = NxM (SA)
        if fs_init:
            # S, C = init.fit_s(self.X, C, epochs=50)
            C, S  = init.init_C_S(X, rank, epochs=50, return_tilde=True)
            
            self.C_tilde = torch.nn.Parameter(torch.tensor(C, requires_grad=True, dtype=torch.double))
            self.S_tilde = torch.nn.Parameter(torch.tensor(S, requires_grad=True, dtype=torch.double))
            
        else:
            self.C_tilde = torch.nn.Parameter(torch.randn(rank, N, requires_grad=True,dtype=torch.double))
            self.S_tilde = torch.nn.Parameter(torch.randn(N, rank, requires_grad=True, dtype=torch.double))
        
        
        # mat = scipy.io.loadmat('helpers/PCHA/C.mat')
        # self.C_tilde = mat.get('c')
        # self.C_tilde = torch.tensor(self.C_tilde, requires_grad=True, dtype=torch.double)
        # self.C_tilde = torch.nn.Parameter(self.C_tilde.T)
        
        # mat = scipy.io.loadmat('helpers/PCHA/S.mat')
        # self.S_tilde = mat.get('s')
        # self.S_tilde = torch.tensor(self.S_tilde, requires_grad=True, dtype=torch.double)
        # self.S_tilde = torch.nn.Parameter(self.S_tilde.T)
        
        self.tau_tilde = torch.nn.Parameter(torch.zeros(N, rank, requires_grad=False, dtype=torch.double))

        self.C = lambda:self.softmax(self.C_tilde).type(torch.cdouble)
        self.S = lambda:self.softmax(self.S_tilde).type(torch.cdouble)
        #self.tau = lambda:torch.tanh(self.tau_tilde)*self.shift_constraint
        # self.tau = lambda: torch.round(self.tau_tilde)
        self.tau = lambda: self.tau_tilde

        self.optimizer = Adam(self.parameters(), lr=lr)
        # self.optimizer = SGD(self.parameters(), lr=lr)
        self.stopper = ChangeStopper(alpha=alpha, patience=patience)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=factor, patience=patience-2)

    
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
        #X_align = torch.fft.ifft(X_F_align)
        #The A matrix, (d,M) A, in frequency domain
        #self.A = torch.einsum('dN,NdM->dM', self.C(), X_align)
        #A_F = torch.fft.fft(self.A)
        self.A_F = torch.einsum('dN,NdM->dM',self.C(), X_F_align)
        #S_F = torch.einsum('Nd,NdM->NdM', self.S().double(), omega)

        # archetypes back shifted
        #A_shift = torch.einsum('dM,NdM->NdM', self.A_F.double(), omega.double())
        self.S_shift = torch.einsum('Nd,NdM->NdM', self.S(), omega) 

        # Reconstruction
        x = torch.einsum('NdM,dM->NM', self.S_shift, self.A_F)
        
        return x

    def fit(self, verbose=False, return_loss=False, max_iter=1e10):
        self.stopper.reset()
        # Convergence criteria
        running_loss = []
        iters = 0
        while not self.stopper.trigger() and iters < max_iter:
            iters += 1
            # zero optimizer gradient
            self.optimizer.zero_grad()

            # forward
            output = self.forward()

            # backward
            loss = self.lossfn.forward(output)
            loss.backward()

            # self.tau_tilde.grad = self.tau_tilde.grad
            # print(torch.sign(self.tau_tilde.grad))
            # self.tau_tilde.grad = torch.sign(self.tau_tilde.grad)
            
            # print("tau: ", self.tau_tilde.grad)
            change = torch.sign(self.tau_tilde.grad)
            #set gradient 0 - possibly not needed since tau tilde is overwritten
            self.tau_tilde.grad = self.tau_tilde.grad * 0
            #update tau
            self.tau_tilde = torch.nn.Parameter(self.tau_tilde + change)
            
            # self.tau_tilde = torch.nn.Parameter(self.tau_tilde + torch.sign(self.tau_tilde.grad.clone()))
            # self.tau_tilde.grad = self.tau_tilde.grad * 0
            
            #divide tau by the learning rate in the optimizer
            # print(self.optimizer.param_groups[0].get('lr'))
            # exit()
            
            
            #round the gradient of tau to the nearest integer
            # print("loss gradient: ", self.tau_tilde.grad)
            # self.tau_tilde.grad = torch.round(self.tau_tilde.grad)
            
            # self.tau_tilde.grad = self.tau_tilde.grad / self.optimizer.param_groups[0].get('lr')
            
            
            
            #print the gradient of the loss function
            
            
            # Update parameters
            self.optimizer.step()
            self.scheduler.step(loss)
            
            #round tau to the nearest integer
            # self.tau_tilde = torch.nn.Parameter(torch.round(self.tau_tilde))
            
            # append loss for graphing
            running_loss.append(loss.item())

            # count with early stopping
            self.stopper.track_loss(loss)
            self.stopper.track_loss(loss)

            # print loss
            if verbose:
                print(f"epoch: {len(running_loss)}, Loss: {1-loss.item()}, Tau: {np.linalg.norm(self.tau().detach().numpy())}", end="\r")
        
        C = self.softmax(self.C_tilde)
        S = self.softmax(self.S_tilde)
        tau = self.tau().detach().numpy() 

        C = C.detach().numpy()
        S = S.detach().numpy()
        #self.tau = lambda: torch.round(self.tau_tilde)
        output = self.forward()
        self.recon = torch.fft.ifft(output)
        if return_loss:
            return C, S, tau, running_loss
        else:
            return C, S, tau


if __name__ == "__main__":
    # import numpy as np
    import matplotlib.pyplot as plt
    import scipy.io
    mat = scipy.io.loadmat('helpers/data/NMR_mix_DoE.mat')

    #Get X and Labels. Probably different for the other dataset, but i didn't check :)
    X = mat.get('xData')
    X = X[:10]
    N, M = X.shape
    rank = 3
    D = rank
    AA = torchShiftAADisc(X, rank, lr=0.3)
    print("test")
    C,S, tau = AA.fit(verbose=True, max_iter=100)

    recon = AA.recon.detach().resolve_conj().numpy()
    A = torch.fft.ifft(AA.A_F).detach().numpy()

    # # For visualizing the aligned data
    # X_torch = torch.Tensor(X)
    # f = torch.arange(0, M) / M
    # omega = torch.exp(-1j * 2 * torch.pi * torch.einsum('Nd,M->NdM', AA.tau(), f))
    # omega_neg = torch.conj(omega)
    # X_F = torch.fft.fft(X_torch)
    # X_F_align = torch.einsum('NM,NdM->NdM', X_F, omega_neg)
    # X_align = torch.fft.ifft(X_F_align).detach().numpy()

    # plt.figure()
    # plt.plot(X[0], label="First signal")
    # plt.plot(X_align[:,0,:][0], label="Data aligned with 1st archetype (1st signal)")
    # plt.legend()
    # plt.show()

    
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