# ShiftNMF test
import numpy as np
import torch
import matplotlib.pyplot as plt

N, M, d = 10, 1000, 3

W = np.random.rand(N,d)
H = np.random.rand(d, M)
tau = np.random.rand(N, d)

# We need to define W^(f) for each frequency, e.g. we can represent W^(f)
# as an n x d x m matrix
Wf = np.empty((N, d, M), dtype=complex)
for f in range(M):
    omega = np.ones((N,d))*2*np.pi*f/M
    exp_tau = np.exp(-1j*omega*tau)
    Wf[:, :, f] = W*exp_tau

#%%
# Torch version
W = torch.rand(N,d)
H = torch.rand(d, M)
tau = torch.rand(N, d)

Wf = torch.empty((N, d, M), dtype=torch.cfloat)
for f in range(M):
    omega = torch.ones((N,d))*2*torch.pi*f/M
    exp_tau = torch.exp(-1j*omega*tau)
    Wf[:, :, f] = W*exp_tau
    
# Defining the loss

X = torch.rand(N, M)
ms_loss = 0
Hf = torch.fft.fft(H)
for f in range(M):
    WHt = torch.matmul(Wf[:,:, f], Hf[:, f])
    Xf = torch.fft.fft(X)[:,f]
    
    ms_loss += torch.matmul(torch.conj(Xf - WHt), Xf-WHt)

ms_loss = ms_loss.real/(2*M)