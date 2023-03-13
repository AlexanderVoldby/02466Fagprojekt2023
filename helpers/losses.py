import torch

#Defining frobenius Loss
class frobeniusLoss(torch.nn.Module):
    def __init__(self, X):
        super(frobeniusLoss, self).__init__()
        self.loss = torch.linalg.matrix_norm
        self.X = torch.tensor(X)
    
    def forward(self, input):
        return self.loss(self.X - input, ord='fro')