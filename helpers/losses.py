import torch

#Defining frobenius Loss
class frobeniusLoss(torch.nn.Module):
    def __init__(self):
        super(frobeniusLoss, self).__init__()
        self.loss = torch.linalg.matrix_norm
    
    def forward(self, input):
        return self.loss(input, ord='fro')