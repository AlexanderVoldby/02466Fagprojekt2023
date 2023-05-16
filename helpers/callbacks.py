import numpy as np
import torch.linalg


#Custom implementation of early stopping to allow continous running of the model until optimal paramters have been found
#this allows for an offset to be added or removed from the lowest loss to stop early when the gradient flattens
class earlyStop():
    def __init__(self, patience = 5, offset = 0) -> None:
        self.patience = patience
        
        self.counter = 0
        self.lowest = np.Inf
        
        self.offset = offset
        
    def trigger(self):
        return True if self.counter > self.patience else False

    def count(self, loss_val):
        if loss_val <= self.lowest + self.offset:
            self.lowest = loss_val
            self.counter = 0
        else:
            self.counter += 1

class RelativeStopper:

    def __init__(self, data, alpha=1e-6):
        self.norm = torch.linalg.matrix_norm(data, ord="fro").item()**2
        self.alpha = alpha
        self.loss = 1e9

    def track_loss(self, loss):
        self.loss = loss

    def trigger(self):
        return self.loss/self.norm < self.alpha


class ChangeStopper:
    def __init__(self, alpha=1e-8):
        self.alpha = alpha
        self.ploss = None
        self.loss = None

    def track_loss(self, loss):
        if self.loss is None:
            self.loss = loss

        else:
            self.ploss = self.loss
            self.loss = loss

    def trigger(self):
        if self.ploss is None:
            return False
        else:
            return abs((self.ploss - self.loss)/self.ploss) < self.alpha
