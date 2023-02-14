import numpy as np

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