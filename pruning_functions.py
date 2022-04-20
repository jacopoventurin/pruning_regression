import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
from Spectral_Layer import Spectral
from tqdm import trange
from time import sleep

class BaseModel(nn.Module):
    """
    Base model to implement using specific layer and with a given forward module. The loss function is F.mse_loss 
    
    """
    def __init__(self):
        super().__init__()
        
    def training_step(self, batch):
        x, y = batch 
        out = self(x)                  # Generate predictions
        loss = F.mse_loss(out, y) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        x, y= batch 
        out = self(x)                    # Generate predictions
        loss = F.mse_loss(out, y)   # Calculate loss as Mean Squared Error
        mae = F.l1_loss(out, y)     # Calculate Mean Absolute Error
        rmse = torch.sqrt(F.mse_loss(out, y))    # Calculate Root Mean Squared Error
        return {'val_loss': loss, 'val_MAE': mae, 'val_RMSE': rmse}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_mae = [x['val_MAE'] for x in outputs]
        epoch_mae = torch.stack(batch_mae).mean()      # Combine Mean Absolute Error
        batch_rmse = [x['val_RMSE'] for x in outputs]
        epoch_rmse = torch.stack(batch_rmse).mean()      # Combine Root Mean Squared Error
        
        return {'val_loss': epoch_loss.item(), 'val_MAE': epoch_mae.item(), 'val_RMSE': epoch_rmse.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.6f}, val_MAE: {:.6f}, val_RMSE: {:.6}".format(epoch, result['val_loss'], result['val_MAE'], result['val_RMSE']))
        

        
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
    history = [] # for recording epoch-wise results
    
    with trange(epochs, desc='Progress', unit='epochs') as tepoch:
        for epoch in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            # Training Phase 
            for batch in train_loader:
                loss = model.training_step(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            # Validation phase
            result = evaluate(model, val_loader)
            history.append(result)
            
            tepoch.set_postfix(loss=loss.item())
        
        # Print final result
        model.epoch_end(epoch, result)
            
    return history, result


class LinearForPruning(nn.Module):
    def __init__(self, 
                 in_dim: int,
                 out_dim: int,
                 bias: bool = False,
                 device=None, 
                 dtype=torch.double):
        
        super().__init__()
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias
        
        self.weight = nn.Parameter(torch.rand(self.out_dim, self.in_dim, **factory_kwargs) - 0.5, requires_grad=True)
        if self.bias:
            self.Bias = nn.Parameter(torch.rand(self.outdim, **factory_kwargs) - 0.5, requires_grad=True)
        
    def forward(self, x):
        if self.bias:
            return F.linear(x, self.weight, self.Bias)
        else:
            return F.linear(x, self.weight)
    
    
    
    def prune(self,
             perc: float,
             grad: bool = True):
        """
        This method apply a pruning procedure, and return new pruned layer
        
        Parameters
        ----------
        perc:
            Percentile of elements we want to prune
        grad:
            If set to True the new layer is trainable
            
        Example:
        --------
        model = LinearForPrune(20, 20)
        model1 = model.prune(50)
            
        """
        m = self.weight.detach().numpy()    
        abs_m = np.abs(m)
        threshold = np.percentile(abs_m, perc)  # Find the point where to cut 
        m[abs_m < threshold] = 0.0  # Pruning: put 0 elements under threshold
       
        self.weight = nn.Parameter(torch.from_numpy(m), requires_grad=grad)
        
        
        