import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Spectral(nn.Module):
    
    """Spectral layer base model
    
    Base Spectral layer model as presented in https://journals.aps.org/pre/abstract/10.1103/PhysRevE.104.054312, implemented using PyTorch-based code. Here eigenvectors and eigenvalues are sampled from a uniform distribution from -0.5 to 0.5.
     
    Parameters
    ----------
    in_dim:
        Input size
    out_dim:
        Output size 
    base_grad:
        If set to True the eigenvectors are trainable
    start_grad:
        If set to True the starting eigenvalues are trainable
    end_grad:
        If set to True the ending eigenvalues are trainable
    bias:
        If set to True add bias 
    device:
        Device for training
    dtype:
        Type for the training parameters
        
        
    Example
    -------
    model = torch.nn.Sequential(
                            Spectral(1, 20),
                            Spectral(20,20),
                            F.elu()
                            spectral(20,1)
                            )
    """

    __constants__ = ['in_dim', 'out_dim']
    in_dim: int
    out_dim: int
        
        
    def __init__(self, 
                 in_dim: int, 
                 out_dim: int,
                 base_grad: bool = True,  
                 start_grad: bool = True,
                 end_grad: bool = True, 
                 bias: bool = False,
                 device=None, 
                 dtype=torch.double):
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        super(Spectral, self).__init__()
        self.in_dim = in_dim 
        self.out_dim = out_dim 
        self.base_grad = base_grad
        self.start_grad = start_grad
        self.end_grad = end_grad
        self.bias = bias
        
        
        # Build the model 
        
        # Eigenvectors
        self.base = nn.Parameter(torch.rand(self.in_dim, self.out_dim, **factory_kwargs) - 0.5, requires_grad=self.base_grad)
        # Eigenvalues start 
        self.diag_start = nn.Parameter(torch.rand(self.in_dim, 1, **factory_kwargs) - 0.5,  requires_grad=self.start_grad)
        # Eigenvalues end
        self.diag_end = nn.Parameter(torch.rand(1, self.out_dim, **factory_kwargs) - 0.5, requires_grad=self.end_grad)
        # Bias
        if self.bias:
            self.Bias = nn.Parameter(torch.rand(self.outdim, **factory_kwargs) - 0.5, requires_grad=True)
    
        
    def forward(self, x):
        kernel = torch.mul(self.base, self.diag_start - self.diag_end)
        if self.bias:
            outputs = F.linear(x, kernel.t(), self.Bias)
        else:
            outputs = F.linear(x, kernel.t())
        
        return outputs
    
    
    def direct_space(self):
        return torch.mul(self.base, self.diag_start - self.diag_end).detach().numpy().T

    def return_base(self):
        c = self.base.shape[0]
        N = reduce_sum(self.base.shape).detach().numpy()
        phi = np.eye(N)
        phi[c:, :c] = self.base.detach().numpy().T
        return phi

    def return_diag(self):
        d = np.concatenate([self.diag_start.detach().numpy()[:, 0], self.diag_end.detach().numpy()[0, :]], axis=0)
        return d
    
    def pruning_diag(self, 
                     perc: float, 
                     start_grad: bool = False, 
                     end_grad: bool = False, 
                     base_grad: bool = True):
        """
        This method apply a pruning procedure, acting directly on the elements in the diagonal of this class 
        
        Parameters
        ----------
        perc:
            Percentile of elements in the diagonal that we want to prune
        start_grad:
            If set to True self.diag_start is trainable
        end_grad:
            If set to True self.diag_end is trainable
        base_grad:
            if Set to True self.base is trainable
            
        Example:
        --------
        model = Spectral(20, 20)
        model.pruning_diag(50)
            
        """
        d = self.return_diag()
        abs_d = np.abs(d)
        threshold = np.percentile(abs_d, perc)  # Find the point where to cut 
        d[abs_d < threshold] = 0.0  # Pruning: put 0 elements under threshold
        # New diag_start
        diag_start = d[:self.in_dim]
        diag_start = np.reshape(diag_start, (self.in_dim, 1))
        self.diag_start = nn.Parameter(torch.from_numpy(diag_start), requires_grad=start_grad)
        # New diag_end
        diag_end = d[self.in_dim:]
        diag_end = np.reshape(diag_end, (1, self.out_dim))
        self.diag_end = nn.Parameter(torch.from_numpy(diag_end), requires_grad=end_grad)
        # Eventually change base requires_grad
        self.base.requires_grad_(requires_grad=base_grad)
        
    
    
    
    def pruning_diag_new(self,
                         perc: float, 
                         start_grad: bool = False, 
                         end_grad: bool = False, 
                         base_grad: bool = True):
        """
        This method apply a pruning procedure acting on the direct space, and return new pruned layer
        
        Parameters
        ----------
        perc:
            Percentile of elements in the diagonal that we want to prune
        start_grad:
            If set to True self.diag_start is trainable
        end_grad:
            If set to True self.diag_end is trainable
        base_grad:
            if Set to True self.base is trainable
            
        Example:
        --------
        model = Spectral(20, 20)
        model1 = model.pruning_diag_new(50)
            
        """
        d = self.return_diag()
        abs_d = np.abs(d)
        threshold = np.percentile(abs_d, perc)  # Find the point where to cut 
        d[abs_d < threshold] = 0.0  # Pruning: put 0 elements under threshold
        # New diag_start
        diag_start = d[:self.in_dim]
        diag_start = np.reshape(diag_start, (self.in_dim, 1))
        diag_start = torch.from_numpy(diag_start)
        # New diag_end
        diag_end = d[self.in_dim:]
        diag_end = np.reshape(diag_end, (1, self.out_dim))
        diag_end = torch.from_numpy(diag_end)
        # Base
        base = self.base
        # Return new layer
        return NewSpectral(diag_start, diag_end, base, start_grad, end_grad, base_grad)
    
    
    def pruning_direct_space_new(self,
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
        model = Spectral(20, 20)
        model1 = model.pruning_diag_new(50)
            
        """
        m = self.direct_space()
        abs_m = np.abs(m)
        threshold = np.percentile(abs_m, perc)  # Find the point where to cut 
        m[abs_m < threshold] = 0.0  # Pruning: put 0 elements under threshold
        # New block
        m = np.reshape(m, (self.in_dim, self.out_dim))
        m = torch.from_numpy(m)
        # Return new layer
        return NewClassical(m, grad)
        
        
        
class NewSpectral(nn.Module):
    """
    Spectral layer model with eigenvectors and eigenvalues taken as imput. With this class it is not possible to add a baias 
    during the learning procedure, otherwise the pruning procedure shouldn't work
    
    Parameters
    ----------
    diag_start:
        Imput diagonal, expected torch tensor with shape (input_dimension, 1)
    diag_end:
        Output diagonal, expected torch tensor with shape (1, output_dimension)
    base:
        Eigenvectors base, expected torch tensor with shape (input_dimension, output_dimension)
    start_grad:
        If set to True the starting eigenvalues are trainable
    end_grad:
        If set to True the ending eigenvalues are trainable
    base_grad:
        If set to True the eigenvectors are trainable
    device:
        Device for training
    dtype:
        Type for the training parameters
        
    Example:
    --------
    diag_start = torch.rand(20,1) - 0.5
    diag_end = torch.rand(1,20) - 0.5
    base = torch.rand(20,20) - 0.5
    model = NewSpectral(diag_start, diag_end, base)
    
    """
    def __init__(self,
                 diag_start: torch.double,
                 diag_end: torch.double,
                 base: torch.double, 
                 start_grad: bool = False,
                 end_grad: bool = False,
                 base_grad: bool = True,
                 device=None, 
                 dtype=torch.double):
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        super(NewSpectral, self).__init__()
        self.diag_start = diag_start
        self.diag_end = diag_end
        self.base = base
        
        self.diag_start.requires_grad_(requires_grad=start_grad)
        self.diag_end.requires_grad_(requires_grad=end_grad)
        self.base.requires_grad_(requires_grad=base_grad)
        
    
    def forward(self, x):
        kernel = torch.mul(self.base, self.diag_start - self.diag_end)
        outputs = F.linear(x, kernel.t())
        
        return outputs
    
class NewClassical(nn.Module):
    """
    Classical layer model with weights matrix taken as imput. With this class it is not possible to add a baias 
    during the learning procedure, otherwise the pruning procedure shouldn't work
    
    Parameters
    ----------
    diag_start:
        Imput diagonal, expected torch tensor with shape (input_dimension, 1)
    diag_end:
        Output diagonal, expected torch tensor with shape (1, output_dimension)
    base:
        Eigenvectors base, expected torch tensor with shape (input_dimension, output_dimension)
    start_grad:
        If set to True the starting eigenvalues are trainable
    end_grad:
        If set to True the ending eigenvalues are trainable
    base_grad:
        If set to True the eigenvectors are trainable
    device:
        Device for training
    dtype:
        Type for the training parameters
        
    Example:
    --------
    base = torch.rand(20,20) - 0.5
    model = NewClassical(base)
    
    """
    def __init__(self,
                 weights: torch.double,
                 grad: bool = True,
                 device=None, 
                 dtype=torch.double):
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        super(NewClassical, self).__init__()
        
        self.weights = weights
        self.weights.requires_grad_(requires_grad=grad)
        
        
    
    def forward(self, x):
        outputs = F.linear(x, self.weights.t())
        
        return outputs
        
