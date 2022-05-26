import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


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
        
        
        # Build the model 
        
        # Eigenvectors
        self.base = nn.Parameter(torch.empty(self.in_dim, self.out_dim, **factory_kwargs), requires_grad=self.base_grad)
        # Eigenvalues start 
        self.diag_start = nn.Parameter(torch.empty(self.in_dim, 1, **factory_kwargs),  requires_grad=self.start_grad)
        # Eigenvalues end
        self.diag_end = nn.Parameter(torch.empty(1, self.out_dim, **factory_kwargs), requires_grad=self.end_grad)
        # bias
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_dim, **factory_kwargs), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        
        # Inizialize the layer 
        self.reset_parameters()
            
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        # Same of torch.nn.modules.linear
        nn.init.kaiming_uniform_(self.base, a=math.sqrt(5))
        
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.base)
        bound_in = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        bound_out = 1 / math.sqrt(fan_out) if fan_out > 0 else 0
        nn.init.uniform_(self.diag_start, -bound_in, bound_in)
        nn.init.uniform_(self.diag_end, -bound_out, bound_out)
        
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound_in, bound_in)
    
        
    def forward(self, x):
        kernel = torch.mul(self.base, self.diag_start - self.diag_end)
        if self.bias:
            outputs = F.linear(x, kernel.t(), self.bias)
        else:
            outputs = F.linear(x, kernel.t())
        
        return outputs
    
    def extra_repr(self) -> str:
        return 'in_dim={}, out_dim={}, base_grad={}, start_grad={}, end_grad={}, bias={}'.format(
            self.in_dim, self.out_dim, self.base_grad, self.start_grad, self.end_grad, self.bias is not None
        )
    
    def assign(self, 
               in_dim: int,
               out_dim: int,
               base: torch.Tensor = None, 
               diag_start: torch.Tensor = None, 
               diag_end: torch.Tensor = None,
               bias: torch.Tensor = None):
        """
        This method assigns the from inputs eigenvectors, eigenvalues and bias, and change the dimension of the layer.
        
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        if base is not None:
            self.base = base
        if diag_start is not None:
            self.diag_start = diag_start
        if diag_end is not None:
            self.diag_end = diag_end
        if bias is not None:
            self.base = bias
            
        
    
    
    
    def direct_space(self):
        return torch.mul(self.base, self.diag_start - self.diag_end).detach().numpy().T

    def return_base(self):
        c = self.base.shape[0]
        N = reduce_sum(self.base.shape).detach().numpy()
        phi = np.eye(N)
        phi[c:, :c] = self.base.detach().numpy().T
        return phi
    
    def return_diag(self):
        """
        Returns the eigenvalues as [start, end]. Start are in relation with the first neurons and end with the last
        of the linear transfer between layer k and k+1
        """
        if self.start_grad and self.end_grad:
            return np.concatenate([self.diag_start.numpy().reshape([-1]), self.diag_end.numpy().reshape([-1])], axis=0)
        elif self.start_grad and not self.end_grad:
            return self.diag_start.numpy().reshape([-1])
        elif not self.start_grad and self.end_grad:
            return self.diag_end.numpy().reshape([-1]) 
    
    
    def pruning_diag(self, 
                     perc: float,
                     in_dim: int,
                     out_dim: int,
                     start_grad: bool = False, 
                     end_grad: bool = False, 
                     base_grad: bool = True):
        """
        This method apply a pruning procedure, acting directly on the elements in the diagonal of this class. 
        If it is necessary to mantain the original model and the pruned model it is possible to realize a copy of the model
        using new_model = copy.deepcopy(old_model)
        
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
        
        diag_start = diag_start[diag_start]
        
        b = a[a < 0.5]
        diag_start = np.reshape(diag_start, (self.in_dim, 1))
        self.diag_start = nn.Parameter(torch.from_numpy(diag_start), requires_grad=start_grad)
        # New diag_end
        diag_end = d[self.in_dim:]
        diag_end = np.reshape(diag_end, (1, self.out_dim))
        self.diag_end = nn.Parameter(torch.from_numpy(diag_end), requires_grad=end_grad)
        # Eventually change base requires_grad
        self.base.requires_grad_(requires_grad=base_grad)
        
    
    
    
    
        

        