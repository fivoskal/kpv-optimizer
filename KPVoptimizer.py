import torch
# from . import _functional as F
from torch.optim import Optimizer
from torch.optim.optimizer import required
import numpy as np

import torch
# from . import _functional as F
# from torch.optim import Optimizer
from torch.optim.optimizer import Optimizer, required
from itertools import tee

class KPV(Optimizer):

    def __init__(self, params, lr=required, p=0.001, k=-1.5, var_bounds=[0.0, 1.0], objective='maximize' ):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if objective not in ['maximize', 'max', 'minimize', 'min']:
            raise ValueError("Agent can be a maximizer or a minimizer.")
            
            
        defaults = dict(lr=lr, k=k, p=p, objective=1.0 if objective=='maximize' else -1.0 )
        params, params_copy = tee(params, 2)
        self.thetas = [ torch.rand_like(param) for param in params_copy ]
        self.p = p
        self.k = k
        self.var_bounds = var_bounds
        self.lr = lr
        
        super(KPV, self).__init__(params, defaults)

        
    def __setstate__(self, state):
        super(KPV, self).__setstate__(state)
        # for group in self.param_groups:
        #     group.setdefault()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            lr = group['lr']
            sign = group['objective']
        
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(sign * p.grad )
                    state = self.state[p]

            for idx, (param, d_p, theta) in enumerate(zip(params_with_grad, d_p_list, self.thetas)):
                if self.k != 0 and self.p != 0:
                    feedback = self.k*( param - theta )
                    theta.add_(param-theta, alpha=lr*self.p)
                    theta.clamp_(self.var_bounds[0], self.var_bounds[1])

                    param.add_(d_p+feedback, alpha=lr)
                    param.clamp_(self.var_bounds[0], self.var_bounds[1])
                else:
                    param.add_(d_p, alpha=lr)
                    param.clamp_(self.var_bounds[0], self.var_bounds[1])
        return loss

class KPVSimplex(Optimizer):

    def __init__(self, params, lr=required, p=0.001, k=-1.5, var_bounds=[0.0, 1.0], objective='maximize' ):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if objective not in ['maximize', 'max', 'minimize', 'min']:
            raise ValueError("Agent can be a maximizer or a minimizer.")
            
            
        defaults = dict(lr=lr, k=k, p=p, objective=1.0 if objective=='maximize' else -1.0 )
        params, params_copy = tee(params, 2)
        self.thetas = [ torch.rand_like(param) for param in params_copy ]
        self.p = p
        self.k = k
        self.var_bounds = var_bounds
        self.lr = lr
        
        super(KPVSimplex, self).__init__(params, defaults)

        
    def __setstate__(self, state):
        super(KPVSimplex, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            lr = group['lr']
            sign = group['objective']
        
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(sign * p.grad )
                    state = self.state[p]

            for idx, (param, d_p, theta) in enumerate(zip(params_with_grad, d_p_list, self.thetas)):
                if self.k != 0 and self.p != 0:
                    feedback = self.k*( p - theta )
                    theta.add_(param-theta, alpha=lr*self.p)
                    theta.clamp_(self.var_bounds[0], self.var_bounds[1])
                    
                    param.add_(d_p+feedback, alpha=lr)
                    param.copy_( projsplx(param.data ))
                else:
                    param.add_(d_p, alpha=lr)
                    param.copy_( projsplx(param.data ))
                    # param.clamp_(self.var_bounds[0], self.var_bounds[1])
        return loss

def projsplx(y):
    """Python implementation of:
    https://arxiv.org/abs/1101.6081"""
    with torch.no_grad():
        s, _ = torch.sort(y)
        print(s)
        n = len(y) ; flag = False
        print(n)
        parsum = 0
        tmax = -np.inf
        for idx in range(n-2, -1, -1):
            print(idx)
            parsum += s[idx+1]
            print('here')
            tmax = (parsum - 1) / (n - (idx + 1) )
            if tmax >= s[idx]:
                flag = True ; break

        if not flag:
            tmax = (torch.sum(s) - 1) / n

        return torch.maximum(y - tmax, torch.Tensor([0]))