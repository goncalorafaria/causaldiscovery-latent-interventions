import numpy as np
import torch

from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

delta = 1e-6
c = - 0.5 * np.log(2 * np.pi)



def log(x):
    return torch.log(x * 1e2) - np.log(1e2)


def log_normal(x, mean, std):#TODO: this will create issues put the correct version.

    sqstd = std**2.
    sqstd = torch.maximum(torch.ones_like(std), sqstd)

    return - (1/(sqstd+delta)) *(x-mean)**2 - log(sqstd)/2. + c


def logsigmoid(x):
    return -softplus(-x)


def log_sum_exp(A, axis=-1, sum_op=torch.sum):
    def maximum(x):
        return x.max(axis)[0]

    A_max = oper(A, maximum, axis, True)

    def summation(x):
        return sum_op(torch.exp(x - A_max), axis)

    B = torch.log(oper(A, summation, axis, True)) + A_max
    return B


def oper(array, oper, axis=-1, keepdims=False):
    a_oper = oper(array)
    if keepdims:
        shape = []
        for s in array.size():
            shape.append(s)
        shape[axis] = -1
        a_oper = a_oper.view(*shape)
    return a_oper


def softplus(x):
    return F.softplus(x) + 0.001

def squareplus(x):
    return (x + (x**2+4).sqrt())/2


def softmax(x, dim=-1):
    e_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
    out = e_x / e_x.sum(dim=dim, keepdim=True)
    return out


class SigmoidFlow(torch.nn.Module):
    """
    Layer used to build Deep sigmoidal flows

    Parameters:
    -----------
    num_ds_dim: uint
        The number of hidden units

    """

    def __init__(self,
            n=3,
            sharedpar=2):

        super(SigmoidFlow, self).__init__()

        self.shared_density_params = torch.nn.Parameter(
            torch.zeros( (1, n, 3, sharedpar) )
        )

    def act_a(self, x):
        return softplus(x) # ensure a is positive.

    def act_b(self, x):
        return x

    def act_w(self, x):
        return softmax(x, dim=2) # ensure w lie in the simplex.

    def forward(self, x, logdet, dsparams, mollify:float=0.0, delta:float=delta):

        # Apply activation functions to the parameters produced by the hypernetwork

        shared_density_params = self.shared_density_params.repeat([x.shape[0],1,1,1])
        dsparams = torch.cat([dsparams,shared_density_params],axis=-1)

        a_ = self.act_a(dsparams[:, :, 0])
        b_ = self.act_b(dsparams[:, :, 1])
        w = self.act_w(dsparams[:, :, 2])

        if mollify :
            a = a_ * (1 - mollify) + 1.0 * mollify
            b = b_ * (1 - mollify) + 0.0 * mollify
        else:
            a = a_
            b = b_

        pre_sigm = a * x.unsqueeze(-1) + b  # C
        sigm = torch.sigmoid(pre_sigm)
        x_pre = torch.sum(w * sigm, dim=2)  # D
        x_pre_clipped = x_pre * (1 - delta) + delta * 0.5
        x_ = log(x_pre_clipped) - log(1 - x_pre_clipped)  # Logit function (so H)
        xnew = x_

        logj = F.log_softmax(dsparams[:, :, 2], dim=2) + \
               logsigmoid(pre_sigm) + \
               logsigmoid(-pre_sigm) + log(a)

        logj = log_sum_exp(logj, 2).sum(2)

        logdet_ = logj + np.log(1 - delta) - (log(x_pre_clipped) + log(-x_pre_clipped + 1))
        logdet += logdet_

        return xnew, logdet
