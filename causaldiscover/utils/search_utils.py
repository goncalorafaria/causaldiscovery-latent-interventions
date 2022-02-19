import numpy as np
from scipy.linalg import expm
import torch

def generetate_all_bitvector(n=3):
    s = []
    if n > 1 :
        s_p = generetate_all_bitvector(n-1)

        for c in s_p:
            s.append([0]+c)
            s.append([1]+c)
    else:
        s.append([0])
        s.append([1])
    return s

def generetate_all_dags(n=3):
    gs = generetate_all_bitvector(n=n*n)
    gs = torch.tensor(gs,dtype=torch.float).reshape((-1,n,n))
    gsf = gs[(torch.diagonal(gs, dim1=-2, dim2=-1).sum(-1) == 0)]

    constraint = ( torch.diagonal(
        torch.matrix_exp(gsf), dim1=-2, dim2=-1).sum(-1) - n )

    candidates = gsf[constraint== 0]

    return candidates
