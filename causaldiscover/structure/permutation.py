import torch
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau
import torch
import matplotlib.pyplot as plt

is_cuda = False

def to_var(x):
    if is_cuda:
        x = x.cuda()
    return x

def my_sample_gumbel(shape, eps=1e-20):
    """Samples arbitrary-shaped standard gumbel variables.
    Args:
    shape: list of integers
    eps: float, for numerical stability
    Returns:
    A sample of standard Gumbel random variables
    """
    #Sample from Gumbel(0, 1)
    U = torch.rand(shape).float()
    return -torch.log(eps - torch.log(U + eps))

def simple_sinkhorn(MatrixA, n_iter = 5):
    #performing simple Sinkhorn iterations.

    for i in range(n_iter):
        MatrixA = torch.nn.functional.normalize(MatrixA, p=1,dim=1)
        MatrixA = torch.nn.functional.normalize(MatrixA, p=1,dim=2)

    return MatrixA

def matching(log_alpha):
    # applies the 1-sinkhorn operator in log space.
    # computes the regularizer l1-2
    log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
    log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=0, keepdim=True)

    P = torch.exp(log_alpha)

    l1i = torch.norm(P, p=1,dim=0)
    l2i = torch.norm(P, p=2,dim=0)
    l1j = torch.norm(P, p=1,dim=1)
    l2j = torch.norm(P, p=2,dim=1)

    regularizer = (l1i-l2i).sum(-1) + (l1j-l2j).sum(-1)

    return P, regularizer

class NeuralSort(torch.nn.Module):
    def __init__(self, n=3, hard=False):
        super(NeuralSort, self).__init__()
        self.hard = hard

        self.register_buffer(
                'one',torch.ones(( n,1 )) )

        scaling = (n + 1 - 2 * (torch.arange(n) + 1)).float()

        self.register_buffer(
                'scaling',scaling)

    def forward(self, scores: torch.Tensor, temperature=1.0):
        """
        scores: elements to be sorted. Typical shape: batch_size x n x 1
        """
        scores = scores.unsqueeze(-1)
        bsize = scores.size()[0]
        dim = scores.size()[1]
        #one = torch.cuda.FloatTensor(dim, 1).fill_(1)

        A_scores = torch.abs(scores - scores.permute(0, 2, 1))
        B = torch.matmul(A_scores, torch.matmul(
            self.one, torch.transpose(self.one, 0, 1)))

        C = torch.matmul(scores, self.scaling.unsqueeze(0))

        P_max = (C-B).permute(0, 2, 1)
        sm = torch.nn.Softmax(-1)
        P_hat = sm(P_max / temperature)

        if self.hard:
            P = torch.zeros_like(P_hat, device=scores.device)
            b_idx = torch.arange(bsize, device=scores.device).repeat([1, dim]).view(dim, bsize).transpose(
                dim0=1, dim1=0).flatten().long()
            r_idx = torch.arange(dim, device=scores.device).repeat(
                [bsize, 1]).flatten().long()
            c_idx = torch.argmax(P_hat, dim=-1).flatten()  # this is on cuda
            brc_idx = torch.stack((b_idx, r_idx, c_idx))

            P[brc_idx[0], brc_idx[1], brc_idx[2]] = 1
            P_hat = (P-P_hat).detach() + P_hat


        return P_hat


class PLDistribution(torch.nn.Module):
    def __init__(self, n=3, hard=False):

        super(PLDistribution, self).__init__()

        self.relaxedsort = NeuralSort(n=n,hard=hard)
        self.logscores = torch.nn.Parameter(
            torch.randn(n)
            )
        self.n = n

    def rsample(self, batch=1, temperature=1.0, eps=1e-20):

        gshape = [batch, self.logscores.shape[0]]

        U = torch.rand(
            gshape,
            device=self.logscores.device
        ).float()

        gsample = -torch.log(eps - torch.log(U + eps))

        log_s_perturb = self.logscores.unsqueeze(0) + gsample

        Psamples = self.relaxedsort(scores=log_s_perturb,temperature=temperature)

        emp_kp = - self.log_prob(Psamples)

        return Psamples, emp_kp

    def log_prob(self, value):

        permuted_scores = torch.squeeze(torch.matmul(value, self.logscores))
        log_numerator = torch.sum(permuted_scores, dim=-1)
        idx = torch.LongTensor([i for i in range(self.n-1, -1, -1)])
        invert_permuted_scores = permuted_scores.index_select(-1, idx)
        denominators = torch.cumsum(invert_permuted_scores, dim=-1)
        log_denominator = torch.sum(torch.log(denominators), dim=-1)
        return (log_numerator - log_denominator)


def main():

    #torch.set_printoptions(precision=5, sci_mode=False)
    torch.manual_seed(4)

    device = torch.device("cuda:2")

    l = []

    tmp = torch.randn(1, 20, 20, requires_grad=True, device=device)


    match = torch.jit.trace(matching,tmp)

    for i in range(8):
        match(tmp)



    for n in range(100,1000,50):
       # n= 3
        x = torch.randn(n, n, requires_grad=True, device=device)

        start = time.time()
        P = x.matrix_exp().trace()
        P.backward()
        end = time.time()

        e_time = end - start

        torch.manual_seed(4)

        x = torch.randn(1, n, n, requires_grad=True, device=device)

        start = time.time()
        S = simple_sinkhorn((x/0.5).exp(),n_iter=20)
        #print(S)
        #print(S.argmax(-1))
        #print(S.sum(1))
        #print(S.sum(2))
        S[0,0,0].backward()
        end = time.time()

        s10_time = end - start

        ns = NeuralSort(n=n)
        ns.to(device)

        x_c = x[:1,:,0]
        #print(x_c.shape)
        s_time = end - start
        start = time.time()
        S = ns(x_c).sum()
        S.backward()
        end = time.time()

        s5_time = end - start

        l.append(
            [e_time,s10_time, s5_time]
        )

    plt.plot(np.array(l)[:,0],label="exponetial and trace")
    plt.plot(np.array(l)[:,1],label="sinkhorn:20")
    plt.plot(np.array(l)[:,2],label="sort")
    plt.legend()
    print("done")
    plt.savefig("sorta.png")


"""
if __name__ == '__main__':

    device = torch.device("cpu")
    mask = torch.ones(3, 3, device=device)
    mask = torch.tril(mask, diagonal=-1).bool()

    print(mask)
    print("--"*20)

    tmp = torch.randn(3, 3, device=device)
    tmp[~mask] = float('-inf')

    print(tmp)
    print("--"*20)
    probs = tmp.sigmoid()
    print(probs)

    log_perm = torch.randn(3, 3, device=device)/0.0001
    P, r = matching(log_perm)

    print("--"*20)

    print(P.shape)
    print(P.sum(0))
    print(P.sum(1))
    print(P)
    print(r)


    M = torch.tensor([[0,0,0],[1,0,0],[1,1,0]])*2
    P = torch.tensor([[0,1,0],[1,0,0],[0,0,1]])

    print(M)
    print(P)

    S = P.T @ M @ P

    print(S)

"""


#main()

#device = torch.device("cpu")


#ns = NeuralSort(n=4,tau=0.1)

#t = ns(torch.randn((1,4)))
#print(torch.randn((1,2)).shape)

#print(t)
