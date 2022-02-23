import torch
#from adamp import AdamP
#from sklearn import metrics
from typing import Tuple
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
#i = 4
#hdim = 3

def normalize(u_par):
    return u_par/(u_par**2).sum(-1).sqrt().unsqueeze(-1)

def get_cross_distances(u_par):
        u_par = normalize(u_par)

        ucross = torch.cdist(u_par, u_par, p=2)

        loss = ucross.sum()

        return loss

def init_ro_scheduler(device, minr=0.0, inverse=False, perc=0.6, epochs=2000):

    def step(epoch):
        if inverse:
            pred = 1.0 - ( (epoch+1)/(epochs*perc) )**1.8
            d = max(pred, minr)

        else:
            pred = ( (epoch+1)/(epochs*perc) )**1.8
            d = min(pred, 1.0)

        return torch.tensor( d, device=device)

    return step

def confusion_matrix_2(y_true, y_pred, N):
    y = N * y_true + y_pred
    y = torch.bincount(y)
    if len(y) < N * N:
        y = torch.cat([y, torch.zeros(N * N - len(y), dtype=torch.long, device=y_true.device)])
    y = y.reshape(N, N).float()
    return y

def rand_score(y_true, y_pred, N):

    (tn, fp), (fn, tp) = pair_confusion(y_true, y_pred, N)

    return (tp + tn)/(tp + tn + fp + fn)

def adjusted_rand_score(y_true, y_pred, N):

    (tn, fp), (fn, tp) = pair_confusion(y_true, y_pred, N)

    return 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) +
                                       (tp + fp) * (fp + tn))

def pair_confusion(y_true, y_pred, N):

    n_samples = y_true.shape[0]
    contingency = confusion_matrix_2(y_true, y_pred, N)

    n_c = torch.flatten(contingency.sum(dim=1))
    n_k = torch.flatten(contingency.sum(dim=0))

    sum_squares = (contingency.data ** 2).sum()
    C = torch.empty((2, 2), dtype=torch.float)
    C[1, 1] = sum_squares - n_samples
    C[0, 1] = (contingency@n_k).sum() - sum_squares
    C[1, 0] = (contingency@n_c).sum() - sum_squares
    C[0, 0] = n_samples ** 2 - C[0, 1] - C[1, 0] - sum_squares

    (tn, fp), (fn, tp) = (C[0, 0], C[0, 1]), (C[1, 0], C[1, 1])

    return (tn, fp), (fn, tp)

"""
def test_adjusted_rand_score():
    y_true = torch.tensor([0,1,0,1,2,2])
    y_pred = torch.tensor([1,1,1,1,2,2])

    our = adjusted_rand_score(y_true,y_pred,3)
    theirs = metrics.adjusted_rand_score(y_true,y_pred)

    assert (our-theirs)**2 < 1e-8, "Rand index is badly calculated."

    return True
"""
def onehot(n,i):
    tz = torch.zeros(n)
    if i is not None:
        tz[i]=1
    return tz


#test_adjusted_rand_score()


def graph_precision( pred_G, true_G ):
    relevant = (true_G > 0)

    if relevant.sum() > 0 :
        p = ( (pred_G == true_G) * relevant ).sum() / relevant.sum()
        p = float(p)
    else:
        p = 0

    return p

def graph_recall( pred_G, true_G ):
    relevant = (pred_G > 0)

    if relevant.sum() > 0 :
        r = ( (pred_G == true_G) * relevant ).sum() / relevant.sum()
        r = float(r)
    else:
        r = 0

    return r

def graph_f1( pred_G, true_G ):
    r = graph_recall( pred_G, true_G )
    p = graph_precision( pred_G, true_G )

    if (r + p) > 0 :
        f1 = (2*r*p) / (r+p)
        f1 = float(f1)
    else:
        f1 = 0.0
        
    return f1




def logprobs(probs):
    LOWER_CONST = torch.tensor(1e-7,device=probs.device)
    probs = torch.maximum(probs, LOWER_CONST)
    return torch.log(probs)

def kl_kuma_beta(a,b, alpha,beta):
    """
    (a,b) are the parameters of the kuma posterior.
    (alpha,beta) are the parameters of the beta prior.
    """
    euler_constant = torch.tensor(
        0.57721566490153286060,
        device=a.device
    )  # Euler Mascheroni Constant

    # b, t
    p1 = ((a - alpha)/a)*(-euler_constant-digamma(b)-1/b) +  logprobs(a*b) + lbeta(alpha, beta) - ((b-1)/b )

    moments = [ beta_moments(a, b, i)* 1/float(i) for i in range(1,11)]
    l1_v = torch.stack(moments).sum(0)
    p2 = (beta-1) * b * l1_v ## this should have been an expectation, so this is an approaximation.

    return p1 + p2

def beta_moments(a, b, n):
    """
    Computes nth moment of Kumaraswamy using using torch.lgamma
    """
    arg1 = 1 + n / a
    log_value = torch.lgamma(arg1) + torch.lgamma(b) - torch.lgamma(arg1 + b)
    return b * torch.exp(log_value)

def lbeta(alpha,beta):
    # [\sum_j log(gamma(xj))] - [ log(gamma(\sum xj)) ]
    #  ....., <parameters of beta>
    log_prod_gamma_x = torch.lgamma(alpha) + torch.lgamma(beta)
    sum_x = alpha + beta

    log_gamma_sum_x = torch.lgamma(sum_x)
    result = log_prod_gamma_x - log_gamma_sum_x

    return result

def gamma_prior(x,a,b):
    return (a-1)*torch.log(x) - x/b # - a * torch.log(b) # - torch.lgamma(a)

def clamp_probs(probs):
    eps = 1e-7
    return probs.clamp(min=eps, max=1 - eps)

def gumblesoftmax_rsample(probs, temperature:float, shape:Tuple[int]):
        shape = shape + probs.shape
        uniforms = clamp_probs(torch.rand(shape, dtype=probs.dtype, device=probs.device))
        probs =  clamp_probs(probs)
        logits = (uniforms.log() - (-uniforms).log1p() + probs.log() - (-probs).log1p()) / temperature
        return logits.sigmoid()

def bernoulli_entropy(probs):
    iprobs = (1.0 - probs)
    negentropy = logprobs(iprobs)*iprobs + logprobs(probs)*probs
    return -negentropy

def categorical_entropy(probs):
    entropy= -( probs * logprobs(probs) ).sum(-1)

    return entropy

def categorical_kl_divergence(p,q):
    kl = - (p * logprobs(q)).sum(-1) - categorical_entropy(p)
    return kl

@torch.jit.ignore
def categorical_rsamples(probs, temperature:float=1.0,shape=()):
    return RelaxedOneHotCategorical(
                    temperature,
                    probs=probs).rsample(shape)

@torch.jit.ignore
def beta_rsample(p, w, alpha, batch:int, ones) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    #i_posterior = Beta(p*w,(1.0-p)*w)
    #i_prior = Beta(ones[0], alpha)
    #i_samples = i_posterior.rsample((1,))[:,:-1]

    #ikl = kl_divergence(p=i_posterior,q=i_prior)[:-1].sum()
    posterior = ( p*w, (1.0-p)*w)
    prior = (ones[0],alpha)
    ikl = _kl_beta_beta( posterior, prior )[:,:-1].sum(-1)

    return ikl

def beta_kl(p, w, ones, alpha):

    w = w + 1e-4

    posterior = ( p*w, (1.00001-p)*w)
    prior = (ones[0],alpha)
    ikl = _kl_beta_beta(posterior, prior)

    return ikl

def digamma(x):
    return torch.digamma(x + 1e-6)

def _kl_beta_beta(p, q):
    sum_params_p = p[1] + p[0]
    sum_params_q = q[1] + q[0]
    t1 = q[1].lgamma() + q[0].lgamma() + (sum_params_p).lgamma()
    t2 = p[1].lgamma() + p[0].lgamma() + (sum_params_q).lgamma()
    t3 = (p[1] - q[1]) * digamma(p[1])
    t4 = (p[0] - q[0]) * digamma(p[0])
    t5 = (sum_params_q - sum_params_p) * digamma(sum_params_p)
    return t1 - t2 + t3 + t4 + t5

def kuma_rsample(p, w, alpha, batch, ones):


    kl = kl_kuma_beta(p*w,(1.00000001-p)*w, ones, alpha)


    return None
