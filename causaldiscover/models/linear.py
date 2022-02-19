import torch
from torch import nn
from typing import Any, Dict, Iterable, List, Tuple
import math
from causaldiscover.utils.model_utils import logprobs
from causaldiscover.utils.flow_utils import softplus, squareplus,log_normal


from causaldiscover.models.base import HyperModel,approximator, get_meta
from causaldiscover.models.assigments import AssigmentsSSL, OracleAssigments, AssigmentsCategorical, StochasticAssigments, OracleStochasticAssigments, AssigmentsSSLExtended
from causaldiscover.models.interventions import InterventionDirichletProcess, TargetBernoulliExact, TargetBernoulli, TargetSingle, AtomNormal, OracleInterventionDirichletProcess


class PerfectLinearHyperModel(HyperModel):
    def __init__(self,
            n: int,
            hdim: int=10,
            k: int=10,
            target_sparsity: float=1e-3,
            weight_sparsity: float=1e-1,
            alpha=None,
            nintv: int=2,
            use_z_entropy: bool= False,
            train_std: bool=True,
            nonlinear: bool=True,
            mlpdim: int=40,
            mlplayers: int=3,
            hard: bool=False,
            prior: float =0.4,
            atomic:bool=False
            ):

        super(PerfectLinearHyperModel,self).__init__(
            n=n,k=k,target_sparsity=target_sparsity,
            weight_sparsity=weight_sparsity)

        self.atomic = atomic
        self.interventions = InterventionDirichletProcess(
            targets= TargetBernoulli(
                n=n,
                nintv=nintv,
                nonlinear=nonlinear,
                hard=hard,
                prior=prior,
                mlplayers=mlplayers
            ),
            atoms=AtomNormal(
                hdim=hdim,
                k=k,
                mlpdim=mlpdim,
                nonlinear=nonlinear,
                mlplayers=mlplayers,
                nintv=nintv,
                kl=True
            ),
            hdim=hdim,
            k=k,
            alpha=alpha,
            nintv=nintv,
            nonlinear=nonlinear,
            mlplayers=mlplayers,
            mlpdim=mlpdim
        )

        self.assigments = StochasticAssigments(
            n=n, k=k,
            use_z_entropy=use_z_entropy,
            nonlinear=nonlinear,
            nintv=nintv,
            hard=hard,
            mix=False,
            hdim=hdim
        )

        self.n = n
        self.scm = nn.Linear(n,n)

        self.logstd = torch.nn.Parameter( torch.randn(n) )

        fb, applier = approximator(
            k, 2*n,
            hdim=mlpdim, nonlinear=nonlinear,
            mlplayers=mlplayers,
            activate_mollify=False
        )
        
        #self.bp = fb
        self.b_proj= fb #applier(self.bp)

        self.register_buffer('eps', torch.ones(n)*1)
        self.register_buffer('leps', torch.ones(n)*(-1.0))

        self.train_std = train_std

        self.gconstant = math.log(2.0*math.pi)/2.0


    def get_params(self, z, h, target_samples, assigment_samples, mollify:float=0.0):

        W = self.scm.weight
        #B = self.scm.bias
        dout = self.b_proj(h)

        dout = torch.einsum("bt,btj->bj",  assigment_samples, dout)

        mean_k = dout[:,:self.n]
        unstd_k = dout[:,self.n:2*self.n]

        zexp_r = torch.einsum("bt,btj->bj",assigment_samples,target_samples) # r^

        if len(z.shape) == 3:
            z_reduced = z * ( 1 - zexp_r.unsqueeze(-1) )

            w = torch.einsum("bij,ij->bij",z_reduced,W)
        else:
            z_reduced = z * ( 1 - zexp_r.unsqueeze(-1) )
            
            w = torch.einsum("ij,ij->ij",z_reduced,W) # here

        #import pdb; pdb.set_trace()

        std = squareplus(
            self.logstd
        )

        std_k = squareplus(
            unstd_k 
        )

        return w, self.scm.bias, zexp_r, mean_k, 0.0, std, std_k

    def scm_forward(self, x, model_params:Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor],mollify:float=0.0):
        w, bias, zexp_r, mean_k, zvar_r, std, std_k = model_params

        if len(w.shape)==3:
            mean = torch.einsum("bi,bji->bj",x,w) + bias
        else:
            mean = torch.einsum("bi,ji->bj",x,w) + bias

        #y = (1-zexp_r)*d + zexp_r*b_i

        if self.atomic:
            std_k = 1e-2 * torch.ones_like(std_k)

        #if len(beta.shape)==3:
        #    zexp_r = zexp_r.unsqueeze(1)
        #    logstd = logstd.unsqueeze(0).unsqueeze(0)
        #    logstd = ((1-zexp_r)*logstd + zexp_r*beta).sum(1)
        #else:
        #    logstd = (1-zexp_r)*logstd + zexp_r*beta

        #  torch.zeros_like(logstd)
        return mean, std, std_k, mean_k, zexp_r

    def density(self, x, dist_params:Tuple[torch.Tensor,torch.Tensor], mollify:float=0.0):
        mean, std, std_k, mean_k, zexp_r = dist_params

        """
        def log_normal(x, mean, log_var):#TODO: this will create issues put the correct version.
            return - 1/(log_var.exp()*2.+delta) * (x - mean) ** 2 - log_var/2. + c
        """

        if self.train_std:
            true_mse_0 = -log_normal(x, mean, std)
            true_mse_k = -log_normal(x, mean_k, std_k)
        else:
            true_mse_0 = (x-mean)**2 + self.gconstant
            true_mse_k = (x-mean_k)**2 + self.gconstant

        true_mse = (1- zexp_r)*true_mse_0 + zexp_r*true_mse_k

        Ldata = true_mse.sum(1)

        return Ldata

    def penalize_model_complexity(self, model_params:Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]):
        w, _, _ , _, _, _, beta = model_params
        return w.abs().mean()

class PerfectLinearOracleHyperModel(PerfectLinearHyperModel):
    def __init__(self,
            n: int,
            hdim: int=10,
            k: int=10,
            target_sparsity: float=1e-3,
            weight_sparsity: float=1e-1,
            alpha=None,
            nintv: int=2,
            use_z_entropy: bool= False,
            train_std: bool=True,
            nonlinear: bool=True,
            mlpdim: int=40,
            mlplayers: int=3,
            hard: bool=False,
            prior: float =0.4,
            atomic:bool=False,
            ):

        super(PerfectLinearOracleHyperModel,self).__init__(
            n=n,
            hdim=hdim,
            k=k,
            target_sparsity=target_sparsity,
            weight_sparsity=weight_sparsity,
            alpha=alpha,
            nintv=nintv,
            use_z_entropy=use_z_entropy,
            train_std=train_std,
            nonlinear=nonlinear,
            mlpdim=mlpdim,
            mlplayers=mlplayers,
            hard=hard,
            prior=prior,
            atomic=atomic
        )

        self.assigments = OracleStochasticAssigments()

class PerfectLinearSSLHyperModel(PerfectLinearHyperModel):
    def __init__(self,
            n: int,
            hdim: int=10,
            k: int=10,
            target_sparsity: float=1e-3,
            weight_sparsity: float=1e-1,
            alpha=None,
            nintv: int=2,
            use_z_entropy: bool= False,
            train_std: bool=True,
            nonlinear: bool=True,
            mlpdim: int=40,
            mlplayers: int=3,
            hard: bool=False,
            prior: float =0.4,
            atomic:bool=False,
            ):

        super(PerfectLinearSSLHyperModel,self).__init__(
            n=n,
            hdim=hdim,
            k=k,
            target_sparsity=target_sparsity,
            weight_sparsity=weight_sparsity,
            alpha=alpha,
            nintv=nintv,
            use_z_entropy=use_z_entropy,
            train_std=train_std,
            nonlinear=nonlinear,
            mlpdim=mlpdim,
            mlplayers=mlplayers,
            hard=hard,
            prior=prior,
            atomic=atomic
        )

        self.assigments = AssigmentsSSL(
            n=n, k=k,
            use_z_entropy=use_z_entropy,
            nonlinear=nonlinear,
            nintv=nintv,
            hard=hard,
            mix=False,
            hdim=hdim
        )

class PerfectLinearFullOracleHyperModel(PerfectLinearOracleHyperModel):
    def __init__(self,
            ground_truth_targets,
            n: int,
            hdim: int=10,
            k: int=10,
            target_sparsity: float=1e-3,
            weight_sparsity: float=1e-1,
            alpha=None,
            nintv: int=2,
            use_z_entropy: bool= False,
            train_std: bool=True,
            nonlinear: bool=True,
            mlpdim: int=40,
            mlplayers: int=3,
            hard: bool=False,
            prior: float =0.4,
            atomic:bool=False,
            ):

        super(PerfectLinearFullOracleHyperModel,self).__init__(
            n=n,
            hdim=hdim,
            k=k,
            target_sparsity=target_sparsity,
            weight_sparsity=weight_sparsity,
            alpha=alpha,
            nintv=nintv,
            use_z_entropy=use_z_entropy,
            train_std=train_std,
            nonlinear=nonlinear,
            mlpdim=mlpdim,
            mlplayers=mlplayers,
            hard=hard,
            prior=prior,
            atomic=atomic
        )

        self.interventions = OracleInterventionDirichletProcess(
            ground_truth_targets=ground_truth_targets,
            atoms=AtomNormal(
                hdim=hdim,
                k=k,
                mlpdim=mlpdim,
                nonlinear=nonlinear,
                mlplayers=mlplayers,
                nintv=nintv,
                kl=True
            ),
            hdim=hdim,
            k=k,
            alpha=alpha,
            nintv=nintv,
            nonlinear=nonlinear,
            mlplayers=mlplayers,
            mlpdim=mlpdim
        )

class LinearHyperModel(HyperModel):
    def __init__(self,
            n: int,
            hdim: int=10,
            k: int=10,
            target_sparsity: float=1e-3,
            weight_sparsity: float=1e-1,
            alpha=None,
            nintv: int=2,
            use_z_entropy: bool= False,
            train_std: bool=True,
            nonlinear: bool=True,
            mlpdim: int=10,
            mlplayers: int=1,
            hard: bool=True,
            prior:float =0.5
            ):
        super(LinearHyperModel,self).__init__(
            n=n,k=k,target_sparsity=target_sparsity,
            weight_sparsity=weight_sparsity)
        
        self.interventions = InterventionDirichletProcess(
            atoms=AtomNormal(
                hdim=hdim,
                k=k,
                mlpdim=mlpdim,
                nonlinear=nonlinear,
                mlplayers=mlplayers,
                nintv=nintv,
                kl=True
            ),
            targets= TargetBernoulli(
                n=n,
                nintv=nintv,
                nonlinear=nonlinear,
                hard=hard,
                prior=prior,
                mlplayers=mlplayers
            ),
            hdim=hdim,
            k=k,
            alpha=alpha,
            nintv=nintv,
            nonlinear=nonlinear
        )

        self.assigments = AssigmentsCategorical(
            n=n,k=k,use_z_entropy=use_z_entropy,
            nonlinear=nonlinear,hard=hard, 
            mlplayers=mlplayers, mix=False,nintv=nintv,
            hdim=hdim
        )

        fw, applier = approximator(
            k, n*n + 2*n,
            hdim=mlpdim, nonlinear=nonlinear,
            mlplayers=mlplayers
        )


        self.wp = fw
        self.w_proj= fw #applier(self.wp)

        self.train_std = train_std

        self.gconstant = math.log(2.0*math.pi)/2.0

        self.register_buffer('eps', torch.ones(n)*1)
        self.register_buffer('leps', torch.ones(n)*(-2))

    def get_params(self, z, h, target_samples, assigment_samples, mollify:float=0.0):

        dout = self.w_proj(h)

        if len(h.shape) == 2 :

            dw = dout[:,:self.n*self.n]
            b = dout[:,self.n*self.n:self.n*(self.n+1)]
            beta_logit = dout[:,self.n*(self.n+1) : self.n*(self.n+2)]

            dw = dw.view(h.shape[0],self.n,self.n) # "tl,lij->tij"
        else:
            dw = dout[:,:, :self.n*self.n]
            b = dout[:,:, self.n*self.n:self.n*(self.n+1)]
            beta_logit = dout[:, :, self.n*(self.n+1) : self.n*(self.n+2)]

            dw = dw.view(h.shape[0],h.shape[1],self.n,self.n)

        #b = self.b_proj(h, mollify) #torch.einsum("tl,lj->tj", h_t, self.b_proj)

        #beta_logit = self.beta_proj(h, mollify)
        beta = squareplus(beta_logit)

        if len(z.shape) > 2:
            w = torch.einsum("bij,btij->btij",z,dw)
            wobs = w[:,:1]
            wk = w[:,1:,:,:]
            deltaw = (w-wobs)
            w = torch.einsum("bti,btij->btij", target_samples, deltaw) + wobs

        else:
            w = torch.einsum("ij,btij->btij",z,dw)
            wobs = w[:,:1]
            wk = w[:,1:,:,:]
            deltaw = (w-wobs)
            w = torch.einsum("bti,btij->btij", target_samples, deltaw) + wobs


        bobs = b[:1,:]
        betaobs = beta[:1,:]

        b = torch.einsum("bti,bti->bti", target_samples, (b-bobs)) + bobs
        beta = torch.einsum("bti,bti->bti", target_samples, (beta-betaobs)) + betaobs

        #beta = torch.ones(target_samples.shape, device=target_samples.device)*2
        return w, b, wk, beta

    def scm_forward(self, x, model_params:Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor],mollify:float=0.0):
        w, b, _, beta = model_params
        return torch.einsum("bi,btji->btj",x,w) + b, beta

    def density(self, x, dist_params:Tuple[torch.Tensor,torch.Tensor], mollify:float=0.0):
        y_mean, beta = dist_params

        true_mse = (x.unsqueeze(1) - y_mean)**2

        if self.train_std:
            mse = torch.einsum("btj,btj->bt", true_mse, beta/2.0)
            Ldata = mse - (logprobs(beta) + self.gconstant).sum(-1)
        else:
            Ldata = (true_mse + self.gconstant).sum(-1)

        return Ldata

    def penalize_model_complexity(self, model_params:Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]):
        (_, _, wk,_) = model_params
        return wk.abs().mean()

class LinearOracleHyperModel(LinearHyperModel):
    def __init__(self,
            n: int,
            hdim: int=10,
            k: int=10,
            target_sparsity: float=1e-3,
            weight_sparsity: float=1e-1,
            alpha=None,
            nintv: int=2,
            use_z_entropy: bool= False,
            train_std: bool=False,
            nonlinear: bool=True,
            mlpdim: int=10,
            mlplayers: int=1,
            hard: bool=False,
            prior:float =0.5
            ):

        super(LinearOracleHyperModel,self).__init__(
            n=n,
            hdim=hdim,
            k=k,
            target_sparsity=target_sparsity,
            weight_sparsity=weight_sparsity,
            alpha=alpha,
            nintv=nintv,
            use_z_entropy=use_z_entropy,
            train_std=train_std,
            nonlinear=nonlinear,
            mlpdim=mlpdim,
            mlplayers=mlplayers,
            hard=hard,
            prior=prior
        )

        #del self.assigments
        #del self.interventions

        self.assigments = OracleAssigments()

class LinearFullOracleHyperModel(LinearOracleHyperModel):
    def __init__(self,
            ground_truth_targets: torch.Tensor,
            n: int,
            hdim: int=10,
            k: int=10,
            target_sparsity: float=1e-3,
            weight_sparsity: float=1e-1,
            alpha=None,
            nintv: int=2,
            use_z_entropy: bool= False,
            train_std: bool=False,
            nonlinear: bool=True,
            mlpdim: int=10,
            mlplayers: int=1,
            hard: bool=False,
            prior:float =0.5
            ):

        super(LinearFullOracleHyperModel, self).__init__(
            n=n,
            hdim=hdim,
            k=k,
            target_sparsity=target_sparsity,
            weight_sparsity=weight_sparsity,
            alpha=alpha,
            nintv=nintv,
            use_z_entropy=use_z_entropy,
            train_std=train_std,
            nonlinear=nonlinear,
            mlpdim=mlpdim,
            mlplayers=mlplayers,
            hard=hard,
            prior=prior
        )

        self.interventions = OracleInterventionDirichletProcess(
            ground_truth_targets=ground_truth_targets,
            atoms=AtomNormal(
                hdim=hdim,
                k=k,
                mlpdim=mlpdim,
                nonlinear=nonlinear,
                mlplayers=mlplayers,
                nintv=nintv,
                kl=True
            ),
            hdim=hdim,
            k=k,
            alpha=alpha,
            nintv=nintv,
            nonlinear=nonlinear,
            mlplayers=mlplayers,
            mlpdim=mlpdim
        )

class LinearSSLHyperModel(LinearHyperModel):
    def __init__(self,
            n: int,
            hdim: int=10,
            k: int=10,
            target_sparsity: float=1e-3,
            weight_sparsity: float=1e-1,
            alpha=None,
            nintv: int=2,
            use_z_entropy: bool= False,
            train_std: bool=False,
            nonlinear: bool=True,
            mlpdim: int=10,
            mlplayers: int=1,
            hard: bool=False,
            prior:float =0.5
            ):

        super(LinearSSLHyperModel,self).__init__(
            n=n,
            hdim=hdim,
            k=k,
            target_sparsity=target_sparsity,
            weight_sparsity=weight_sparsity,
            alpha=alpha,
            nintv=nintv,
            use_z_entropy=use_z_entropy,
            train_std=train_std,
            nonlinear=nonlinear,
            mlpdim=mlpdim,
            mlplayers=mlplayers,
            hard=hard,
            prior=prior
        )

        self.assigments = AssigmentsSSLExtended(
            n=n, k=k,
            use_z_entropy=use_z_entropy,
            nonlinear=nonlinear,
            nintv=nintv,
            hard=hard,
            mix=False,
            hdim=hdim
        )


#"perfect,oracle,known"

meta_ext = {
    "1110":PerfectLinearFullOracleHyperModel,
    "1100":PerfectLinearOracleHyperModel,
    "1000":PerfectLinearHyperModel,
    "1001":PerfectLinearSSLHyperModel,
    "0110":LinearFullOracleHyperModel,
    "0100":LinearOracleHyperModel,
    "0000":LinearHyperModel,
    "0001":LinearSSLHyperModel
}

meta = get_meta(meta_ext,n=4)

"""
meta = {True:
            { 
            True:{
                True:PerfectLinearFullOracleHyperModel ,
                False:PerfectLinearOracleHyperModel},
            False:{
                True:None,
                False:PerfectLinearHyperModel}
            },
        False:
            { 
            True:{
                True:LinearFullOracleHyperModel,
                False:LinearOracleHyperModel},
            False:{
                True:None,
                False:LinearHyperModel}
            }
        }
"""