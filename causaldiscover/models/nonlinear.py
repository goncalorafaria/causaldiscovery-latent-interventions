import torch
from torch import nn
from typing import Any, Dict, Iterable, List, Tuple
from torch.autograd import Variable

from causaldiscover.models.base import HyperModel,approximator, get_meta
from causaldiscover.models.assigments import OracleAssigments, AssigmentsCategorical, StochasticAssigments, OracleStochasticAssigments
from causaldiscover.models.interventions import InterventionDirichletProcess, TargetBernoulliExact, TargetBernoulli, AtomNormal, OracleInterventionDirichletProcess
from causaldiscover.utils.flow_utils import log_normal

class NNHyperModel(HyperModel):
    def __init__(self,
            n: int,
            hdim: int=10,
            k: int=10,
            target_sparsity: float=1e-3,
            weight_sparsity: float=1e-1,
            alpha = None,
            nintv: int=2,
            use_z_entropy: bool= False,
            mlpdim: int =40,
            mlplayers: int=2,
            nonlinear: bool=True,
            hard: bool=False,
            prior: float=0.4):

        super(NNHyperModel,self).__init__(
            n=n,k=k,target_sparsity=target_sparsity,
            weight_sparsity=weight_sparsity)

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
            hdim=hdim,k=k,
            alpha=alpha,
            nintv=nintv,
            nonlinear=nonlinear
        )

        self.assigments = StochasticAssigments(
            n=n, k=k, use_z_entropy=use_z_entropy,
            nonlinear=nonlinear, hard=hard, mix=False,nintv=nintv
        )

        fmodule, applier = approximator(
            self.n+self.k,
            1,
            hdim=mlpdim,
            nonlinear=nonlinear,
            mlplayers=mlplayers,
            n=n,
            hyper=True)

        #self.fmodule = fmodule

        self.predictor = fmodule # applier(self.fmodule)

        self.logstd = torch.nn.Parameter( torch.randn(n) )

        fstd, applier = approximator(
            self.k,
            n,
            hdim=mlpdim,
            nonlinear=nonlinear,
            mlplayers=mlplayers,
            n=n,
            hyper=False)

        self.fstd = fstd


    def get_params(self, z, h, target_samples, assigment_samples, mollify:float=0.0):


        single_h = torch.einsum("bt,btk->bk", assigment_samples, h)
        single_r = torch.einsum("bt,btj->bj", assigment_samples, target_samples).unsqueeze(-1)

        logstd = self.fstd(single_h)

        single_h = single_h.unsqueeze(1)

        single_tile_h = single_h.repeat([1,self.n,1])
        #single_tile_r = single_r.repeat([1,1,single_r.shape[1]])

        relaxed_h_sample = single_r*single_tile_h + (1.0-single_r)*h[:,:1,:]

        i = torch.argmax(assigment_samples,dim=-1)

        return relaxed_h_sample, z, logstd

    def scm_forward(self, x, model_params:Tuple[torch.Tensor,torch.Tensor],mollify:float=0.0):
        relaxed_h_sample, z, logstd = model_params

        if len(z.shape) == 2 :
            z = z.unsqueeze(0)

        x_expand = x.unsqueeze(1).repeat([1,self.n,1]) * z
        
        x_input = torch.cat([x_expand,relaxed_h_sample],axis=-1)
        # x_input : [ b x n x (n + k) ]
        # torch.Size([248, 3, 5])
        a = self.predictor(x_input)

        return a, logstd 

    def density(self, x, dist_params:torch.Tensor, mollify:float=0.0):

        dist_params, logstd = dist_params
        dist_params = dist_params.squeeze(-1)

        pseudo_joint_nll = log_normal(
            x, 
            dist_params, 
            torch.zeros_like(logstd)
        )

        return -pseudo_joint_nll.sum(-1)


    def penalize_model_complexity(self, model_params:Tuple[torch.Tensor,torch.Tensor]):
        return torch.zeros((),device=model_params[0].device)

class NNOracleHyperModel(NNHyperModel):
    def __init__(self,
            n: int,
            hdim: int=10,
            k: int=10,
            target_sparsity: float=1e-3,
            weight_sparsity: float=1e-1,
            alpha = None,
            nintv: int=2,
            use_z_entropy: bool= False,
            mlpdim=40,
            mlplayers=3,
            nonlinear=True,
            hard=True,
            prior: float=0.4):


        super(NNOracleHyperModel,self).__init__(
            n=n,
            hdim=hdim,
            k=k,
            target_sparsity=target_sparsity,
            weight_sparsity=weight_sparsity,
            alpha=alpha,
            nintv=nintv,
            use_z_entropy=use_z_entropy,
            mlpdim=mlpdim,
            mlplayers=mlplayers,
            nonlinear=nonlinear,
            hard=hard,
            prior=prior
        )

        self.assigments = OracleStochasticAssigments()

class NNFullOracleHyperModel(NNOracleHyperModel):

    def __init__(self,
            ground_truth_targets,
            n: int,
            hdim: int=10,
            k: int=10,
            target_sparsity: float=1e-3,
            weight_sparsity: float=1e-1,
            alpha = None,
            nintv: int=2,
            use_z_entropy: bool= False,
            mlpdim=40,
            mlplayers=3,
            nonlinear=True,
            hard=True,
            prior: float=0.4):

        super(NNFullOracleHyperModel,self).__init__(
            n=n,
            hdim=hdim,
            k=k,
            target_sparsity=target_sparsity,
            weight_sparsity=weight_sparsity,
            alpha=alpha,
            nintv=nintv,
            use_z_entropy=use_z_entropy,
            mlpdim=mlpdim,
            mlplayers=mlplayers,
            nonlinear=nonlinear,
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

class NNPerfectHyperModel(NNHyperModel):
    def __init__(self,
        n: int,
        hdim: int=10,
        k: int=10,
        target_sparsity: float=1e-3,
        weight_sparsity: float=1e-1,
        alpha = None,
        nintv: int=2,
        use_z_entropy: bool= True,
        mlpdim: int =40,
        mlplayers: int=2,
        nonlinear: bool=True,
        hard: bool=False,
        prior: float=0.4,
        atomic: bool=False):

        super(NNPerfectHyperModel,self).__init__(
            n=n,
            hdim=hdim,
            k=k,
            target_sparsity=target_sparsity,
            weight_sparsity=weight_sparsity,
            alpha=alpha,
            nintv=nintv,
            use_z_entropy=use_z_entropy,
            mlpdim=mlpdim,
            mlplayers=mlplayers,
            nonlinear=nonlinear,
            hard=hard,
            prior=prior
        )

        self.atomic = atomic
        fmodule, applier = approximator(
            self.n,
            1,
            hdim=mlpdim,
            nonlinear=nonlinear,
            mlplayers=mlplayers,
            hyper=True,
            n=n)

        self.predictor = fmodule

        #self.predictor = applier(self.fmodule)

        fb, applier = approximator(
            k, n*2,
            hdim=mlpdim, nonlinear=nonlinear,
            mlplayers=mlplayers,
            activate_mollify=False,
            n=n,
        )

        #self.bp = fb
        self.b_proj= fb


    def get_params(self, z, h, target_samples, assigment_samples, mollify:float=0.0):

        out = self.b_proj(h) # [B, t, d]
        out = torch.einsum("bt,btj->bj",  assigment_samples, out) # [B, t]

        b_i = out[:,:self.n]
        beta_i = out[:,self.n:self.n*2]

        single_h = torch.einsum("bt,btk->bk", assigment_samples, h).unsqueeze(1) # [B, 1, k]
        single_r = torch.einsum("bt,btj->bj", assigment_samples, target_samples).unsqueeze(2) # [B, d, 1]

        single_tile_h = single_h.repeat([1,self.n,1]) # [B, d, k]

        relaxed_h_sample = single_r*single_tile_h + (1.0-single_r)*h[:,:1,:]

        zexp_r = torch.einsum("bt,btj->bj",assigment_samples, target_samples)

        z_reduced = z * ( 1 - zexp_r.unsqueeze(-1) )
        
        return relaxed_h_sample, z_reduced, zexp_r, b_i, beta_i

    def scm_forward(self, x, model_params:Tuple[torch.Tensor,torch.Tensor],mollify:float=0.0):
        relaxed_h_sample, z, zexp_r,b_i, beta_i = model_params

        if len(z.shape) == 2 :
            z = z.unsqueeze(0)

        x_expand = x.unsqueeze(1).repeat([1,self.n,1]) * z

        # x_input : [ b x n x (n + k) ]
        a = self.predictor(x_expand).squeeze(-1)

        return a, zexp_r, b_i, beta_i

    def density(self, x, dist_params:torch.Tensor, mollify:float=0.0):
        dist_params, zexp_r, b_i, beta_i = dist_params
        
        if self.atomic:
            beta_i = torch.ones_like(beta_i)*0.01

        dist_params = dist_params
        pseudo_joint_nll = log_normal(
            x, 
            dist_params, 
            self.logstd
        )

        joint_nll = pseudo_joint_nll

        return -joint_nll.sum(-1)

class NNPerfectOracleHyperModel(NNPerfectHyperModel):
    def __init__(self,
            n: int,
            hdim: int=10,
            k: int=10,
            target_sparsity: float=1e-3,
            weight_sparsity: float=1e-1,
            alpha = None,
            nintv: int=2,
            use_z_entropy: bool= False,
            mlpdim=40,
            mlplayers=3,
            nonlinear=True,
            hard=True,
            prior: float=0.4,
            atomic: bool = False):


        super(NNPerfectOracleHyperModel,self).__init__(
            n=n,
            hdim=hdim,
            k=k,
            target_sparsity=target_sparsity,
            weight_sparsity=weight_sparsity,
            alpha=alpha,
            nintv=nintv,
            use_z_entropy=use_z_entropy,
            mlpdim=mlpdim,
            mlplayers=mlplayers,
            nonlinear=nonlinear,
            hard=hard,
            prior=prior,
            atomic=atomic
        )

        self.assigments = OracleStochasticAssigments()

class NNFullPerfectOracleHyperModel(NNPerfectOracleHyperModel):
    def __init__(self,
            ground_truth_targets,
            n: int,
            hdim: int=10,
            k: int=10,
            target_sparsity: float=1e-3,
            weight_sparsity: float=1e-1,
            alpha = None,
            nintv: int=2,
            use_z_entropy: bool= False,
            mlpdim=40,
            mlplayers=3,
            nonlinear=True,
            hard=True,
            prior: float=0.4,
            atomic:bool=False):

        super(NNFullPerfectOracleHyperModel,self).__init__(
            n=n,
            hdim=hdim,
            k=k,
            target_sparsity=target_sparsity,
            weight_sparsity=weight_sparsity,
            alpha=alpha,
            nintv=nintv,
            use_z_entropy=use_z_entropy,
            mlpdim=mlpdim,
            mlplayers=mlplayers,
            nonlinear=nonlinear,
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

# ["perfect", "oracle", "known"]

meta_ext = {
    "1110":NNFullPerfectOracleHyperModel ,
    "1100":NNPerfectOracleHyperModel,
    "1000":NNPerfectHyperModel,
    "0110":NNFullOracleHyperModel,
    "0100":NNOracleHyperModel,
    "0000":NNHyperModel
}

meta = get_meta(meta_ext,n=4)