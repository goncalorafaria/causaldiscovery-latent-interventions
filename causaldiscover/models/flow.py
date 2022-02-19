import torch
from torch import nn
from typing import Any, Dict, Iterable, List, Tuple
from torch.autograd import Variable

from causaldiscover.models.base import HyperModel,approximator, get_meta
from causaldiscover.models.assigments import OracleAssigments, AssigmentsCategorical, StochasticAssigments, OracleStochasticAssigments
from causaldiscover.models.interventions import InterventionDirichletProcess, TargetBernoulliExact, TargetBernoulli, AtomNormal,  OracleInterventionDirichletProcess

from causaldiscover.utils.model_utils import logprobs
from causaldiscover.utils.flow_utils import SigmoidFlow, log_normal


class DeepFlowHyperModel(HyperModel):
    def __init__(self,
            n: int,
            hdim: int=10,
            k: int=10,
            target_sparsity: float=1e-3,
            weight_sparsity: float=1e-1,
            alpha = None,
            nintv: int=2,
            use_z_entropy: bool= False,
            mlpdim:int=40,
            mlplayers:int=2,
            flowlayers:int=2,
            flowdim:int=10,
            sharedpar:int=2,
            nonlinear:bool=True,
            hard:bool=False,
            prior:float=0.5):

        super(DeepFlowHyperModel,self).__init__(
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
                nintv=nintv,
                mlplayers=mlplayers,
                kl=True
            ),
            hdim=hdim,k=k,alpha=alpha,
            nintv=nintv,
            nonlinear=nonlinear
        )

        self.assigments = StochasticAssigments(
            n=n, k=k, use_z_entropy=use_z_entropy,
            nonlinear=nonlinear, hard=hard,nintv=nintv
        )

        self.flowlayers = flowlayers
        self.flowdim = flowdim

        fmodule, applier = approximator(
            self.n+self.k,
            3*self.flowlayers*self.flowdim,
            hdim=mlpdim,
            nonlinear=nonlinear,
            mlplayers=mlplayers,
            hyper=True,
            activate_mollify=True,
            n=n)


        self.predictor = fmodule

        for i in range(self.flowlayers):
            s = SigmoidFlow(n=self.n, sharedpar=sharedpar)
            #self.__dict__["flowmodel"+str(i)] = s
            self.add_module("flowmodel"+str(i),s)
        #self.flowmodel = [ SigmoidFlow(n=self.n, sharedpar=sharedpar) for _ in range(self.flowlayers) ]


    def get_params(self, z, h, target_samples, assigment_samples, mollify:float=0.0):

        single_h = torch.einsum("bt,btk->bk", assigment_samples, h).unsqueeze(1)
        single_r = torch.einsum("bt,btj->bj", assigment_samples, target_samples).unsqueeze(-1)

        single_tile_h = single_h.repeat([1,self.n,1])
        single_tile_r = single_r.repeat([1,1,single_r.shape[1]])

        relaxed_h_sample = single_r*single_tile_h + (1.0-single_r)*h[:,:1,:]

        i = torch.argmax(assigment_samples,dim=-1)
        campled_h_sample = h[i]

        return relaxed_h_sample, z, campled_h_sample

    def scm_forward(self, 
            x, 
            model_params:Tuple[torch.Tensor,torch.Tensor],
            mollify:float=0.0):

        relaxed_h_sample, z, campled_h_sample = model_params

        if len(z.shape) == 2 :
            z = z.unsqueeze(0)

        x_expand = x.unsqueeze(1).repeat([1,self.n,1]) * z
        x_input = torch.cat([x_expand,relaxed_h_sample],axis=-1)
        # x_input : [ b x n x (n + k) ]

        a = self.predictor(x_input)

        return a

    def density(self, x, dist_params:torch.Tensor, mollify:float=0.0):


        dist_params = dist_params.view(list(dist_params.shape)[:-1]+[3, self.flowdim, self.flowlayers])

        zs = torch.zeros(
            (x.shape[0], self.n),
            device=dist_params.device
        )

        logdet = zs

        for i in range(self.flowlayers):
            x, logdet = self._modules["flowmodel"+str(i)].forward(
                x, logdet,
                dist_params[:,:,:,:,i],
                mollify=mollify)

        pseudo_joint_nll = log_normal(x, zs, zs) + logdet

        return -pseudo_joint_nll.sum(-1)

    def penalize_model_complexity(self, model_params:Tuple[torch.Tensor,torch.Tensor]):
        return torch.zeros((),device=model_params[0].device)

class DeepFlowOracleHyperModel(DeepFlowHyperModel):
    def __init__(self,
            n: int,
            hdim: int=10,
            k: int=10,
            target_sparsity: float=1e-3,
            weight_sparsity: float=1e-1,
            alpha = None,
            nintv: int=2,
            use_z_entropy: bool= False,
            mlpdim:int=40,
            mlplayers:int=2,
            flowlayers:int=2,
            flowdim:int=10,
            sharedpar:int=2,
            nonlinear:bool=True,
            hard:bool=False,
            prior:float=0.5):

        super(DeepFlowOracleHyperModel,self).__init__(
            n=n,
            hdim=hdim,
            k=k,
            target_sparsity=target_sparsity,
            weight_sparsity=weight_sparsity,
            alpha = None,
            nintv = nintv,
            use_z_entropy=use_z_entropy,
            mlpdim=mlpdim,
            mlplayers=mlplayers,
            flowlayers=flowlayers,
            flowdim=flowdim,
            sharedpar=sharedpar,
            nonlinear=nonlinear,
            hard=hard,
            prior=prior)

        self.assigments = OracleStochasticAssigments()

class DeepFlowFullOracleHyperModel(DeepFlowOracleHyperModel):
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
            mlpdim:int=40,
            mlplayers:int=2,
            flowlayers:int=2,
            flowdim:int=10,
            sharedpar:int=2,
            nonlinear:bool=True,
            hard:bool=False,
            prior:float=0.5):

        super(DeepFlowFullOracleHyperModel,self).__init__(
            n=n,
            hdim=hdim,
            k=k,
            target_sparsity=target_sparsity,
            weight_sparsity=weight_sparsity,
            alpha = None,
            nintv = nintv,
            use_z_entropy=use_z_entropy,
            mlpdim=mlpdim,
            mlplayers=mlplayers,
            flowlayers=flowlayers,
            flowdim=flowdim,
            sharedpar=sharedpar,
            nonlinear=nonlinear,
            hard=hard,
            prior=prior)

        self.interventions = OracleInterventionDirichletProcess(
            ground_truth_targets=ground_truth_targets,
            atoms=AtomNormal(
                hdim=hdim,
                k=k,
                mlpdim=mlpdim,
                nonlinear=nonlinear,
                mlplayers=mlplayers,
                nintv=nintv,
                kl=False
            ),
            hdim=hdim,
            k=k,
            alpha=alpha,
            nintv=nintv,
            nonlinear=nonlinear,
            mlplayers=mlplayers,
            mlpdim=mlpdim
        )

class DeepFlowPerfectHyperModel(DeepFlowHyperModel):
    def __init__(self,
            n: int,
            hdim: int=10,
            k: int=10,
            target_sparsity: float=1e-3,
            weight_sparsity: float=1e-1,
            alpha = None,
            nintv: int=2,
            use_z_entropy: bool= False,
            mlpdim:int=40,
            mlplayers:int=2,
            flowlayers:int=2,
            flowdim:int=10,
            sharedpar:int=2,
            nonlinear:bool=True,
            hard:bool=False,
            prior:float=0.5,
            atomic:bool=False):

        super(DeepFlowPerfectHyperModel,self).__init__(
            n=n,
            hdim=hdim,
            k=k,
            target_sparsity=target_sparsity,
            weight_sparsity=weight_sparsity,
            alpha = alpha,
            nintv = nintv,
            use_z_entropy=use_z_entropy,
            mlpdim=mlpdim,
            mlplayers=mlplayers,
            flowlayers=flowlayers,
            flowdim=flowdim,
            sharedpar=sharedpar,
            nonlinear=nonlinear,
            hard=hard,
            prior=prior)

        self.atomic = atomic
        
        fmodule, applier = approximator(
                self.n,
                3*self.flowlayers*self.flowdim,
                hdim=mlpdim,
                nonlinear=nonlinear,
                mlplayers=mlplayers,
                hyper=True,
                activate_mollify=True,
                n=n)

        self.predictor = fmodule

        fb, applier = approximator(
            k, n,
            hdim=mlpdim, nonlinear=nonlinear,
            mlplayers=mlplayers,
            activate_mollify=True,
            n=n,
        )

        self.b_proj= fb

        fbeta, applier = approximator(
            k, n,
            hdim=mlpdim, nonlinear=nonlinear,
            mlplayers=mlplayers,
            activate_mollify=True,
            n=n
        )

        self.beta_proj= fbeta

    def get_params(self, z, h, target_samples, assigment_samples, mollify:float=0.0):
        
        b = self.b_proj(h) # [B, t, d]
        beta = self.beta_proj(h) # [B, t, d]
        
        b_i = torch.einsum("bt,btj->bj",  assigment_samples, b) # [B, t]
        beta_i = torch.einsum("bt,btj->bj",  assigment_samples, beta) # [B, t]

        single_h = torch.einsum("bt,btk->bk", assigment_samples, h).unsqueeze(1)
        single_r = torch.einsum("bt,btj->bj", assigment_samples, target_samples).unsqueeze(-1)

        single_tile_h = single_h.repeat([1,self.n,1])
        single_tile_r = single_r.repeat([1,1,single_r.shape[1]])

        relaxed_h_sample = single_r*single_tile_h + (1.0-single_r)*h[:,:1,:]

        zexp_r = torch.einsum("bt,btj->bj",assigment_samples, target_samples)

        z_reduced = z * ( 1 - zexp_r.unsqueeze(-1) )

        i = torch.argmax(assigment_samples,dim=-1)
        campled_h_sample = h[i]

        return  z_reduced, zexp_r, b_i, beta_i


    def scm_forward(self, 
            x, 
            model_params:Tuple[torch.Tensor,torch.Tensor],
            mollify:float=0.0):
            
        z, zexp_r, b_i, beta_i = model_params

        if len(z.shape) == 2 :
            z = z.unsqueeze(0)

        x_input = x.unsqueeze(1).repeat([1,self.n,1]) * z
        # x_input : [ b x n x (n + k) ]

        a = self.predictor(x_input)

        return a, zexp_r, b_i, beta_i


    def density(self, x, dist_params:torch.Tensor, mollify:float=0.0):
        a, zexp_r, b_i, beta_i = dist_params
        dist_params = a

        dist_params = dist_params.view(
            list(dist_params.shape)[:-1]+[3, self.flowdim, self.flowlayers]
            )

        zs = torch.zeros(
            (x.shape[0], self.n),
            device=dist_params.device
        )

        logdet = zs

        for i in range(self.flowlayers):
            x, logdet = self._modules["flowmodel"+str(i)].forward(
                x, logdet,
                dist_params[:,:,:,:,i],
                mollify=mollify)

        pseudo_joint_nll = log_normal(x, zs, zs) + logdet

        #if self.atomic :
        #    beta_i = torch.ones_like(beta_i)*0.1
             
        #interv_function_nll = log_normal(x, b_i, beta_i )

        #joint_nll = (1-zexp_r)*pseudo_joint_nll + zexp_r*interv_function_nll

        joint_nll = pseudo_joint_nll

        return -joint_nll.sum(-1)

class DeepFlowPerfectOracleHyperModel(DeepFlowPerfectHyperModel):
    def __init__(self,
            n: int,
            hdim: int=10,
            k: int=10,
            target_sparsity: float=1e-3,
            weight_sparsity: float=1e-1,
            alpha = None,
            nintv: int=2,
            use_z_entropy: bool= False,
            mlpdim:int=40,
            mlplayers:int=2,
            flowlayers:int=2,
            flowdim:int=10,
            sharedpar:int=2,
            nonlinear:bool=True,
            hard:bool=False,
            prior:float=0.5,
            atomic:bool=False):

        super(
            DeepFlowPerfectOracleHyperModel,
            self).__init__(
                n=n,
                hdim=hdim,
                k=k,
                target_sparsity=target_sparsity,
                weight_sparsity=weight_sparsity,
                alpha = alpha,
                nintv = nintv,
                use_z_entropy=use_z_entropy,
                mlpdim=mlpdim,
                mlplayers=mlplayers,
                flowlayers=flowlayers,
                flowdim=flowdim,
                sharedpar=sharedpar,
                nonlinear=nonlinear,
                hard=hard,
                prior=prior,
                atomic=atomic)

        self.assigments = OracleStochasticAssigments()

class DeepFlowFullPerfectOracleHyperModel(DeepFlowPerfectOracleHyperModel):
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
            mlpdim:int=40,
            mlplayers:int=2,
            flowlayers:int=2,
            flowdim:int=10,
            sharedpar:int=2,
            nonlinear:bool=True,
            hard:bool=False,
            prior:float=0.5,
            atomic:bool=False):

        super(DeepFlowFullPerfectOracleHyperModel,
                self).__init__(
                    n=n,
                    hdim=hdim,
                    k=k,
                    target_sparsity=target_sparsity,
                    weight_sparsity=weight_sparsity,
                    alpha = alpha,
                    nintv = nintv,
                    use_z_entropy=use_z_entropy,
                    mlpdim=mlpdim,
                    mlplayers=mlplayers,
                    flowlayers=flowlayers,
                    flowdim=flowdim,
                    sharedpar=sharedpar,
                    nonlinear=nonlinear,
                    hard=hard,
                    prior=prior,
                    atomic=atomic)
        
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

meta_ext = {
    "1110":DeepFlowFullPerfectOracleHyperModel,
    "1100":DeepFlowPerfectOracleHyperModel,
    "1000":DeepFlowPerfectHyperModel,
    "0110":DeepFlowFullOracleHyperModel,
    "0100":DeepFlowOracleHyperModel,
    "0000":DeepFlowHyperModel
}

meta = get_meta(meta_ext,n=4)

