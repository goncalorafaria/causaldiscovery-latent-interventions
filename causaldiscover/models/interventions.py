import torch
from causaldiscover.models.base import approximator
from causaldiscover.utils.model_utils import logprobs, clamp_probs, gumblesoftmax_rsample, digamma
from causaldiscover.utils.model_utils import bernoulli_entropy,categorical_entropy, categorical_kl_divergence, beta_kl, categorical_rsamples
from torch.nn.parameter import Parameter
from typing import Any, Dict, Iterable, List, Tuple
from causaldiscover.utils.flow_utils import softplus, squareplus

class InterventionDirichletProcess(torch.nn.Module):
    def __init__(self,
                targets,
                atoms,
                hdim: int=10,
                k: int=10,
                nintv: int=2,
                alpha=None,
                nonlinear=False,
                mlpdim: int=10,
                mlplayers: int=1):

        super(InterventionDirichletProcess,self).__init__()

        self.targets = targets

        print(alpha)

        if alpha is None :
            self.alpha = Parameter(
                torch.ones(1))
        else:
            alpha = 1/(alpha+1e-7)-1
            self.register_buffer(
                'alpha', torch.ones(1)*alpha )

        self.atoms = atoms

        """
        fmodule, applier = approximator(
            k, 2,
            normalize_shape=None,
            hdim=mlpdim, nonlinear=nonlinear,
            mlplayers=mlplayers,
            activate_mollify=False
        )
        self.freemodule = fmodule
        """

        self.free_params = Parameter(
                torch.randn(nintv, 2) )

    def posterior_inference(self, h, batch:int):

        free_params = self.free_params

        p = clamp_probs(free_params[:,0].sigmoid())
        w = squareplus(
            free_params[:,1]
        )
        ones = torch.ones( [1], device=free_params.device)

        p = torch.cat(
            [ p[:-1] , ones ]
            ,dim = 0
        )

        alpha = self.alpha
        #loghyperprior = gamma_prior(alpha,self.a_hyper,self.b_hyper)

        #(iprior_probs, iposterior_probs,

        #intervention_kl = beta_kl(p, w, ones, alpha).mean()
        intervention_kl = torch.zeros(1, device=free_params.device)
        #intervention_kl = beta_rsample(p,w,alpha,batch,ones)
        #(i_prior, i_posterior, ikl, i_samples) = packed[0],packed[1],packed[2],packed[3]

        gpw = digamma( p*w )
        gpm_w = digamma( (1-p)*w )
        gw = digamma( w )

        cumgpm_w = torch.cumsum( torch.cat(
            [   torch.zeros((1),device=h.device),
                gpm_w[:-1]], dim=-1
            )-gw, -1)

        expected_logbeta = gpw + cumgpm_w

        ipars = (p,w,alpha)

        return intervention_kl, ipars, expected_logbeta

    def forward(self,
            batch:int,
            mollify:float=0.0,
            temperature:float=1.0,
            deterministic:bool=False):

        functional_samples, atom_kl = self.atoms(mollify=mollify,batch=batch)

        (intervention_kl,
            ipars, expected_logbeta) = self.posterior_inference(
                functional_samples, batch=batch)

        (target_probs, target_kl, target_samples) = self.targets(
            functional_samples, batch=batch, deterministic=deterministic,
            temperature=temperature, mollify=mollify
        )

        kl = intervention_kl + target_kl + atom_kl

        return target_samples, functional_samples, kl, ipars, expected_logbeta

class TargetBernoulli(torch.nn.Module):
    def __init__(
            self,
            n,
            nintv,
            hard=True,
            hdim=10,
            mlplayers=1,
            nonlinear=False,
            prior=0.5,
            hardcodedobs=True):

        super(TargetBernoulli,self).__init__()

        self.hard = hard

        self.r = Parameter(
                torch.randn(
                    nintv,
                    n) #+ 0.5 
            )

        self.forward = self.posterior_inference
        
        self.register_buffer(
            'prior', clamp_probs(torch.ones(1)*prior))

        self.register_buffer(
            'zeros', torch.zeros( (1,n) ) )

        self.hardcodedobs = hardcodedobs

    def get_params(self, h, mollify):
        probs = clamp_probs(
                    self.r.sigmoid()
                )

        if self.hardcodedobs: 
            probs = torch.cat(
                [self.zeros,
                probs[1:]],
                axis=0)

        return probs

    def get_kl(self,target_probs):
        #kl = target_probs * (logprobs(target_probs) - logprobs(self.prior)) + \
        #    (1-target_probs) * ( logprobs(1-target_probs) - logprobs(1-self.prior))

        kl = - torch.logit(self.prior)*target_probs

        tsparsity = target_probs.sum(-1)

        kl += 2 * torch.maximum(
            - tsparsity**2 , -torch.ones_like(tsparsity) )[1:].sum()

        if self.hardcodedobs:
            target_kl = kl.sum(-1)[1:]
        else:
            target_kl = kl.sum(-1)

        return target_kl.sum()

    def posterior_inference(self, h, batch:int, deterministic: bool, temperature, mollify:float=0.0):
        target_probs = self.get_params(h,mollify)

        if deterministic:
            target_samples = (target_probs > 0.5).float()
            target_samples = target_samples.unsqueeze(0).repeat(batch,1,1)
        else:
            target_samples = gumblesoftmax_rsample(probs=target_probs, temperature=temperature, shape=(batch,))
            
            if self.hard:
                target_samples = target_samples - target_samples.detach() + (target_samples>0.5)
            else:
                target_samples = target_samples

        target_kl = self.get_kl(target_probs) # uniform prior.

        return target_probs, target_kl, target_samples


class TargetSingle(torch.nn.Module):
    def __init__(
        self,
        n,
        nintv,
        hard=True,
        hdim=10,
        mlplayers=1,
        nonlinear=False,
        prior=0.5):

        super(TargetSingle,self).__init__()

        self.hard = hard

        self.r = Parameter(
                torch.randn(
                    nintv,
                    n)
            )

        self.forward = self.posterior_inference

        self.register_buffer(
            'zeros', torch.zeros( (1,n) ) )

    def get_params(self, h, mollify):

        probs = self.r.softmax(axis=-1)
                
        probs = torch.cat(
            [self.zeros,
            probs[1:]],
            axis=0)

        return probs

    def get_kl(self,target_probs):

        kl = -categorical_entropy(target_probs)

        target_kl = kl[1:]

        return target_kl.sum()

    def posterior_inference(self, h, batch:int, deterministic: bool, temperature, mollify:float=0.0):
        target_probs = self.get_params(h,mollify)

        if deterministic:
            target_samples = (target_probs > 0.5).float()
            target_samples = target_samples.unsqueeze(0).repeat(batch,1,1)
        else:
            target_samples = categorical_rsamples(probs=target_probs[1:], temperature=temperature, shape=(batch,))

            target_samples= torch.cat(
                [self.zeros.unsqueeze(0).repeat(batch,1,1),
                target_samples],
                axis=1
            )
            
            if self.hard:
                target_samples = target_samples - \
                    target_samples.detach() + \
                    (target_samples >= torch.max(target_samples,dim=-1)[0].unsqueeze(-1))

        target_kl = self.get_kl(target_probs) # uniform prior.

        return target_probs, target_kl, target_samples


class OracleTarget(torch.nn.Module):
    def __init__(self, ground_truth):
        super(OracleTarget,self).__init__()

        self.register_buffer(
            "ground_truth",torch.tensor(
                ground_truth, dtype=torch.float)
                )

        self.forward = self.posterior_inference

    def posterior_inference(self, batch:int):

        batched = self.ground_truth.unsqueeze(0).repeat( (batch,1,1) )

        return self.ground_truth, 0.0, batched

class TargetBernoulliExact(TargetBernoulli):
    def __init__(
            self,
            n,
            k,
            hard=True,
            hdim=10,
            mlplayers=1,
            nonlinear=False,
            prior=0.5):

        super(TargetBernoulliExact,self).__init__(
            n=n,k=k,hard=hard,hdim=hdim,mlplayers=mlplayers,
            nonlinear=nonlinear, prior=prior
        )
        self.forward = self.posterior_expected_value

    def posterior_expected_value(self, h, batch:int, deterministic: bool, temperature, mollify:float=0.0):
        target_probs = self.get_params(h,mollify)
        target_kl = self.get_kl(target_probs) # uniform prior.

        return target_probs, target_kl, target_probs

class AtomNormal(torch.nn.Module):

    def __init__(self,
        hdim:int = 10,
        k:int = 10,
        mlpdim:int = 10,
        nintv: int =10,
        nonlinear:bool = False,
        mlplayers:int=2,
        activate_mollify:bool = False,
        kl:bool=False):

        super(AtomNormal, self).__init__()

        nonlinear = False

        self.muu = Parameter(
                torch.randn(
                    nintv,
                    hdim) #* 2
            )

        self.stu = Parameter(
                torch.randn(
                    nintv,
                    hdim) #* 2
            )
        self.kl = kl
        """
        u = torch.randn(
            nintv,
            hdim)

        self.register_buffer(
            'u', u)
        """

        fmodule, applier = approximator(
            hdim, k,
            hdim=mlpdim, nonlinear=nonlinear,
            mlplayers=mlplayers,
            activate_mollify=activate_mollify
        )
        self.fmodule = fmodule
        self.M = applier(self.fmodule)

    def forward(self, batch:int, mollify:float=0.0):

        #
        #hparms = h.view(self.u.shape[0],h.shape[1]//2,2)
        hparms_mu = self.muu
        hparms_scale = squareplus(self.stu)

        posterior = torch.distributions.normal.Normal(
                hparms_mu, hparms_scale, validate_args=False)

        samples = posterior.rsample((batch,))
        h = self.M(samples,mollify=mollify)

        prior = torch.distributions.normal.Normal(
            torch.zeros_like(hparms_mu),
            torch.ones_like(hparms_scale)
        )

        if self.kl:
            kl = torch.distributions.kl.kl_divergence(
                posterior,prior).sum(-1).mean()
        else:
            kl = 0.0

        return h, kl

class OracleInterventionDirichletProcess(InterventionDirichletProcess):
    def __init__(self,
                ground_truth_targets,
                atoms,
                hdim: int=10,
                k: int=10,
                nintv: int=2,
                alpha=None,
                nonlinear=False,
                mlpdim: int=10,
                mlplayers: int=1):

        super(OracleInterventionDirichletProcess,self).__init__(
            targets=OracleTarget(ground_truth=ground_truth_targets),
            atoms=atoms,
            hdim=hdim,
            k=k,
            nintv=nintv,
            alpha=alpha,
            nonlinear=nonlinear,
            mlpdim=mlpdim,
            mlplayers=mlplayers
        )

    def forward(self,
            batch:int,
            mollify:float=0.0,
            temperature:float=1.0,
            deterministic:bool=False):

        functional_samples, atom_kl = self.atoms(mollify=mollify,batch=batch)

        (intervention_kl,
            ipars, expected_logbeta) = self.posterior_inference(
                functional_samples, batch=batch)

        (target_probs, target_kl, target_samples) = self.targets(
            batch=batch
        )

        kl = intervention_kl + target_kl + atom_kl

        return target_samples, functional_samples, kl, ipars, expected_logbeta
