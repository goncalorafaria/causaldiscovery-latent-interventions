
import torch
import math
from causaldiscover.models.base import approximator

from causaldiscover.utils.model_utils import logprobs, clamp_probs, gumblesoftmax_rsample
from causaldiscover.utils.model_utils import bernoulli_entropy,categorical_entropy, categorical_kl_divergence, beta_rsample, categorical_rsamples

class AssigmentsCategorical(torch.nn.Module):
    def __init__(self,
                 n,
                 k,
                 use_z_entropy=False,
                 nonlinear=False,
                 hard=True,
                 mlplayers=1,
                 hdim=10,
                 mix=False,
                 nintv=3):

        super(AssigmentsCategorical,self).__init__()

        print("using mixxing: " + str(mix) )
        self.hard=hard
        self.use_z_entropy = use_z_entropy
        self.mix = mix

        fmodule, applier = approximator(
            n, k,
            hdim=k, nonlinear=nonlinear,
            mlplayers=mlplayers,
            activate_mollify=True
        )
        #self.fmodule = fmodule
        self.T = fmodule

        self.k = k
        self.forward = self.posterior_inference
        self.sattention = math.sqrt(k)

    def posterior_inference(self, 
        functional_samples, 
        x, 
        expected_logbeta, 
        targets, 
        mollify:float=0.0, 
        mixer:float=1.0,
        temperature:float=1.0,
        colapse = True ):

        """
        hs = torch.cat(
            [
                x.unsqueeze(1).repeat([1,functional_samples.shape[1],1]),
                functional_samples
            ],
            axis=-1
        )
        """

        #hs = x.unsqueeze(1).repeat([1,functional_samples.shape[1],1])
        hs_proj = self.T(x) # bxk
        #torch.einsum("btk,bk->bt", h, x_proj)/math.sqrt(self.k)
        expected_logbeta = expected_logbeta.unsqueeze(0)

        assigment_params = torch.einsum("bk,btk->bt",hs_proj, functional_samples)/ self.sattention #+ expected_logbeta
        assigment_probs = torch.softmax(assigment_params, dim=-1)
        
        if self.mix and colapse :
            assigment_kl_beta = ( assigment_probs * ( logprobs(assigment_probs) - expected_logbeta ) ).sum(-1)
            assigment_kl_ent = -categorical_entropy(assigment_probs)

            assigment_kl = mixer*assigment_kl_beta + (1-mixer)*assigment_kl_ent
        else:
            if not self.use_z_entropy :
                assigment_kl = ( logprobs(assigment_probs) - expected_logbeta ) 
            else:
                assigment_kl = logprobs(assigment_probs)

        assigment_samples = categorical_rsamples(
                probs=assigment_probs,
                temperature=temperature)

        if self.hard:
            assigment_samples = assigment_samples - \
                assigment_samples.detach() + \
                (assigment_samples >= torch.max(assigment_samples,dim=-1)[0].unsqueeze(-1))

        if colapse :
            assigment_kl = (assigment_probs *  assigment_kl).sum(-1)

        return assigment_probs, assigment_kl, assigment_samples

    def estimate_expectation(self, assigment_probs, Ldata, extended_reg):

        extended_loss = Ldata + extended_reg
        estimate = (extended_loss*assigment_probs).sum(-1)
        return estimate

class OracleAssigments(torch.nn.Module):
    def __init__(self):
        super(OracleAssigments,self).__init__()
        self.forward = self.posterior_inference

    def posterior_inference(self,
            functional_samples,
            x,
            expected_logbeta,
            y,
            mollify:float=0.0,
            mixer:float=0.0,
            temperature:float=1.0):

        assigment_samples=torch.nn.functional.one_hot(y, num_classes=functional_samples.shape[1]).float()
                 
        return assigment_samples, torch.zeros((),device=functional_samples.device), assigment_samples

    def estimate_expectation(self, assigment_probs, Ldata, extended_reg):

        estimate_kl = (extended_reg*assigment_probs).sum(-1)

        index = assigment_probs.argmax(axis=1,keepdim=True)
        estimate = torch.gather(Ldata, index=index, axis=1)

        return estimate.unsqueeze(-1) + estimate_kl

class StochasticAssigments(AssigmentsCategorical):
    def __init__(self,
                 n,
                 k,
                 use_z_entropy=False,
                 nonlinear=False,
                 hard=True,
                 mix=False,
                 nintv=3,
                 hdim=10):

        super(StochasticAssigments,self).__init__(
            n=n,
            k=k,
            use_z_entropy=use_z_entropy,
            nonlinear=nonlinear,
            hard=hard,
            mix=mix,
            nintv=nintv,
            hdim=hdim
        )

    def estimate_expectation(self, assigment_probs, Ldata, extended_reg):
        estimate = Ldata + (extended_reg*assigment_probs).sum(-1)
        return estimate

class OracleStochasticAssigments(OracleAssigments):
    def __init__(self):
        super(OracleStochasticAssigments,self).__init__()

    def estimate_expectation(self, assigment_probs, Ldata, extended_reg):
        estimate = Ldata + (extended_reg*assigment_probs).sum(-1)
        return estimate


class AssigmentsSSL(StochasticAssigments):
    def __init__(self,
                n,
                k,
                use_z_entropy=False,
                nonlinear=False,
                hard=True,
                mix=False,
                nintv=3,
                hdim=10,
                eps=1e-2,
                alpha=0.1):

        super(AssigmentsSSL,self).__init__(
            n=n,
            k=k,
            use_z_entropy=use_z_entropy,
            nonlinear=nonlinear,
            hard=hard,
            mix=mix,
            nintv=nintv,
            hdim=hdim
        )

        self.eps = eps
        
        self.forward = self.posterior_inference

        self.alpha = alpha

    
    def posterior_inference(self,
            functional_samples,
            x,
            expected_logbeta,
            y,
            mollify:float=0.0,
            mixer:float=0.0,
            temperature:float=1.0):
        
        missing = (y==-1).long()

        # unlabelled posterior and samples.
        assigment_probs_z, assigment_kl, assigment_samples_z = super(AssigmentsSSL,self).posterior_inference(
            functional_samples=functional_samples, 
            x=x, 
            expected_logbeta=expected_logbeta, 
            targets=None, 
            mollify=mollify, 
            mixer=mixer,
            temperature=temperature,
            colapse = True)

        #import pdb; pdb.set_trace()
        # labelled posterior and samples
        y = y * (1 - missing) 

        y = y.long()

        
        assigment_samples_true = torch.nn.functional.one_hot(y, num_classes=functional_samples.shape[1]).float()
        assigment_probs_smooth = assigment_samples_true * ( 1 - self.eps) - \
          ( torch.ones_like(assigment_samples_true) * self.eps/assigment_samples_true.shape[-1] )
        

        missing = missing.unsqueeze(-1)
        # ssl posterior and samples.
        assigment_probs = (1-missing)*assigment_probs_smooth + missing*assigment_probs_z
        #assigment_probs_kl = (-1)*(1-missing)*assigment_probs_smooth + missing*assigment_probs_z
        assigment_samples = (1-missing)*assigment_samples_true + missing*assigment_samples_z

        negcrossent = ( - (1-missing) * assigment_probs_smooth * logprobs( assigment_probs_z ) ).sum(-1)

        return assigment_probs, assigment_kl + self.alpha * negcrossent, assigment_samples


class AssigmentsSSLExtended(AssigmentsSSL):
    def __init__(self,
                n,
                k,
                use_z_entropy=False,
                nonlinear=False,
                hard=True,
                mix=False,
                nintv=3,
                hdim=10,
                eps=10e-2):

        super(AssigmentsSSLExtended,self).__init__(
            n=n,
            k=k,
            use_z_entropy=use_z_entropy,
            nonlinear=nonlinear,
            hard=hard,
            mix=mix,
            nintv=nintv,
            hdim=hdim,
            eps=eps
        )


    def estimate_expectation(self, assigment_probs, Ldata, extended_reg):

        extended_loss = Ldata + extended_reg
        estimate = (extended_loss*assigment_probs).sum(-1)
        return estimate
    