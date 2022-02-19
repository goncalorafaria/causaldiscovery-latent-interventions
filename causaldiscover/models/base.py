import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math
from typing import Any, Dict, Iterable, List, Tuple
from causaldiscover.utils.model_utils import rand_score, adjusted_rand_score

def get_meta(d,n):

    left = {}
    right = {}

    cache = { "0":left, "1":right}
    middle = {}

    for k,v in d.items():

        if n == 1 :
            middle["1"==k] = v

        else :
            cache[k[0]][ k[1:] ] = v

    if n > 1 :
        return { (k == "1"): get_meta(v,n=n-1) for k,v in cache.items() }

    else: 
        if len(middle) == 2:
            return middle
        elif len(middle) == 1:
            middle[not list(middle.keys())[0]] = None
            return middle
        else:
            return {True:None,False:None}

class SkipConnection(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs

class FeedForward(nn.Module):
    def __init__(
            self,
            d_in,
            dropout=0.2):
        super().__init__()
        d_hid=d_in*4
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.act = nn.SiLU()

        #self.layer_norm = nn.LayerNorm((d_in), eps=1e-6)
    
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mollify=None):
        residual = x
        x = self.w_2(
            self.act(
                self.w_1(x)
            )
        )
        x = self.dropout(x)
        x += residual

        #x = self.layer_norm(x)

        if isinstance(mollify,torch.Tensor):
            return x + mollify * ( residual - x )
        else:
            return x
        #else:
    #        return (1.0 - mollify)*self.layer_norm(x) + residual * mollify

class AssigmentLinear(nn.Module):

    def __init__(
            self,
            in_features,
            out_features,
            assigments,
            bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(assigments, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(assigments, out_features))
        else:
            self.register_parameter('bias', None)

        self.fan_in = in_features
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x) -> torch.Tensor:
        # self.weight = assigments, out_features, in_features
        tx = torch.einsum("noi,bni->bno", self.weight, x) + self.bias
        return tx

class AssigmentwiseFeedForward(nn.Module):
    def __init__(
            self,
            d_in,
            assigments,
            dropout=0.1):
        super().__init__()

        d_hid=d_in*4
        self.w_1 = AssigmentLinear(d_in, d_hid, assigments=assigments) # position-wise
        self.w_2 = AssigmentLinear(d_hid, d_in, assigments=assigments) # position-wise
        self.act = nn.SiLU()

        #self.layer_norm = nn.LayerNorm(normalize_shape, eps=1e-6)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(self.act(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        #if isinstance(mollify,torch.Tensor):
        #    return x + mollify * ( residual - x )
        #else:
        return x
        #else:
    #        return (1.0 - mollify)*self.layer_norm(x) + residual * mollify

def approximator(
    din:int,
    dout:int,
    hdim:int=10,
    nonlinear:bool=True,
    mlplayers:int=1,
    activate_mollify:bool=True,
    hyper:bool=False,
    n=None,
    ):

    def __sequence(
        nns:nn.Module,
        x,
        mollify
        ):

        if not activate_mollify:
            mollify=None

        return nns(x)
        """
        ns = list(nns)
        x = ns[0](x)
        sx = x

        for module in ns[1:-1]:
            x = module(x, mollify=mollify)

        x = ns[-1](x)

        return x
        """

    if hyper:
        assigments = n
        if nonlinear:
            sq = [ AssigmentLinear(din, hdim, assigments=assigments) ] + \
                [ SkipConnection( nn.Sequential(*tuple( [ AssigmentwiseFeedForward(
                    hdim,
                    assigments=assigments) for _ in range(mlplayers) ])))] + \
                    [ AssigmentLinear(hdim,dout,assigments=assigments) ]
            return nn.Sequential(*tuple(sq)), lambda f : ( lambda x, mollify : __sequence(f, x) ) 
        else:
            return AssigmentLinear(din, dout,assigments=assigments), lambda f : ( lambda x, mollify : f(x) )

    else:
        if nonlinear:
            sq = [ nn.Linear(din, hdim) ] + \
                [ SkipConnection(nn.Sequential(*tuple([ FeedForward(hdim) for _ in range(mlplayers) ]))) ] + \
                    [ nn.Linear(hdim,dout) ]
            return nn.Sequential(*tuple(sq)), lambda f : ( lambda x : f(x) )
                
        else:
            return nn.Linear(din, dout), lambda f : ( lambda x, mollify : f(x) )

class HyperModel(torch.nn.Module):
    def __init__(self,
            n: int,
            k: int=10,
            target_sparsity: float=1e-3,
            weight_sparsity: float=1e-1
            ):
        super(HyperModel,self).__init__()

        self.target_sparsity = target_sparsity
        self.weight_sparsity = weight_sparsity
        self.n = n
        self.k = k
        #self.beta_ = torch.nn.Parameter( torch.Tensor(n) )
        #torch.nn.init.ones_(self.beta_)
        ### --- Dynamic weights
        ### --- --- --- --- --- --- --- ---
        ## project input into functional space


    def forward(
        self,
        z,
        x,
        temperature,
        ro,
        ground_truth_assigments = None, # here for compatibility w/ oracle models.
        deterministic: bool = False,
        mollify = None,
        mixer:float=0.0):


        batch = x.shape[0]

        (target_samples, functional_samples, \
            intervention_kl, intervention_params, expected_logbeta) = self.interventions(
            batch=batch, mollify=mollify, temperature=temperature)


        if ground_truth_assigments is None:
            (assigment_probs, assigment_kl, assigment_samples) = self.assigments(
                functional_samples=functional_samples,
                x=x,
                targets=target_samples,
                expected_logbeta=expected_logbeta,
                mollify=mollify,
                mixer=mixer,
                temperature=temperature)
        else:
            (assigment_probs, assigment_kl, assigment_samples) = self.assigments(
                functional_samples=functional_samples,
                x=x,
                expected_logbeta=expected_logbeta,
                y=ground_truth_assigments,
                mollify=mollify,
                mixer=mixer,
                temperature=temperature)
        
        model_params = self.get_params(
            z, functional_samples, target_samples,
            assigment_samples, mollify=mollify)

        ## scm forward model
        dist_params = self.scm_forward(x, model_params,mollify=mollify)

        # loss functions.
        Ldata  = self.density(x, dist_params, mollify=mollify)

        expected_extended_loss = self.assigments.estimate_expectation(
                assigment_probs=assigment_probs,
                Ldata=Ldata,
                extended_reg=0.0)

                
        #import pdb; pdb.set_trace()
        total_loss = expected_extended_loss + ro*assigment_kl + ro*intervention_kl \
                + self.weight_sparsity * self.penalize_model_complexity(model_params)# \
                #+ self.target_sparsity * target_probs.mean()

        total_loss = total_loss.mean()

        model_outputs = {
            "total_loss":total_loss,
            "z_probs":assigment_probs,
            "true_mse":Ldata,
            "h":functional_samples,
            "targets_pred":target_samples,
            "ikl":intervention_kl,
            "zkl":assigment_kl,
            "dirichlet_means":intervention_params[0],
            "dirichlet_variance":intervention_params[1],
            "alpha":intervention_params[2]
        }
        #info = {"w":model_params[0],"b":model_params[1], "target_samples":target_samples,
        #           "target_probs":target_probs, "z":z, "x":x}

        info = {}

        return total_loss, model_outputs, info

    def explain(self, model_outputs, y, targets, debug=False, x=None):

        data = {
            "loss": model_outputs["total_loss"].item(),
            "alpha": model_outputs["alpha"].item()
            }

        z_probs = model_outputs["z_probs"]
        h = model_outputs["h"]

        dists = torch.cdist(h, h, p=2.0)


        if y is not None:
            argz = torch.argmax(z_probs,dim=1)
            rand_index = rand_score(
                y.detach().flatten(),
                argz.detach().flatten(),
                torch.tensor(h.shape[1],device=h.device)
            )
            #print(type(rand_index.item()))

            
            #data["acc"] = (y==argz).float().mean().item()
            data["rand_index"] = rand_index.item()

            #import pdb; pdb.set_trace()
            data["target_sparsity"] = model_outputs["targets_pred"].sum(-1).mean().item()
            sample_target_preds = torch.einsum("btn,bt->bn", model_outputs["targets_pred"], z_probs)

            targets_loss = ( (targets-(sample_target_preds>=0.5).float()  )**2 ).mean()

            data["targets_loss"] = targets_loss.item()
            #data["targets_pred"] = model_outputs["targets_pred"].mean(0).cpu().detach()

            if debug :
                #dists = torch.cdist(h, h, p=2.0)

                if len(model_outputs["true_mse"].shape)==2:
                    mse_plot = torch.einsum(
                        "bt,bt->b",model_outputs["true_mse"].detach(), z_probs.detach()
                    ).mean()
                else:
                    mse_plot = model_outputs["true_mse"].detach().mean()

                ikl_plot = model_outputs["ikl"].detach()

                data["mse"] = mse_plot.item()
                #data["kli"] = ikl_plot.item()
                #data["klz"] = model_outputs["zkl"].mean().item()

                #data["assigment_probs0"] = z_probs[:,0].mean().item()

                #if z_probs.shape[1]>=2 :
                    #data["dists01"] = dists[0,1].item()
                #    data["assigment_probs1"] = z_probs[:,1].mean().item()

                #if z_probs.shape[1]>=3 :
                    #data["dists02"] = dists[0,2].item()
                    #data["dists12"] = dists[1,2].item()
                #    data["assigment_probs2"] = z_probs[:,2].mean().item()
                    #
                #    data["p0"] = model_outputs["dirichlet_means"][0].mean().item()
                #    data["p1"] = model_outputs["dirichlet_means"][1].mean().item()
                #    data["p2"] = model_outputs["dirichlet_means"][2].mean().item()


                #if z_probs.shape[1]>= 4 :
                    #data["dists03"] = dists[0,3].item()
                    #data["dists13"] = dists[1,3].item()
                    #data["dists23"] = dists[2,3].item()
                #    data["f3"] = z_probs[:,3].mean().item()

        return data


"""
class OracleInterventions(InterventionDirichletProcess):
    def __init__(self,
            targets : TargetBernoulli,
            hdim: int=10,
            k: int=10,
            nintv: int=2,
            a_hyper: float=4.0,
            b_hyper: float=0.4,
            alpha = None,
            nonlinear: bool=False):

        super(OracleInterventions, self).__init__(
            targets=targets,
            hdim=hdim,
            k=k,
            nintv=nintv,
            a_hyper=a_hyper,
            b_hyper=b_hyper,
            alpha=alpha,
            nonlinear=nonlinear
        )
        self.k = k

    def transform_base_samples(self, mollify:float=0.0,batch:int=0):
        h = torch.eye(n=self.k, device=self.u.device)[:(self.u.shape[0]),:]

        return h, 0.0
"""
