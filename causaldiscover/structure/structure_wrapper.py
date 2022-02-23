from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
from torch.distributions.bernoulli import Bernoulli
from causaldiscover.utils.model_utils import clamp_probs
import torch

class CausalStructure(torch.nn.Module):
    def __init__(self,
        scm,
        graph_prior=0.5,
        estimator="r"):

        super(CausalStructure,self).__init__()

        self.n = scm.n
        #self.graph_sparsity = graph_sparsity
        self.gamma = torch.nn.Parameter(
            torch.randn(self.n,self.n)
        )
        self.scm = scm
        self.estimator = estimator

        self.register_buffer(
                'graph_prior',torch.ones([self.n,self.n])*graph_prior)


    def get_probs(self):
        masked_potencial = self.gamma.clone()
        masked_potencial[~self.edge_mask] = float('-inf')
        return clamp_probs(masked_potencial.sigmoid())

    def get_graph(
        self,
        batch,
        temperature):

        probs = self.get_probs()

        bern = RelaxedBernoulli(
            temperature=temperature,
            probs=probs)

        if self.estimator == "r":
            r_samples = probs

        elif self.estimator == "st":
            r_samples = probs
            r_samples = r_samples - r_samples.detach() + (r_samples > 0.5)

        elif self.estimator == "1-g":
            r_samples = bern.rsample()

        elif self.estimator == "1-gst":
            r_samples = bern.rsample()
            r_samples = r_samples - r_samples.detach() + (r_samples > 0.5)

        elif self.estimator == "b-g":
            r_samples = bern.rsample((batch,))

            #raise ValueError(" not implemented yet")

        elif self.estimator == "b-gst":
            r_samples = bern.rsample((batch,))
            r_samples = r_samples - r_samples.detach() + (r_samples > 0.5)

        else:
            raise ValueError(" not available. ")

        kl = torch.distributions.kl.kl_divergence(
              Bernoulli(probs=probs),
              Bernoulli(probs=self.graph_prior)
        )

        #import pdb; pdb.set_trace()
        #kl = - torch.logit(self.graph_prior)*probs
        kl[~self.edge_mask] = 0.0

        kl = kl.sum()

        return r_samples, kl

    def forward(self, x, y=None, **args):

        batch_size = x.shape[0]

        x_ordered, P, kl = self.permute(x, temperature=args["temperature"])

        bit_vector_z, encoder_entropy = \
            self.get_graph(
                batch=batch_size,
                temperature=args["temperature"])

        loss, model_outputs, info = self.scm(
            z = bit_vector_z,
            x = x_ordered,
            ground_truth_assigments = y,
            **args)

        graph_regularizer1, graph_regularizer2, delta = self.regularizer(
            bit_vector_z, args["mixer"])

        full_loss = loss + encoder_entropy + kl.mean() \
            + graph_regularizer1 + graph_regularizer2

        model_outputs['loss'] = full_loss.item()
        model_outputs['encoder_entropy'] = encoder_entropy.item()
        model_outputs['structure_pred'] = bit_vector_z
        model_outputs['P'] = P
        probs = self.get_probs()
        model_outputs['probs01'] = probs[0,1].item()
        model_outputs['probs10'] = probs[1,0].item()

        model_outputs["delta_constraint"] = delta.item()

        return full_loss, model_outputs, info

    def explain(self,
        model_outputs,
        targets,
        y,
        cmodel,
        debug=False):

        data = self.scm.explain(
            model_outputs,
            targets=targets,
            y=y,
            debug=debug
        )

        structure_pred = self.get_probs()
        P = model_outputs["P"]

        if P is not None:
            structure_pred = torch.einsum(
                "bnx, nm, bmy -> bxy",
                P,
                structure_pred,
                P
            )
            x1 = cmodel.reshape(-1).unsqueeze(0).unsqueeze(0).tile((structure_pred.shape[0],1,1))
            x2 = structure_pred.unsqueeze(0).reshape((structure_pred.shape[0],1,-1))
            #structure_pred  = P.T @ structure_pred @ P
        else:
            x1 = cmodel.reshape(-1).unsqueeze(0).unsqueeze(0).tile((1,1,1))
            x2 = structure_pred.unsqueeze(0).reshape((1,1,-1))

        hamming_dist = torch.cdist(x1,x2,p=1).mean()
        data["sparsity"] = (x2 > 0.5).sum().item()
        data["hamming_dist"] = hamming_dist
        data["delta_constraint"] = model_outputs["delta_constraint"]
        #data["probs01"] = model_outputs["probs01"]
        #data["probs10"] = model_outputs["probs10"]
        data = self.log_state(data)

        return data

    def update(self, h):
        return None

    def log_state(self, model_outputs):
        return model_outputs

    def regularizer(self, z, mixer_percentage):
        return 0.0, 0.0, torch.zeros(1,device=z.device)

    def permute(self, x, temperature):
        return x, None, torch.zeros(1,device=x.device)

class ConstraintMethod(CausalStructure):
    def __init__(self,
        scm,
        graph_prior,
        estimator="r",
        lambda_start=0.0,
        c0=1e-8,
        eta=1.5,
        delta=0.9):

        super(ConstraintMethod, self).__init__(
            scm=scm,
            graph_prior=graph_prior,
            estimator=estimator
            )

        mask = torch.ones((self.n,self.n))-torch.eye(self.n)

        self.register_buffer(
                'edge_mask', mask.bool())

        self.register_buffer(
                'ntensor',
                torch.ones(
                    1,
                    dtype=torch.int,
                    requires_grad=False
                    )*self.n
                )

        self.register_buffer(
                'h_cache',
                torch.ones(
                    1,
                    dtype=torch.float,
                    requires_grad=False)*0.01
                )

        self.register_buffer(
                'lambda_',
                torch.ones(
                    1,
                    dtype=torch.int,
                    requires_grad=False
                    )*lambda_start
                )

        self.register_buffer(
                'c',
                torch.ones(
                    1,
                    dtype=torch.int,
                    requires_grad=False
                    )*c0
                )

        self.register_buffer(
                'lambda_start',
                    torch.ones(
                        1,
                        dtype=torch.float
                        )*lambda_start
                    )

        full_adjacency = torch.ones((scm.n,scm.n)) - torch.eye(scm.n)

        self.register_buffer(
                'constraint_normalization',
                 self.compute_constraint(full_adjacency)
                 )

        self.delta = delta
        self.eta = eta
        #self.lambda_ = torch.nn.Parameter(
    #        torch.ones(1,dtype=torch.float)*lambda_start
    #    )

    def log_state(self, model_outputs):

        #model_outputs["lambda"]= self.lambda_.item()
        #model_outputs["c"]= self.c.item()

        return model_outputs

    def update(self, h):
        #h = h.detach()

        self.lambda_ += h*self.c
        #self.lambda_ = torch.maximum(self.lambda_ ,self.lambda_start)

        if h > self.h_cache * self.delta:
            self.c *= self.eta

        self.h_cache = h

    def regularizer(self, z, mixer_percentage):

        masked_potencial = self.gamma.clone()
        masked_potencial[~self.edge_mask] = float('-inf')
        probs = masked_potencial.sigmoid()

        h = self.compute_constraint(probs)/ self.constraint_normalization

        # *self.c.detach()
        reg = self.lambda_.detach()*h + self.c.detach()/2.0 * h**2

        return 0.0, reg, h

    def compute_constraint(self, probs):
        #import pdb; pdb.set_trace()
        
        h = ( torch.diagonal(torch.matrix_exp(probs), dim1=-2, dim2=-1).sum(-1) - self.ntensor).mean()

        #h = (torch.diagonal(torch.matrix_power(probs,2), dim1=-2, dim2=-1).sum(-1)).mean()

        return h


    def get_graph(
        self,
        batch,
        temperature):

        r_samples, kl = super(ConstraintMethod,self).get_graph(
            batch, temperature)

        probs = super(ConstraintMethod,self).get_probs()

        kl2 = -(torch.logit(self.graph_prior)*probs).sum()

        l = (1e-5 - torch.minimum(self.h_cache,self.ntensor/self.ntensor * 1e-5))/1e-5

        fkl = kl2 
        #fkl = l*kl2 + (1-l)*kl

        #fkl = (l)*kl2 + (1-l)*kl
        #fkl = kl

        #fkl = self.ntensor/self.ntensor * 0.0

        return r_samples, fkl
