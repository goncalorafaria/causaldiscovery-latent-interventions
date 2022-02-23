import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import wandb
import torch_optimizer as optim


from causaldiscover.utils.model_utils import init_ro_scheduler,onehot, graph_precision, graph_recall, graph_f1
from causaldiscover.utils.data_utils import batchify, build_dataset

from causaldiscover.models import Model
from causaldiscover.structure.structure_wrapper import ConstraintMethod


class Logger:
    def __init__(self):
        self._log = []

    def log(self,arrs):
        self._log.append(arrs)

tmp = Logger()
local_log = False


def run_discover(makeup, device, data, use_wandb=False, debug=False):
        
    torch.manual_seed(makeup["model_seed"])
    #scaler = torch.cuda.amp.GradScaler()
    #true_params = { k.name:v for k,v in (makeup["samplers"][0])(1).items()}

    df, y, targets = data

    cmodel = torch.tensor(
        makeup["causal_model"]
        , device=device
    ).view(makeup["n"],makeup["n"]).float()

    print("...-----"*10)
    print(cmodel)

    model = Model(makeup)

   
    if makeup["method"] == "constraint":
        model = ConstraintMethod(
            scm = model,
            graph_prior= makeup["graph_prior"],
            estimator = makeup["estimator"]
        )

    else :
        raise ValueError(
            "The metioned dag enforcement method: " \
                + makeup["method"] +\
                 "is not implemented.")

    #import pdb; pdb.set_trace()

    model = model.to(device=device)

    #scripted_module = torch.jit.script(model)
    scripted_module = model
    # makeup["temperature_percentage"]
    # makeup["ro_percentage"]
    # makeup["mollify_percentage"]
    # makeup["mixer_percentage"]
    def annealing_f(epochs):
        annealing = lambda epoch : ( \
            init_ro_scheduler(
                device=device,
                minr=0.5,
                perc=makeup["temperature_percentage"],
                epochs=epochs,
                inverse=True)(epoch) ,
            init_ro_scheduler(
                device=device,
                minr=0.0,
                perc=makeup["ro_percentage"],#0.6
                epochs=epochs)(epoch),
            init_ro_scheduler(
                device=device,
                minr=0.0,
                perc=makeup["mollify_percentage"],#0.6
                epochs=epochs,
                inverse=True)(epoch),
            init_ro_scheduler(
                device=device,
                minr=0.0,
                perc=makeup["mixer_percentage"],#0.6
                epochs=epochs)(epoch)
        )
        return annealing

    missing = torch.bernoulli(
            torch.ones(y.shape)*makeup["x"])

    def solve_inner_problem(model, optimizer,scheduler, epochs, delta_constraint, decrease_threshold=1e-5, tolerance = 10, first = True, eps = 0):
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.95, patience=10, threshold=0.0001, threshold_mode='rel', 
            cooldown=0, min_lr=1e-4, eps=1e-08, verbose=False)
        
        past_loss = None
        tol = 0
        
        for epoch in range(epochs):
            
            temperature = torch.tensor(1.0,device=device)

            if first :
                temperature, ro, mollify, mixer = annealing_f(epochs)(epoch)
            else:
                temperature = torch.tensor(0.5,device=device)
                ro, mollify, mixer = torch.tensor(1.0,device=device), torch.tensor(0.0,device=device), torch.tensor(1.0,device=device)
                #ro = ( ro*delta_constraint + (1-delta_constraint) )* 0.6 + 0.4

            if use_wandb:
                wandb.log({"temperature":temperature, "ro":ro, "mollify":mollify, "mixer":mixer})

            loss_comulative = 0.0

            for x, yx, tx, yzx in batchify(df, batch=makeup["batch"],device=device, y=y, targets=targets, missing=missing):


                if makeup["oracle"] or makeup["ssl"] :
                    
                    if makeup["oracle"]:
                        gt = yx
                    else:
                        gt = yzx

                    loss, model_outputs, info = model(
                        x=x,
                        y=gt,
                        temperature=temperature,
                        ro=ro,
                        mollify=mollify,
                        mixer=mixer
                    )

                else :

                    loss, model_outputs, info = model(
                        x=x,
                        temperature=temperature,
                        ro=ro,
                        mollify=mollify,
                        mixer=mixer
                    )

                data = model.explain(
                    model_outputs,
                    targets=tx,
                    y=yx,
                    cmodel=cmodel,
                    debug=(debug and use_wandb),
                    )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=2.0
                )

                optimizer.step()
                scheduler.step(loss.item())
                
                optimizer.zero_grad()

                loss_comulative += loss.item()

                if use_wandb:
                    ww=2
                    wandb.log(data)
                    #wandb.log({"lr":scheduler._last_lr[0]})


                if local_log:
                    tmp.log(data)

            if past_loss is None : 
                past_loss = loss_comulative
            else :
                relative_decrease = (past_loss - loss_comulative) / past_loss
                
                tol = int( (relative_decrease*(-1) < decrease_threshold) ) * ( tol + 1 )

                #print(f"rel_decrease:{relative_decrease} ;/ tol:{tol}")
                
                past_loss = loss_comulative

            if ( tol >= tolerance ):
                print(f" stopped at the {epoch} epoch.")
                break

        
    def constraint_eval(model):

        cache = []
        for x, yx, tx, yzx in batchify(
            iterable=df,
            batch=makeup["batch"],
            device=device,
            y=y,
            targets=targets):

            if makeup["oracle"] or makeup["ssl"] :

                loss, model_outputs, info = model(
                    x=x,
                    y=yx,
                    temperature=torch.tensor(0.0000001, device=device),
                    ro=torch.tensor(1.0, device=device),
                    mollify=torch.tensor(0.0, device=device),
                    mixer=torch.tensor(1.0, device=device)
                )
            else:

                loss, model_outputs, info = model(
                    x=x,
                    temperature=torch.tensor(0.0000001, device=device),
                    ro=torch.tensor(1.0, device=device),
                    mollify=torch.tensor(0.0, device=device),
                    mixer=torch.tensor(1.0, device=device)
                )
            cache.append(
                model_outputs["delta_constraint"]
            )

        return np.stack(cache).mean(), model_outputs

    #def get_optimizer(model,epochs):
    optimizer = optim.RAdam(
            model.parameters(),
            lr= makeup["lr"],
            betas=(0.9, 0.999),
            weight_decay=1e-6,
            eps=1e-8,
            )

    #optimizer = optim.Lookahead(optimizer, k=5, alpha=0.5)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.99, patience=10, threshold=0.0001, threshold_mode='rel', 
        cooldown=0, min_lr=1e-4, eps=1e-08, verbose=False)

    delta_constraint = torch.tensor(1.0, device=device)

    anneal = True

    eps = 0

    while delta_constraint > 1e-8:

        if anneal:
            epochs = makeup["epochs"]
        else:
            epochs =  makeup["epochs"] // 10

        model.train()
      
        optimizer.param_groups[0]["lr"] = makeup["lr"]
        
        solve_inner_problem(
            model, 
            optimizer, 
            scheduler,
            epochs=epochs, 
            delta_constraint=delta_constraint, 
            first = anneal,
            eps = eps)

        eps += epochs

        anneal = False

        model.eval()

        delta_constraint, data = constraint_eval(model)

        #print(data["targets_pred"].mean(0))
        #print(model.get_probs())

        with torch.no_grad():
            model.update( 
                model.graph_prior[0,0]/model.graph_prior[0,0] *  delta_constraint)

    G = (model.get_probs()>0.5)

    if makeup["method"]=="permutation":
        P, _ = model.sortdist.rsample(batch=20,temperature=0.01)
        print("P: "+ str(P))
        print("G: "+str(G))
        #G = torch.einsum("bnx, bnm, bmy -> bxy", P, G, P).squeeze(0)

        print(model.sortdist.logscores)

    targetsprobs = data["targets_pred"].mean(0)
    print("predicted graph")
    print("--"*20)
    print(G)
    print("--"*20)
    print("probs:")
    print(model.get_probs())
    print("targets probs:")
    print(targetsprobs)

    hamming_distance = (G.float() - cmodel).abs().sum()
    e_hamming_distance = (model.get_probs() - cmodel).abs().sum()

    f1 = graph_f1( pred_G=G.float(), true_G=cmodel)
    rec = graph_recall(pred_G=G.float(), true_G=cmodel)
    pre = graph_precision(pred_G=G.float(), true_G=cmodel)

    if use_wandb:
        wandb.log({"precision":pre,"recall":rec,"f1":f1})

    if local_log:
        import pickle
        with open("save.txt", "wb") as f:
            pickle.dump(tmp,f)

    return hamming_distance, (f1,rec,pre), e_hamming_distance, model, G

def eval_nll(model, data, makeup, it = 1):
    
    df, y, targets = data
    device = makeup["device"]

    model.eval()
    #import pdb; pdb.set_trace()

    ss = []
    with torch.no_grad():
        for i in range(it):
            mse = []
            for x, yx, _, _  in batchify(
                iterable=df,
                batch=makeup["batch"],
                device=device,
                y=y,
                targets=targets):

                if makeup["oracle"] or makeup["ssl"]:

                    loss, model_outputs, info = model(
                        x=x,
                        y=yx,
                        temperature=torch.tensor(0.2, device=device),
                        ro=torch.tensor(1.0, device=device),
                        mollify=torch.tensor(0.0, device=device),
                        mixer=torch.tensor(1.0, device=device)
                    )
                else:
                    loss, model_outputs, info = model(
                        x=x,
                        temperature=torch.tensor(0.2, device=device),
                        ro=torch.tensor(1.0, device=device),
                        mollify=torch.tensor(0.0, device=device),
                        mixer=torch.tensor(1.0, device=device)
                    )

                mse.append(
                    model_outputs["true_mse"].sum()
                    )

            s = torch.stack(mse,axis=0).sum() / df.shape[0]

            ss.append(s)

    return torch.stack(ss,axis=0).sum() 

