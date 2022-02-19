import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import wandb
import torch_optimizer as optim


from causaldiscover.utils.model_utils import init_ro_scheduler,onehot, graph_precision, graph_recall, graph_f1
from causaldiscover.utils.data_utils import batchify, build_dataset

from causaldiscover.models import Model

from causaldiscover.structure.structure_wrapper import PermutationMethod, ConstraintMethod, ECNOMethod

"""
modeltypes_cache = {
    "linear_perfect": [PerfectLinearHyperModel, PerfectLinearOracleHyperModel, PerfectLinearFullOracleHyperModel],
    "linear_imperfect": [LinearHyperModel, LinearOracleHyperModel, None],
    "nonlinear_imperfect": [ NNHyperModel, NNOracleHyperModel, NNFullOracleHyperModel],
    "nonlinear_perfect": [ NNPerfectHyperModel, NNPerfectOracleHyperModel, NNFullPerfectOracleHyperModel ],
    "flow_imperfect": [ DeepFlowHyperModel, DeepFlowOracleHyperModel, NNFullPerfectOracleHyperModel] }
"""

class Logger:
    def __init__(self):
        self._log = []

    def log(self,arrs):
        self._log.append(arrs)
tmp = Logger()
local_log = False

def run(makeup, device, data, use_wandb=False, debug=False):

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

    print(makeup)

    model = Model(makeup)

    print(model)

    model = model.to(device=device)

    """
    optimizer = optim.DiffGrad(
        model.parameters(),
        lr= makeup["lr"],
        betas=(0.9, 0.999),
        weight_decay=1e-5,
        eps=1e-8,
        )
        optimizer = optim.RAdam(
            model.parameters(),
            lr= makeup["lr"],
            betas=(0.9, 0.999),
            weight_decay=1e-5,
            eps=1e-8,
            )

    """
    
    optimizer = optim.RAdam(
            model.parameters(),
            lr= makeup["lr"],
            betas=(0.9, 0.999),
            weight_decay=1e-6,
            eps=1e-8,
            )

    #optimizer = optim.Lookahead(optimizer, k=5, alpha=0.5)

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #    optimizer, 
    #    makeup["epochs"]*df.shape[0]/makeup["batch"], 1e-8)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.95, patience=10, threshold=0.0001, threshold_mode='rel', 
        cooldown=0, min_lr=0.0, eps=1e-08, verbose=False)

    scripted_module = model

    annealing = lambda epoch : ( \
        init_ro_scheduler(
            device=device,
            minr=0.2,
            perc=makeup["temperature_percentage"],
            epochs=makeup["epochs"],
            inverse=True)(epoch) ,
        init_ro_scheduler(
            device=device,
            minr=0.0,
            perc=makeup["ro_percentage"],#0.6
            epochs=makeup["epochs"])(epoch),
        init_ro_scheduler(
            device=device,
            minr=0.0,
            perc=makeup["mollify_percentage"],#0.6
            epochs=makeup["epochs"],
            inverse=True)(epoch),
        init_ro_scheduler(
            device=device,
            minr=0.0,
            perc=makeup["mixer_percentage"],#0.6
            epochs=makeup["epochs"])(epoch)
    )

    model.train()

    with torch.jit.optimized_execution(True):

        previous_loss = 0.0

        missing = torch.bernoulli(
            torch.ones(y.shape)*makeup["x"])

        for epoch in tqdm(range(makeup["epochs"])):

            temperature, ro, mollify, mixer = annealing(epoch)

            if use_wandb:
                wandb.log({"temperature":temperature, "ro":ro, "mollify":mollify, "mixer":mixer})
            
            losses =[]

            for x, yx, tx, yzx in batchify(df, batch=makeup["batch"],device=device, y=y, targets=targets, missing=missing):

                if makeup["oracle"] or makeup["ssl"] :
                    
                    if makeup["oracle"]:
                        gt = yx
                    else:
                        gt = yzx

                    loss, model_outputs, info = model(
                        z=cmodel, x=x,
                        ground_truth_assigments=gt,
                        temperature=temperature,
                        ro=ro,
                        mollify=mollify.item(),
                        mixer=mixer
                    )
                else:

                    loss, model_outputs, info = model(
                        z=cmodel, x=x,
                        temperature=temperature,
                        ro=ro,
                        mollify=mollify.item(),
                        mixer=mixer
                    )

                data = model.explain(
                    model_outputs,
                    targets=tx,
                    y=yx,
                    debug=(debug and use_wandb)
                )

                loss.backward()
                #scaler.scale(loss).backward()


                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=2.0
                )

                optimizer.step()

                scheduler.step(loss.item())

                optimizer.zero_grad()

                if use_wandb:
                    ww=2
                    wandb.log({"lr":scheduler._last_lr[0]})
                    #print(scheduler._last_lr[0])
                    wandb.log(data)

                losses.append(loss.item())

            current_loss = sum(losses)

            if np.abs( (current_loss - previous_loss) ) < 1e-4:
                print("---"*20)
                print("early stopped at "+str(epoch)+":")
                print("---"*20)
                #break
            else:
                print(  (current_loss - previous_loss) )
                previous_loss = current_loss

    print("targets_pred :")
    print(model_outputs["targets_pred"].mean(0))

    print("targets_true:")
    print(makeup['targets'])
            
    model.eval()
    #import pdb; pdb.set_trace()
    data_cache = {}
    with torch.no_grad():
        for x, yx, tx, yzx in batchify(
            iterable=df,
            batch=makeup["batch"],
            device=device,
            y=y,
            targets=targets):

            if makeup["oracle"] or makeup["ssl"]:
                
                if makeup["oracle"]:
                    gt = yx
                else:
                    gt = yzx

                loss, model_outputs, info = model(
                    z=cmodel,
                    x=x,
                    ground_truth_assigments=gt,
                    temperature=torch.tensor(0.0000001, device=device),
                    ro=torch.tensor(1.0, device=device)
                )
            else:

                loss, model_outputs, info = model(
                    z=cmodel,
                    x=x,
                    temperature=torch.tensor(0.0000001, device=device),
                    ro=torch.tensor(1.0, device=device)
                )

            argz = torch.argmax(model_outputs['z_probs'],dim=1)


            """
            import matplotlib.pyplot as plt
            plt.scatter(
                x=x[:,0].cpu().numpy(),
                c=(argz.cpu().numpy()), 
                y=x[:,1].cpu().numpy()
            )
            plt.show()
            """

            data = model.explain(
                model_outputs,
                targets=tx,
                y=yx,
                debug=debug
            )

            for k,v in data.items() :
                if k not in data_cache :
                    data_cache[k] = [v]
                else:
                    data_cache[k].append(v)

        data = { k :( sum(v)/len(v) ) for k,v in data_cache.items()}

    return  data, makeup, None

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

    if makeup["method"] == "permutation":
        model = PermutationMethod(
            scm = model,
            graph_prior= makeup["graph_prior"],
            hard = makeup["hard_permutation_samples"],
            estimator = makeup["estimator"]
        )
    elif makeup["method"] == "constraint":
        model = ConstraintMethod(
            scm = model,
            graph_prior= makeup["graph_prior"],
            estimator = makeup["estimator"]
        )
    elif makeup["method"] == "ecno":
        model = ECNOMethod(
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

    def solve_inner_problem(model, optimizer,scheduler, epochs, delta_constraint,first = True, eps = 0):

        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.95, patience=10, threshold=0.0001, threshold_mode='rel', 
            cooldown=0, min_lr=1e-4, eps=1e-08, verbose=False)
        

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

                if use_wandb:
                    ww=2
                    wandb.log(data)
                    #wandb.log({"lr":scheduler._last_lr[0]})


                if local_log:
                    tmp.log(data)


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



def run_discover_once(makeup, device, data, use_wandb=False, debug=False):

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

    if makeup["method"] == "permutation":
        model = PermutationMethod(
            scm = model,
            graph_prior= makeup["graph_prior"],
            hard = makeup["hard_permutation_samples"],
            estimator = makeup["estimator"]
        )
    elif makeup["method"] == "constraint":
        model = ConstraintMethod(
            scm = model,
            graph_prior= makeup["graph_prior"],
            estimator = makeup["estimator"]
        )
    elif makeup["method"] == "ecno":
        model = ECNOMethod(
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

    annealing = lambda epoch : ( \
        init_ro_scheduler(
            device=device,
            minr=0.01,
            perc=makeup["temperature_percentage"],
            epochs=makeup["epochs"],
            inverse=True)(epoch) ,
        init_ro_scheduler(
            device=device,
            minr=0.0,
            perc=makeup["ro_percentage"],#0.6
            epochs=makeup["epochs"])(epoch),
        init_ro_scheduler(
            device=device,
            minr=0.0,
            perc=makeup["mollify_percentage"],#0.6
            epochs=makeup["epochs"],
            inverse=True)(epoch),
        init_ro_scheduler(
            device=device,
            minr=0.0,
            perc=makeup["mixer_percentage"],#0.6
            epochs=makeup["epochs"])(epoch)
    )

    #def get_optimizer(model,epochs):
    optimizer = optim.RAdam(
            model.parameters(),
            lr= makeup["lr"],
            betas=(0.9, 0.999),
            weight_decay=1e-6,
            eps=1e-8,
            )

    optimizer = optim.Lookahead(optimizer, k=5, alpha=0.5)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        makeup["epochs"]*df.shape[0]/makeup["batch"], 1e-6)
    
    model.train()
    
    with torch.jit.optimized_execution(True):

        for epoch in tqdm(range(makeup["epochs"])):

            temperature, ro, mollify, mixer = annealing(epoch)

            if use_wandb:
                wandb.log({"temperature":temperature, "ro":ro, "mollify":mollify, "mixer":mixer})
            
            for x, yx, tx in batchify(df, batch=makeup["batch"],device=device, y=y, targets=targets):

                if makeup["oracle"]:

                    loss, model_outputs, info = model(
                        x=x,
                        y=yx,
                        temperature=temperature,
                        ro=ro,
                        mollify=mollify,
                        mixer=mixer
                    )
                else:

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
                    debug=(debug and use_wandb)
                )

                loss.backward()
                #scaler.scale(loss).backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=2.0
                )

                optimizer.step()
                scheduler.step()

                optimizer.zero_grad()

                if use_wandb:
                    ww=2
                    #wandb.log({"lr":scheduler._last_lr[0]})
                    #print(scheduler._last_lr[0])
                    wandb.log(data)

    model.eval()
    #import pdb; pdb.set_trace()
    with torch.no_grad():
        for x, yx, tx in batchify(
            iterable=df,
            batch=makeup["batch"],
            device=device,
            y=y,
            targets=targets):

            if makeup["oracle"]:

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

            data = model.explain(
                model_outputs,
                targets=tx,
                y=yx,
                cmodel=cmodel,
                debug=(debug and use_wandb)
            )

    print(data)

    G = model.get_probs()>0.5

    if makeup["method"]=="permutation":
        P, _ = model.sortdist.rsample(batch=20,temperature=0.01)
        print("P: "+ str(P))
        print("G: "+str(G))
        #G = torch.einsum("bnx, bnm, bmy -> bxy", P, G, P).squeeze(0)

        print(model.sortdist.logscores)

    print("predicted graph")
    print("--"*20)
    print(G)
    print("--"*20)
    print("probs:")
    print(model.get_probs())

    hamming_distance = (G.float() - cmodel).abs().sum()

    if use_wandb:
        wandb.log({"hamming_distance_point":hamming_distance})
    
    return hamming_distance, makeup, None

def maximum_a_posteriori(makeup, device, data, use_wandb=False, debug=False):

    torch.manual_seed(makeup["model_seed"])
    #scaler = torch.cuda.amp.GradScaler()
    #true_params = { k.name:v for k,v in (makeup["samplers"][0])(1).items()}

    df, y, targets = data

    cmodel = makeup["causal_model"]
    cmodel = cmodel.to(device).float()

    #print("...-----"*10)
    #print(cmodel)

    model = Model(makeup)

    #import pdb; pdb.set_trace()

    model = model.to(device=device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr= makeup["lr"],
        betas=(makeup["beta1"], makeup["beta2"]),
        eps=1e-08,
        weight_decay=makeup["wd"],
        amsgrad=True
    )
    optimizer = optim.Lookahead(optimizer, k=5, alpha=makeup["lh"])
    #optimizer = optim.Lookahead(optimizer, k=5, alpha=0.5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=df.shape[0]//makeup["batch"]*makeup["epochs"]\
    )

    #scripted_module = torch.jit.script(model)
    scripted_module = model

    annealing = lambda epoch : ( \
        init_ro_scheduler(
            device=device,
            minr=0.01,
            perc=makeup["temperature_percentage"],
            epochs=makeup["epochs"],
            inverse=True)(epoch) ,
        init_ro_scheduler(
            device=device,
            minr=0.0,
            perc=makeup["ro_percentage"],#0.6
            epochs=makeup["epochs"])(epoch),
        init_ro_scheduler(
            device=device,
            minr=0.0,
            perc=makeup["mollify_percentage"],#0.6
            epochs=makeup["epochs"],
            inverse=True)(epoch),
        init_ro_scheduler(
            device=device,
            minr=0.0,
            perc=makeup["mixer_percentage"],#0.6
            epochs=makeup["epochs"])(epoch)
    )

    model.train()


    for epoch in tqdm(range(makeup["epochs"])):

        temperature, ro, mollify, mixer = annealing(epoch)

        if use_wandb:
            wandb.log({"temperature":temperature, "ro":ro, "mollify":mollify, "mixer":mixer})

        for x, yx, tx in batchify(df, batch=makeup["batch"],device=device, y=y, targets=targets):


            if makeup["oracle"] :
                loss, model_outputs, info = model(
                    z=cmodel,
                    x=x,
                    ground_truth_assigments=yx,
                    temperature=temperature,
                    ro=ro,
                    mollify=mollify.item(),
                    mixer=mixer
                )

            else :

                loss, model_outputs, info = model(
                    z=cmodel,
                    x=x,
                    temperature=temperature,
                    ro=ro,
                    mollify=mollify.item(),
                    mixer=mixer
                )

            data = model.explain(
                model_outputs,
                targets=tx,
                y=yx,
                debug=(debug and use_wandb),
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=2.0
            )

            optimizer.step()
            scheduler.step()


            optimizer.zero_grad()

            if use_wandb:
                ww=2
                wandb.log({"lr":scheduler._last_lr[0]})
                #print(scheduler._last_lr[0])
                wandb.log(data)

    model.eval()
    #import pdb; pdb.set_trace()

    data_cache = {}

    with torch.no_grad():
        for x, yx, tx in batchify(
            iterable=df,
            batch=makeup["batch"],
            device=device,
            y=y,
            targets=targets):

            if makeup["oracle"] :

                loss, model_outputs, info = model(
                    z=cmodel,
                    x=x,
                    ground_truth_assigments=yx,
                    temperature=torch.tensor(0.0000001, device=device),
                    ro=torch.tensor(1.0, device=device),
                    mollify=torch.tensor(0.0, device=device),
                    mixer=torch.tensor(1.0, device=device)
                )
            else:

                loss, model_outputs, info = model(
                    z=cmodel,
                    x=x,
                    temperature=torch.tensor(0.0000001, device=device),
                    ro=torch.tensor(1.0, device=device),
                    mollify=torch.tensor(0.0, device=device),
                    mixer=torch.tensor(1.0, device=device)
                )

            data = model.explain(
                model_outputs,
                targets=tx,
                y=yx,
                debug=debug)

            for k,v in data :
                if k not in data_cache :
                    data_cache[k] = [v]
                else:
                    data_cache[k].append(v)

        print(data_cache)

    return  data, makeup, None
