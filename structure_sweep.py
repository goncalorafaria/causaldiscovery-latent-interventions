import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import wandb
import pickle
import argparse
import random

from pycausal.distributions import norm
from causaldiscover.utils.data_utils import batchify, build_dataset, load_dcdi_data, cxsplit
from causaldiscover.utils.experiment_utils import run_discover, eval_nll

"""
This code implements the sweep discovery experiment:
"""

hyperparameter_defaults = dict(
    lr=2.0,
    batch_size=8000,
    epochs=500,
    ro_percentage=0.00001,
    mollify_percentage=0.00001,
    mixer_percentage=1.0,
    alpha=0.1,
    hdim=264,
    k=264,
    prior=-1,# 44 # 46 #4706
    graph_prior=0.5,#0.496
    beta1=0.9,
    beta2=0.99,
    wd=1e-6,
    lh=0.5,
    mlpdim=64,
    mlplayers=2)

def main():

    wandb_=True

    cluster_metrics = []
    solutions =[]

    #torch.autograd.set_detect_anomaly(mode=True)
    #torch.autograd.set_detect_anomaly(mode=True)

    #print(config)

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help='torch device', default="cpu")
    parser.add_argument('--seed', type=int, help='Random seed.', default=42)
    
    # Synthetic data args. 
    parser.add_argument('--data', help="Sampling model for the underlying SCM {'lin','nonlin','flow'}.", default="lin")
    parser.add_argument('--data_intervention_type', help="Synth data intervention type. {'stochastic', 'atomic', 'imperfect'}", default="stochastic")
    parser.add_argument("--multi_node",
            default=False,
            action="store_true",
            help="Perform multi node interventions.")
    parser.add_argument('--n', type=int, help='Number of graph nodes.', default=10)
    parser.add_argument('--e', type=float, help='Expected number o connections per node.', default=1)
    parser.add_argument('--x', type=float, help='proportions of sample with labels in ssl.', default=0.1)
    parser.add_argument("--ssl",
            default=False,
            action="store_true",
            help="if set, uses self-supervised variant.")

    # model & search args.
    parser.add_argument('--model', help="Sampling model in use. {'linear','nonlinear','flow'} ", default="linear")
    parser.add_argument('--r', 
        help="log target prior probability search range.", 
        default=(-3,0,4))
    parser.add_argument('--g', 
        help="log target prior probability search range.", 
        default=(-3,1,4))

    parser.add_argument('--lr', type=float, help='negative log_10 of learning rate.', default=2.0)
    

    parser.add_argument(
        '--estimator', 
        help="Method for estimating the graph's gradients.", 
        default="b-gst")

    parser.add_argument("--oracle",
        default=False,
        action="store_true",
        help="if set, uses correspondence's label information.")

    parser.add_argument("--perfect",
        default=False,
        action="store_true",
        help="if set, assumes perfect interventions. ")

    parser.add_argument("--known",
        default=False,
        action="store_true",
        help="if set, uses intervention targets information. ")

    parser.add_argument("--observational",
        default=False,
        action="store_true",
        help="if set, uses the observational model. ")

    
    args= parser.parse_args()
    print(args)

    if wandb_:
        wandb.init(
            project="latent-causal-discovery",
            config=hyperparameter_defaults
        )

        config = wandb.config

        flags = [ k for k,v in args.__dict__.items() if str(v) == str(True)  ]

        prop =  [ k+":"+str(v) for k,v in args.__dict__.items() if k in {"e","g","n","r","lr"} ]

        runame =args.data + "/".join(flags+prop)

        wandb.run.name = runame

    else:
        config=hyperparameter_defaults

    setup_args = {
        "problem_type":args.data, #["linear_normal","fourier_normal"],
        "n":args.n,
        "device":args.device,
        "e":args.e,
        "intervention_types":args.data_intervention_type,#["stochastic", "atomic", "imperfect"]
        "wandb":wandb_,
        "debug":False,
        "sample_size":10000,
        "modeltype":args.model,
        "lr":10**(-args.lr),
        "temperature_percentage":1.5, 
        "ro_percentage":config["ro_percentage"],#0.1
        "mollify_percentage":config["mollify_percentage"],#0.3
        "mixer_percentage":config["mixer_percentage"],#0.8
        "epochs":config["epochs"]
        }
    #nonline : {0.4,0.01,0.8} 1e-3
    #flow : {0.4,0.4,0.8} 5e-3

    db = {}

    if args.observational:
        known = False
        oracle = False
    else:
        oracle = args.oracle
        known = args.known

    criterion =[ ]
    ps = []
    rs = []
    f1s = []

    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    
    x, y, t, adj_matrix, K, targets = build_dataset(
        n = setup_args["sample_size"],
        d = setup_args["n"],
        e = setup_args["e"],
        problem_type=setup_args["problem_type"],
        intervention_type=setup_args["intervention_types"],
        rand=args.multi_node,
        permutate=True,
        observational=args.observational,
        seed=args.seed,
        load=False,
        real=False)

    data = (x, y, t)

    if wandb_:
        wandb.log( { 
            "true_graph_sparsity" : np.sum(adj_matrix)
            })

    if args.observational:
        data = (x, np.zeros_like(y), np.zeros_like(t))
        dirichlet_truncation = 1
    else:
        dirichlet_truncation = K
        data = (x, y, t)

    device = torch.device(setup_args["device"])
    data, heldout = cxsplit(data)
    
    sheetsheet = []
    scores = []

    r_space = np.linspace(args.r[0], args.r[1], 2)
    g_space = np.linspace(args.g[0], args.g[1], 4)

    hparms = [ (r,g) for g in g_space for r in r_space ]

    for r, g in hparms:

        makeup = {
            "hdim":config["hdim"],
            "k":config["k"],
            "a":1.0,
            "b":1.0,
            "target_sparsity":0.0,
            "weight_sparsity":0.0,
            "lr":setup_args["lr"],
            "epochs":setup_args["epochs"],
            "batch":config["batch_size"],
            "alpha":config["alpha"],
            "oracle":oracle,
            "use_z_entropy":False,
            "modeltype":setup_args["modeltype"],
            "mixer_percentage":setup_args["mixer_percentage"],
            "mollify_percentage":setup_args["mollify_percentage"],
            "ro_percentage":setup_args["ro_percentage"],
            "temperature_percentage":setup_args["temperature_percentage"],
            "prior":1/(1 + np.exp(10 ** r)),
            "graph_prior": 1/(1 + np.exp(10 ** g)),
            "method":"constraint",
            "hard_permutation_samples":True,
            "estimator":args.estimator,
            "beta1":config["beta1"],
            "beta2":config["beta2"],
            "wd":config["wd"],
            "lh":config["lh"],
            "known":known,
            "targets":torch.tensor(targets,device=device),
            "perfect":args.perfect,
            "mlpdim":config["mlpdim"],
            "mlplayers":config["mlplayers"],
            "hard":True,
            "atomic": setup_args["intervention_types"]=="atomic",
            "device": device
            }

        makeup["i"]=dirichlet_truncation
        makeup["causal_model"] = adj_matrix.T,
        makeup["n"]=adj_matrix.shape[0]
        makeup["model_seed"]=args.seed
        makeup["name"]="structure:fit_test"+str(makeup["n"])+"/oracle:"+str(makeup["oracle"])
        makeup["ssl"]=args.ssl
        makeup["x"]=args.x
        
        hamming_distance, (f1,rec,pre), e_hamming_distance, model, G = run_discover(makeup,
            data=data,
            device=device,
            debug=setup_args["debug"],
            use_wandb=setup_args["wandb"])

        score = eval_nll(model, data = heldout , makeup=makeup, it = 20)
        print(score)

        sheetsheet.append(
            (g, hamming_distance,f1,rec,pre)
        )

        scores.append(score.item())
    
    indmin = np.argmin(
        np.array(scores)
    )
    
    g, hamming_distance, f1, rec, pre = sheetsheet[indmin]

    ps.append(pre)
    rs.append(rec)
    f1s.append(f1)

    print("---"*20)
    print("g:" +str(g))

    print("point hamming_dist:" +str(hamming_distance))
    sz = len(criterion)

    if sz == 0:
        mean = hamming_distance
    else :
        mean = (mean*sz+hamming_distance)/(sz+1)

    criterion.append(
        hamming_distance
    )

    if setup_args["wandb"]:
        wandb.log({"mean_hamming":mean,"gg":g})
        
    print("###running mean###")
    print(mean)

    """
    print(cluster_metrics)
    import json
    file = open('cluster_metrics.txt','w')
    file.write(json.dumps(cluster_metrics))
    file.close()

    file = open('solutions.txt','w')
    file.write(json.dumps(solutions))
    file.close()
    """

    ps = torch.tensor(ps)
    rs = torch.tensor(rs)
    f1s = torch.tensor(f1s)
    
    criterion = torch.stack(criterion)

    if setup_args["wandb"]:
        wandb.log({"final_mean_hamming":torch.mean(criterion).item()})
        wandb.log({"final_std_hamming":torch.std(criterion).item()})
        wandb.log({"ps":torch.mean(ps).item()})
        wandb.log({"rs":torch.mean(rs).item()})
        wandb.log({"f1s":torch.mean(f1s).item()})

    print("---"*40)
    print(torch.mean(criterion).item())
    print(torch.std(criterion).item())
    print("---"*40)


if __name__ == '__main__':
    main()
