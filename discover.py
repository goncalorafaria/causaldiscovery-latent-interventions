import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import wandb
import pickle
import argparse
import random

from pycausal.distributions import norm
from causaldiscover.utils.data_utils import batchify, build_dataset, load_dcdi_data
from causaldiscover.utils.experiment_utils import run_discover, run_discover_once

"""
This code implements the sweep discovery experiment:
"""

hyperparameter_defaults = dict(
    lr=2.0,
    batch_size=248,
    epochs=50,
    ro_percentage=0.55,
    mollify_percentage=0.053,
    mixer_percentage=1.0,
    alpha=0.35,
    hdim=12,
    k=8,
    prior=0.4706,
    graph_prior=0.4996,
    beta1=0.9,
    beta2=0.99,
    wd=1e-6,
    lh=0.5,
    mlpdim=16,
    mlplayers=2)

def main():

    wandb_=True

    cluster_metrics = []
    solutions =[]

    #torch.autograd.set_detect_anomaly(mode=True)
    #torch.autograd.set_detect_anomaly(mode=True)
    if wandb_:
        wandb.init(
            project="latent-causal-discovery",
            config=hyperparameter_defaults
        )

        config = wandb.config
    else:
        config=hyperparameter_defaults

    print(config)

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help='device', default="cpu")
    parser.add_argument('--n', type=int, help='Number of nodes.', default=2)
    parser.add_argument('--e', type=int, help='edge probability.', default=4)
    parser.add_argument('--statistical_model', help='sampling model.', default="linear_normal")
    parser.add_argument('--vae_model', help='vae model.', default="linear")
    parser.add_argument('--method', help='method for learning the graph strucutre.', default="constraint")
    parser.add_argument('--estimator', help='method for learning the graph strucutre.', default="b-gst")
    parser.add_argument("--oracle",
            default=False,
            action="store_true",
            help="if set, will activate wandb")
    parser.add_argument("--perfect",
            default=False,
            action="store_true",
            help="if set, will activate wandb")
    parser.add_argument("--known",
            default=False,
            action="store_true",
            help="if set, will activate wandb")
    parser.add_argument("--observational",
            default=False,
            action="store_true",
            help="if set, will activate wandb")


    args= parser.parse_args()
    print(args)

    setup_args = {
        "problem_type":args.statistical_model, #["linear_normal","fourier_normal"],
        "n":args.n,
        "device":args.device,
        "p":0.666,#2*args.e/(args.n**2-args.n),
        "intervention_types":"stochastic",#["stochastic", "atomic", "imperfect"]
        "wandb":wandb_,
        "debug":True,
        "sample_size":10000,
        "modeltype":args.vae_model,
        "lr":10**(-config["lr"]),
        "temperature_percentage":1.5, # TODO: I'm preety sure this will be problenatical.   
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

    i=0

    criterion =[ ]
    for i in range(1):
        np.random.seed(seed=i)
        random.seed(1+i)

        reproduce = False

        if reproduce :
            x, y, t, adj_matrix, K, targets = load_dcdi_data(
                seed=1,
                file_path = "./simondata/data_p"+str(args.n)+"_e"+ str(args.e) +"0_n10000_linear_struct")
        else:
            x, y, t, adj_matrix, K, targets = build_dataset(
                n = setup_args["sample_size"],
                d = setup_args["n"],
                p = setup_args["p"],
                problem_type=setup_args["problem_type"],
                atomic=(setup_args["intervention_types"]=="atomic"),
                rand=False)

        data = (x, y, t)

        if args.observational:
            data = (x, np.zeros_like(y), np.zeros_like(t))
            dirichlet_truncation = 1
        else:
            dirichlet_truncation = K
            data = (x, y, t)

        device = torch.device(setup_args["device"])

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
            "prior":config["prior"],
            "graph_prior":config["graph_prior"],
            "method":args.method,
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
            "hard":False
            }

        makeup["i"]=dirichlet_truncation
        makeup["causal_model"] = adj_matrix.T,
        makeup["n"]=adj_matrix.shape[0]
        makeup["model_seed"]=47+i
        makeup["name"]="structure:fit_test"+str(makeup["n"])+"/oracle:"+str(makeup["oracle"])
        #import matplotlib.pyplot as plt ;plt.scatter(x[:,0],x[:,1]);plt.show()

        hamming_distance, _, _ = run_discover_once(makeup,
            data=data,
            device=device,
            debug=setup_args["debug"],
            use_wandb=setup_args["wandb"])


        """
        from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score

        s_score = silhouette_score(x,y)
        ch_score = calinski_harabasz_score(x,y)
        db_score = davies_bouldin_score(x,y)
        solutions.append(str(adj_matrix.tolist()))
        cluster_metrics.append(
            (s_score,ch_score,db_score)
        )

        hamming_distance= 0.0
        """
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
            wandb.log({"mean_hamming":mean})

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

    criterion = torch.stack(criterion)

    if setup_args["wandb"]:
        wandb.log({"final_mean_hamming":torch.mean(criterion).item()})
        wandb.log({"final_std_hamming":torch.std(criterion).item()})

    print("---"*40)
    print(torch.mean(criterion).item())
    print(torch.std(criterion).item())
    print("---"*40)



if __name__ == '__main__':
    main()
