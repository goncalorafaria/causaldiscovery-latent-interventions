import wandb
import pickle
import argparse
import random
import numpy as np

from pycausal.distributions import norm
from causaldiscover.utils.data_utils import batchify, build_dataset, load_dcdi_data, cxsplit
from causaldiscover.utils.experiment_utils import run_discover, eval_nll
from cdt.data import load_dataset
import networkx as nx
import torch

"""
This code implements the sweep discovery experiment:
"""

hyperparameter_defaults = dict(
    lr=2.5,
    batch_size=8000,
    epochs=1000,
    ro_percentage=0.00001,
    mollify_percentage=0.00001,
    mixer_percentage=1.0,
    alpha=0.1,
    hdim=264,
    k=512,
    prior=-1,# 44 # 46 #4706
    graph_prior=0.5,#0.496
    beta1=0.9,
    beta2=0.99,
    wd=1e-6,
    lh=0.5,
    mlpdim=128,
    mlplayers=2)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', 
        help='torch device.', 
        default="cpu")

    parser.add_argument('--model', 
        help="sampling model type = {'linear','nonlinear','flow'}.", 
        default="linear")

    parser.add_argument('--r', 
        help="log target prior probability search range.", 
        default=-0.5,
        type=float)

    parser.add_argument('--g', 
        help="log target prior probability search range.", 
        default=-0.5,
        type=float)

    parser.add_argument("--perfect",
        default=False,
        action="store_true",
        help="if set, uses the model with perfect interventions.")

    parser.add_argument("--oracle",
        default=False,
        action="store_true",
        help="if set, uses assigments information.")

    parser.add_argument("--known",
        default=False,
        action="store_true",
        help="if set, will use the targets data.")

    parser.add_argument("--observational",
        default=False,
        action="store_true",
        help="if set, will use observational model.")

    parser.add_argument("--seed",
        default=42,
        help="sets random seed.")

    args= parser.parse_args()
    print(args)

    return args


def main():

    wandb_ = True
    debug_ = False

    cluster_metrics = []
    solutions =[]

    args = parse_args()

    if wandb_:
        wandb.init(
            project="sachs-causal-discovery",
            config=hyperparameter_defaults
        )
        config = wandb.config
        flags = [ k for k,v in args.__dict__.items() if str(v) == str(True)  ]
        prop =  [ k+":"+str(v) for k,v in args.__dict__.items() if k in {"e","g","n","r","lr"} ]
        runame = "sachs" + "/".join(flags+prop) + "/" + args.model
        wandb.run.name = runame
    else:
        config=hyperparameter_defaults

    db = {}

    if args.observational:
        known = False
        oracle = False
    else:
        oracle = args.oracle
        known = args.known

    s_data, s_graph = load_dataset('sachs')
    
    adj_matrix = nx.to_numpy_array(s_graph)
    x = s_data.values
    y = np.zeros((x.shape[0],))
    t = np.zeros_like(x)
    K = 16
    targets = np.zeros((K,x.shape[1]))

    data = (x, y, t)

    if wandb_:
        wandb.log( { 
            "true_graph_sparsity" : np.sum(adj_matrix)
            })

    device = torch.device(args.device)

    data, heldout = cxsplit(data)
    
    sheetsheet = []
    scores = []

    makeup = {
            "hdim":config["hdim"],
            "k":config["k"],
            "a":1.0,
            "b":1.0,
            "target_sparsity":0.0,
            "weight_sparsity":0.0,
            "lr":10**(-config["lr"]),
            "epochs":config["epochs"],
            "batch":config["batch_size"],
            "alpha":config["alpha"],
            "oracle":oracle,
            "use_z_entropy":False,
            "modeltype":args.model,
            "mixer_percentage":config["mixer_percentage"],
            "mollify_percentage":config["mollify_percentage"],
            "ro_percentage":config["ro_percentage"],
            "temperature_percentage":1.5,
            "method":"constraint",
            "hard_permutation_samples":True,
            "estimator":"b-gst",
            "beta1":config["beta1"],
            "beta2":config["beta2"],
            "wd":config["wd"],
            "lh":config["lh"],
            "known":known,
            "targets":torch.tensor(targets,device=device),
            "perfect":args.perfect,
            "mlpdim":config["mlpdim"],
            "mlplayers":config["mlplayers"],
            "hard": False,
            "atomic": False,
            "device":device,
            "i":K,
            "causal_model":adj_matrix.T,
            "n":adj_matrix.shape[0],
            "model_seed":args.seed,
            "name":f"structure:fit_test{adj_matrix.shape[0]}/oracle:{oracle}",
            "ssl":False,
            "x":0.0
            }

    #r_space = np.linspace(args.r[0], args.r[1], args.r[2])
    #g_space = np.linspace(args.g[0], args.g[1], args.g[2])

    #hparms = [ (r,g) for g in g_space for r in r_space ]
    hparms = [ (args.r, args.g) ]
    for r, g in hparms:
        
        print(f"r:{r} // g:{g} | ")

        makeup["graph_prior"] = torch.tensor(
            1/(1 + np.exp(10 ** g)))
        makeup["prior"] = torch.tensor(
            1/(1 + np.exp(10 ** r)))
        
        try :
            hamming_distance, (f1,rec,pre), e_hamming_distance, model, G = run_discover(makeup,
                data=data,
                device=device,
                debug=debug_,
                use_wandb=wandb_)

            score = eval_nll(model, data = heldout , makeup=makeup, it = 20)

            sheetsheet.append(
                (g, hamming_distance, f1, rec, pre, G)
            )

            scores.append(score.item())


        except Exception as e:
            print(e)
            continue

        
    
    indmin = np.argmin(
        np.array(scores)
    )
    
    g, hamming_distance, f1, rec, pre, G = sheetsheet[indmin]

    G_true = makeup["causal_model"]
    G = G.cpu().numpy()


    GisTrue = ( G - 1 )**2 < 1e-6
    GisFalse = ( G  )**2 < 1e-6
    

    are_equal = ( G - G_true )**2 < 1e-6

    tp = np.logical_and(
        GisTrue,
        are_equal
    ).sum()

    fn = np.logical_and(
        GisFalse,
        G != G_true
    ).sum()

    fp = np.logical_and(
        GisTrue,
        G != G_true
    ).sum()

    rev = np.logical_and(
        (G.T-1)**2 < 1e-6,
        (G.T-G_true)**2 < 1e-6
    ).sum()

    # precision =  ( predicted & correct ) / predicted
    # rec = ( predicted & correct ) / correct
    #
    print("---"*20)
    print("g:" +str(g))

    print("point hamming_dist:" +str(hamming_distance))
    criterion = hamming_distance

    if wandb_:
        wandb.log({
            "#hamming":hamming_distance,
            "#gg":g,
            "#f1":f1,
            "#rec":rec,
            "#pre":pre,
            "#tp":tp,
            "#fn":fn,
            "#fp":fp,
            "#rev":rev,
            "#nll":score})
        
    print("###running mean###")
    print(hamming_distance)


if __name__ == '__main__':
    main()
