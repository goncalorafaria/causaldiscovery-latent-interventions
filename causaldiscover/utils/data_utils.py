import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import csv
import random
from pycausal.scm import relu

from pycausal.problems import NormalMediator, NormalFork, \
    RandomLinearNormal, RandomFourierNormal, \
        sample_perfect_intervention_and_targets, \
            isource, sample_imperfect_intervention_and_targets, \
            sample_imperfect_intervention
import numpy as np
import networkx as nx

def build_dataset(
    n=10000, 
    d = 10, 
    problem_type="linear", 
    rand=True, 
    permutate=True,
    e=2,
    intervention_type="stochastic",
    observational=False,
    load = True,
    seed = 1,
    real=False
    ):# expected number of edges


    if real: 
        
        # df, ys, ts , adj_matrix, K, targets
        from cdt.data import load_dataset
        s_data, s_graph = load_dataset('sachs')
        
        adj_matrix = nx.to_numpy_array(s_graph)
        df = s_data.values
        ys = np.zeros((df.shape[0],))
        ts = np.zeros_like(df)
        K = 10
        targets = np.zeros((K,df.shape[1]))
        
        #import pdb; pdb.set_trace()

    else:
        if load :
            name = "_".join( map(str,[problem_type,
                d,
                int(e),
                intervention_type,
                n]) )
            fname = problem_type+"/"+name+"#"+str(seed)

            #import pdb; pdb.set_trace()

            if os.path.exists(fname+".npz") :
                print("loading :{}".format(fname))
                with np.load(fname+".npz") as data:
                    K = data['K']
                    x = data['x']
                    y = data['y']
                    t = data['t']
                    adj_matrix = data['adj_matrix']
                    targets = data['targets']

                    return x, y, t, adj_matrix, K, targets

        atomic = (intervention_type == "atomic")
        imperfect = (intervention_type == "imperfect")

        p = 2 * d * e / ((d-1)*d) # expected number of edges / fully connected. 

        print(p)

        if problem_type=="lin":
            scm , vars, adj_matrix = RandomLinearNormal(n=d, p=p)
        elif problem_type=="nonlin":
            scm , vars, adj_matrix = RandomFourierNormal(n=d, p=p, transform=relu )
        elif problem_type=="flow":
            scm , vars, adj_matrix = RandomFourierNormal(n=d, p=p,transform=relu,nonnormal=True)
        else:
            raise ValueError("unknown problem type: " + problem_type )
        

        """
        dts = scm(4000)
        d0 = np.array([ dts[v] for v in vars])

        import matplotlib.pyplot as plt; plt.scatter(d0[0],d0[1]);plt.show()
        #import matplotlib.pyplot as plt; plt.scatter(d0[1],d0[2]);plt.show()
        #import matplotlib.pyplot as plt; plt.scatter(d0[0],d0[2]);plt.show()

        import pdb; pdb.set_trace()
        """
        print(adj_matrix)
        #
        
        targets = [ np.zeros(d,dtype=np.int) ]
        isamplers = []

        if rand :

            K = 2*d+1

            for i in range(d*2):
                
                if imperfect :
                    conditioning, ohot = sample_imperfect_intervention_and_targets(
                        n=min(d-1,random.choice(np.arange(1,((d+2)//3)+1, dtype=np.int))),
                        vars=vars)
                else:
                    conditioning, ohot = sample_perfect_intervention_and_targets(
                        adj_matrix,
                        n=min(d-1,random.choice(np.arange(1,((d+2)//3)+1, dtype=np.int))),
                        vars=vars,
                        atomic = atomic)

                targets.append(
                    ohot
                )
                isamplers.append(
                    conditioning
                )

            targets = np.array(targets)

        else:
            K = d+1

            if imperfect :
                targets = np.concatenate([ np.zeros((1,d),dtype=np.int), np.eye(d,dtype=np.int)], axis=0)
                isamplers = [ sample_imperfect_intervention(vars[i]) for i in range(d)]
            else:
                targets = np.concatenate([ np.zeros((1,d),dtype=np.int), np.eye(d,dtype=np.int)], axis=0)
                isamplers = [ { vars[i] : isource(atomic) } for i in range(d)]


        if observational : 
            samplers = [ ~scm ]
        else:
            samplers = [ ~scm ] + [
                ~(scm&isam) for isam in isamplers ]

            n = n // len(samplers)

        dts = [ s(n) for s in samplers ]

        """
        d0 = np.array([ dts[0][v] for v in vars])
        d1 = np.array([ dts[1][v] for v in vars])
        d2 = np.array([ dts[2][v] for v in vars])
        d3 = np.array([ dts[3][v] for v in vars])
        
        import pdb; pdb.set_trace()
        """
        
        obs = lambda s: tuple([ s[v] for v in vars ])

        v_stacks = zip(*[ obs(s) for s in dts ])

        """
        a, b, c =  v_stacks

        import matplotlib.pyplot as plt; 
        
        plt.scatter(b[0],c[0])
        plt.scatter(b[1],c[1])
        plt.scatter(b[2],c[2])
        
        plt.show()
        """

        df = np.concatenate(
            [ np.concatenate(t,axis=0) for t in v_stacks ],
            axis=1
        )

        df = ( df - df.mean(0) ) / df.std(0)

        idx = np.arange(df.shape[0])
        np.random.shuffle(idx)
        inds = [ (np.ones([n])*i).astype(np.int) for i in range(len(samplers))]

        y = np.concatenate(inds,axis=0)

        ts = [ targets[yi] for yi in y ]

        df = df[idx]
        ts = np.stack(ts)[idx]
        ys = y[idx]

        if permutate :
            arg = np.argsort(
                np.random.random(size=d)
            )
            perm = np.eye(d)[arg]

            print("perm: " + str(perm))
            adj_matrix = perm.T @ adj_matrix @ perm
            print("shifted graph"+str(adj_matrix))
            df = df @ perm
            ts = ts @ perm
            targets = targets @ perm
        
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        
        ax.scatter(df[:,0][ys==0],df[:,1][ys==0],df[:,2][ys==0], label="obs"); 
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        print(targets)
        ax.scatter(df[:,0][ys==0],df[:,1][ys==0],df[:,2][ys==0], label="obs"); 
        ax.scatter(df[:,0][ys==1],df[:,1][ys==1],df[:,2][ys==1], label="r1"); 
        ax.scatter(df[:,0][ys==2],df[:,1][ys==2],df[:,2][ys==2], label="r2"); 
        ax.scatter(df[:,0][ys==3],df[:,1][ys==3],df[:,2][ys==3], label="r3"); 
        
        plt.legend();
        plt.show()

        
        import seaborn as sns

        x, y = df[:,0],df[:,1]
        #f, ax = plt.subplots(figsize=(6, 6))
        sns.jointplot(x=x, y=y, s=5, hue=ys, color=".15", palette="tab10")
        #sns.histplot(x=x, y=y, bins=200, pthresh=.1, cmap="mako")
        #sns.kdeplot(x=x, y=y, levels=5, color="w", linewidths=1)
        #plt.show()
        plt.savefig("yxcatter.png",bbox_inches='tight', transparent="True", pad_inches=0)
        import pdb; pdb.set_trace()
        
        """

        print(">>>>>>>>>> K:" +str(K))
    
    return df, ys, ts , adj_matrix, K, targets

def cxsplit(data,frac=0.8):
    df, ys, ts = data 

    trainsize = int(df.shape[0]*frac)

    idx = np.arange(df.shape[0])
    np.random.shuffle(idx)

    train_idx = idx[:trainsize]
    test_idx = idx[trainsize:]

    train_df = df[train_idx]
    train_ts = ts[train_idx]
    train_ys = ys[train_idx]

    test_df = df[test_idx]
    test_ts = ts[test_idx]
    test_ys = ys[test_idx]

    return (train_df,train_ys,train_ts), (test_df,test_ys,test_ts)

def randomize_data(its, n):
    idx = np.arange(n)
    np.random.shuffle(idx)
    
    return [ i[idx] for i in its ] 

def batchify(iterable, device, batch=1,y=None, targets=None, randomize=True, missing=None):
    l = len(iterable)
    
    if randomize:
        if y is not None and targets is not None and missing  is not None:
            iterable, y, targets, missing = randomize_data([iterable, y, targets,missing], l)
        elif y is not None and targets is not None  :
            iterable, y, targets = randomize_data([iterable, y, targets ], l)
        elif y is None:
            iterable, targets = randomize_data([iterable, targets], l)
        elif targets is None:
            iterable, y = randomize_data([iterable, y], l)

    for ndx in range(0, l, batch):
        if y is not None:
            x = torch.tensor(
                iterable[ndx:min(ndx + batch, l)],dtype=torch.float32,device=device)

            yx = torch.tensor(
                y[ndx:min(ndx + batch, l)],dtype=torch.long,device=device)


            if missing is not None :
                mx = missing[ndx:min(ndx + batch, l)].to(device)
                yzx = mx * yx  + (1-mx) * -1

            else: 
                yzx = yx

            tx = torch.tensor(
                targets[ndx:min(ndx + batch, l)],dtype=torch.long,device=device)

            yield (x,yx,tx,yzx)
        else:
            yield torch.tensor(
                iterable[ndx:min(ndx + batch, l)],dtype=torch.float32,device=device)


def load_dcdi_data(
    d,
    e,
    problem_type,
    intervention_type,
    seed=1,
    file_path = "simondata/",
    observational=False):

    if problem_type == "nonlin":
        pt = "nnadd"
    elif problem_type == "lin":
        pt = "linear"
    elif problem_type == "flow":
        pt = "nn"
    else:
        raise ValueError("problem_type must be 'nonlin', 'lin', or 'flow'")

    if intervention_type == "imperfect":
        file_path = file_path+"imperfect/data_p"+str(d)+"_e"+str(e)+"0_n10000_"+pt+"_brutal_param"
    else:
        file_path = file_path+"perfect/data_p"+str(d)+"_e"+str(e)+"0_n10000_"+pt+"_struct"
    
    print(file_path)

    #print(path)
    adjacency = np.load(os.path.join(file_path, f"DAG{seed}.npy"))
    #data = np.load(os.path.join(file_path,f"data{seed}.npy")) # 10000, 10
    print(adjacency)

    data = np.load(os.path.join(file_path,f"data_interv{seed}.npy"))
    regimes = np.genfromtxt(os.path.join(file_path, f"regime{seed}.csv"), delimiter=",")
    regimes = regimes.astype(int)

    if observational :
        data = data[ regimes == 0 ]

    D = adjacency.shape[0]
    N = regimes.shape[0]
    K = np.unique(regimes).shape[0]

    interv_path = os.path.join(file_path, f"intervention{seed}.csv")
    r = np.zeros((N,D),dtype=np.int)

    masks = []
    with open(interv_path, 'r') as f:
        interventions_csv = csv.reader(f)
        for row in interventions_csv:
            mask = [int(x) for x in row]

            masks.append(mask)
    
    for i in range(len(masks)):
        for j in masks[i]:
            r[i,j] = 1

    #idx = np.arange(N)
    #np.random.shuffle(idx)

    targets = np.zeros( (K,D), dtype=np.int )

    for i in range(len(regimes)):
        ri = regimes[i]
        targets[ri] = r[i]

    return data, regimes, r, adjacency, K, targets


#load_dcdi_data(file_path = "./simondata/data_p10_e10_n10000_linear_struct")
#def from_simons_format():
