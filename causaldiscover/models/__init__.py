
from causaldiscover.models.linear import meta as metalin
from causaldiscover.models.nonlinear import meta as metanon
from causaldiscover.models.flow import meta as metaflow

def general_model_picker(args, meta):
    meta_order = ["perfect", "oracle", "known", "ssl"]
    
    try :
        value = meta
        for i in range(len(meta_order)):
            k = meta_order[i]
            value = value[args[k]]
    except Exception as e:
        print(e)
        raise ValueError("Model combination not found or " + str(meta_order)+ "not in args.")

    return value

Linear = lambda x : general_model_picker(x, metalin)
NonLinear = lambda x : general_model_picker(x, metanon)
Flow = lambda x : general_model_picker(x, metaflow)

def density_picker(model_name, problem):
    pickers_dict = {"linear":Linear,"nonlinear":NonLinear,"flow":Flow}
    model_cask = pickers_dict[model_name]
    return model_cask(problem)


def Model(args):
    model = density_picker(args["modeltype"], args)
    print(model)
    
    common = {
        "n":args["n"],
        "k":args["k"],
        "hdim":args["hdim"],
        "target_sparsity":args["target_sparsity"],
        "weight_sparsity":args["weight_sparsity"],
        "alpha":args["alpha"],
        "nintv":args["i"],
        "prior":args["prior"],
        "use_z_entropy":args["use_z_entropy"],
        "mlpdim":args["mlpdim"],
        "mlplayers":args["mlplayers"],
        "hard":args["hard"]
    }

    if args["known"]:
        common["ground_truth_targets"]=args["targets"]

    if args["perfect"]:
        common["atomic"] = args["atomic"]
        
    return model(**common)
