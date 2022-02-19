# Differentiable Causal Discovery Under Latent interventions.

Gonçalo Rui Alves Faria, Andre Martins, Mario A. T. Figueiredo

Paper: https://openreview.net/forum?id=hDrn2Dmb7_I

Abstract: 

Recent work has shown promising results in causal discovery by leveraging interventional data with gradient-based methods, even when the intervened variables are unknown. However, previous work assumes that the correspondence between samples and interventions is known, which is often unrealistic. We envision a scenario with an extensive dataset sampled from multiple intervention distributions and one observation distribution, but where we do not know which distribution originated each sample and how the intervention affected the system, \textit{i.e.}, interventions are entirely latent. We propose a method based on neural networks and variational inference that addresses this scenario by framing it as learning a shared causal graph among a infinite mixture (under a Dirichlet process prior) of intervention structural causal models . Experiments with synthetic and real data show that our approach and its semi-supervised variant are able to discover causal relations in this challenging scenario. 

----
## Experiments:

The training scripts are : 

* sachs_experiment.py

Experiments on the flow cytometry data set (Sachs et al.2005). 

* structure_sweep.py

Experiments on synthetic data. The data is generated using PyCausal ( https://github.com/goncalorafaria/PyCausal ). 

-----

## Citation: 

````
@inproceedings{
    faria2022differentiable,
    title={Differentiable Causal Discovery Under Latent Interventions},
    author={Gon{\c{c}}alo Rui Alves Faria and Andre Martins and Mario A. T. Figueiredo},
    booktitle={First Conference on Causal Learning and Reasoning},
    year={2022},
    url={https://openreview.net/forum?id=hDrn2Dmb7_I}
}
````
