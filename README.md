# robust-causal-code-public

## Implementation code for the paper: 

###  Partial Identification with Noisy Covariates: A Robust Optimization Approach
https://arxiv.org/abs/2202.10665


### Instructions:

To run the study for the robust estimators, please see backdoor_logistic_demo.ipynb for a simple demo of the backdoor adjustment method with the logistic data sampler. 

code/data.py contains more functions for other different data samplers; 

code/dro_training.py contains the main code for the distributed robust optimization program;

code/CEVAE_ks contains the main code for the CEVAE baseline comparison.




### Reference:

@article{guo2022partial,
  title={Partial Identification with Noisy Covariates: A Robust Optimization Approach},
  author={Guo, Wenshuo and Yin, Mingzhang and Wang, Yixin and Jordan, Michael I},
  journal={arXiv preprint arXiv:2202.10665},
  year={2022}
}
