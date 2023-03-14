import torch
import numpy as np
from pycox.models.utils import pad_col

class TMCLoss(object):

    def __init__(self, beta=1,alpha_ij=0.5,alpha_i=0.25,alpha_j=0.25):
        self.alpha_ij = alpha_ij
        self.alpha_i = alpha_i
        self.alpha_j = alpha_j
        self.beta = beta

    def __call__(self, results_dict, survival_time, state_label,interval_label,eps=1e-7):
       
       
        return torch.tensor(0)