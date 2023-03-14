import pickle,os,glob
from lifelines.utils import concordance_index
from sksurv.metrics import concordance_index_censored
import numpy as np

data_list = r'/data/project/TCGA/code/Patch-GCN/results/5foldcv/PatchGCN_4_spatialsm_nll_surv_a0.0_5foldcv_gc32_sig/tcga_brca_PatchGCN_4_spatialsm_nll_surv_a0.0_5foldcv_gc32_sig_s1/*.pkl'
data_list = glob.glob(data_list)
for f in data_list:
    res = open(f,'rb')
    f_data = pickle.load(res)
    keys = f_data.keys()
    risks = np.array([f_data[x]['risk'] for x in keys])
    survuval = np.array([f_data[x]['survival'] for x in keys])
    censorship = np.array([f_data[x]['censorship'] for x in keys])
    c_index = concordance_index_censored((1-censorship).astype(bool), survuval, risks, tied_tol=1e-08)[0]
    c_index2 = concordance_index(survuval,-risks,censorship)
    print('Val c-Index: {:.4f} val_index2:  {:.4f}'.format(c_index,c_index2))