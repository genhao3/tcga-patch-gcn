import torch
import os
import numpy as np
from utils.utils import get_split_loader
from utils.core_utils import summary_survival
from models.model_graph_mil import PatchGCN_Surv
from datasets.dataset_survival import Generic_MIL_Survival_Dataset
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定第一块gpu

def test(model,val_loader,n_classes,models_dir,k):
    # model.load_state_dict(torch.load(os.path.join(models_dir, "s_{}_latest_checkpoint.pt".format(k))))
    model.load_state_dict(torch.load(os.path.join(models_dir, "k_{}_latest_checkpoint.pt".format(k))))
    results_val_dict, val_cindex = summary_survival(model, val_loader, n_classes)
    
    return val_cindex


if __name__ == "__main__":
    k_folds = 5
    data_set = 'BRCA'
    loss_name = 'ce'


    model_type = 'patchgcn'

    if data_set == 'BRCA':
        split_name = 'tcga_brca'
    elif data_set == 'GBM':
        split_name = 'tcga_gbm'
    elif data_set == 'LUSC':
        split_name = 'tcga_lusc'

    
    split_dir = r'/data/project/TCGA/code/Patch-GCN/splits/5foldcv/'+split_name
    models_dir = 'ljj_results/5foldcv/PatchGCN_'+loss_name+'_a0.0_5foldcv_gc32/'+split_name+'_PatchGCN_'+loss_name+'_a0.0_5foldcv_gc32_s1'
    # models_dir = '/data/project/TCGA/code/Patch-GCN/results/5foldcv/PatchGCN_4_spatialsm_nll_surv_a0.0_5foldcv_gc32_sig/tcga_brca_PatchGCN_4_spatialsm_nll_surv_a0.0_5foldcv_gc32_sig_s1'
    dataset = Generic_MIL_Survival_Dataset(csv_path = '/data/project/TCGA/code/Patch-GCN/datasets_csv/'+split_name+'_all_clean.csv.zip',
                                           mode = 'graph',
                                           data_dir = '/data_local2/ljjdata/TCGA/'+data_set+'/',
                                        #    data_dir= os.path.join(args.data_root_dir, study_dir),
                                           shuffle = False, 
                                           seed = 1, 
                                           print_info = True,
                                           patient_strat= False,
                                           n_bins=4,
                                           label_col = 'survival_months',
                                           ignore=[])
    val_cindexs = []
    c_index2s = []
    train_cindexs = []
    train_index2s = []
    for k in range(k_folds):
        train_dataset, val_dataset = dataset.return_splits(from_id=False, csv_path='{}/splits_{}.csv'.format(split_dir, k))

        print("Training on {} samples".format(len(train_dataset)))
        print("Validating on {} samples".format(len(val_dataset)))

        train_loader = get_split_loader(train_dataset, training=True, testing = False, 
                                    weighted = True, mode='graph', batch_size=1)

        val_loader = get_split_loader(val_dataset,  testing = False, mode='graph', batch_size=1)
        print('Done!')

        if model_type == 'patchgcn':
            model_dict = {'num_layers': 4, 'edge_agg': 'spatial', 'resample': 0.00, 'n_classes': 4}
            model = PatchGCN_Surv(**model_dict)

        model = model.to(torch.device('cuda'))
        # train_cindex,train_index2 = test(model,train_loader,4,models_dir,k)
        # train_cindexs.append(train_cindex)
        # train_index2s.append(train_index2)
        # print('{} Train c-Index: {:.4f} train_index2:  {:.4f}'.format(k, train_cindex,train_index2))

        val_cindex = test(model,val_loader,4,models_dir,k)
        # c_index2s.append(c_index2)
        val_cindexs.append(val_cindex)
        print('{} Val c-Index: {:.4f}'.format(k, val_cindex))
    
    # print('Mean Train c-Index: {:.4f} std {:.4f} train_index2:  {:.4f} std {:.4f}'.format(np.mean(train_cindexs),np.std(train_cindexs),np.mean(train_index2s),np.std(train_index2s)))
    print('Mean Val c-Index: {:.4f} std {:.4f}'.format(np.mean(val_cindexs),np.std(val_cindexs)))