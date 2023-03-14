from textwrap import indent
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, KFold

skf = KFold(n_splits=5, random_state=233, shuffle=True)
df = pd.read_csv('/data/project/TCGA/code/Patch-GCN/datasets_csv/tcga_gbm_all_clean.csv.zip')
save_p = '/data/project/TCGA/code/Patch-GCN/splits/5foldcv/tcga_gbm/'
case_id = df.drop_duplicates(['case_id'])['case_id']
case_id = np.array(case_id)
for flod_idx, (train_idx, val_idx) in enumerate(skf.split(case_id, case_id)):
    train_case_id = case_id[train_idx]
    val_case_id = case_id[val_idx]

    new_df = pd.DataFrame()
    # new_df['train'] = train_case_id
    # new_df['val'] = val_case_id
    new_df = pd.concat([pd.DataFrame({'train': train_case_id}),pd.DataFrame({'val': val_case_id})],axis=1)
    save_path = os.path.join(save_p,'splits_'+str(flod_idx)+'.csv')
    dirname = os.path.dirname(save_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    new_df.to_csv(save_path,index=False)
    # pass