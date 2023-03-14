import pandas as pd
import os

survive_info_path = "/data_local/ljjdata/TCGA/clinical/GBM/survive_info_860.csv"
pt_flie = '/data_local2/ljjdata/TCGA/GBM/graph_files/'
df = pd.read_csv(survive_info_path)
new_df = pd.DataFrame()
new_df['case_id'] = df['patient_id']
new_df['slide_id'] = df['svs_filename']
nan_index = (df['days_to_death'].isnull())
new_df.loc[~nan_index,'survival_months'] = df.loc[~nan_index,'days_to_death']
new_df.loc[nan_index,'survival_months'] = df.loc[nan_index,'days_to_last_follow_up']
new_df['censorship'] = (df['vital_status'] == 'Alive').astype('int')
# cheak pt
for i, f in enumerate(new_df['slide_id']):
    pt_path = os.path.join(pt_flie,f.replace('.svs','.pt'))
    if not os.path.exists(pt_path):
        new_df.drop(i,axis=0,inplace=True)

new_df = new_df.reset_index(drop=True)
# 删除survival_months=0        
for i, f in enumerate(new_df['slide_id']):
    if new_df.loc[i,'survival_months'] == 0:
        new_df.drop(i,axis=0,inplace=True)

new_df = new_df.reset_index(drop=True)
# 删除nan
new_df.dropna(how='any',axis=0,inplace=True)

new_df.to_csv('/data_local/ljjdata/TCGA/clinical/GBM/tcga_gbm_all_clean.csv.zip',index=False)
new_df.to_csv('/data/project/TCGA/code/Patch-GCN/datasets_csv/tcga_gbm_all_clean.csv.zip',index=False)
new_df.to_csv('/data_local/ljjdata/TCGA/clinical/GBM/tcga_gbm_all_clean.csv',index=False)
new_df.to_csv('/data/project/TCGA/code/Patch-GCN/datasets_csv/tcga_gbm_all_clean.csv',index=False)
