from __future__ import print_function, division
import math
import os
import pdb
import pickle
import re
from secrets import choice

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset

from utils.utils import generate_split, nth
np.random.seed(2022)

class Generic_WSI_Survival_Dataset(Dataset):
    def __init__(self,
        csv_path = 'dataset_csv/ccrcc_clean.csv', batch_size=2,mode = 'path',
        shuffle = False, seed = 7, print_info = True, n_bins = 4, ignore=[],
        patient_strat=False, label_col = None, filter_dict = {}, eps=1e-6):
        r"""
        Generic_WSI_Survival_Dataset 

        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir = None
        self.batch_size =batch_size

        slide_data = pd.read_csv(csv_path)#, index_col=0, low_memory=False)
        slide_data = slide_data[['case_id','slide_id','survival_months','censorship']]
        if 'case_id' not in slide_data:
            slide_data.index = slide_data.index.str[:12]
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)

        # if "IDC" in slide_data['oncotree_code']: # must be BRCA (and if so, use only IDCs)
        #     slide_data = slide_data[slide_data['oncotree_code'] == 'IDC']

        if not label_col:
            label_col = 'survival'
        else:
            assert label_col in slide_data.columns
        self.label_col = label_col
        ###shuffle data
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        #self.slide_data = slide_data
        patients_df = slide_data.drop_duplicates(['case_id']).copy()  # 957
        uncensored_df = patients_df[patients_df['censorship'] < 1]  # 130

        #disc_labels, bins = pd.cut(uncensored_df[label_col], bins=n_bins, right=False, include_lowest=True, labels=np.arange(n_bins), retbins=True)
        # 按照数据出现频率百分比划分，比如要把数据分为四份，则四段分别是数据的0-25%，25%-50%，50%-75%，75%-100%，每个间隔段里的元素个数都是相同的
        disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps
        
        disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))

        patient_dict = {}
        slide_data = slide_data.set_index('case_id')
        # not_exists = []
        wsi_num = 0
        
        for patient in patients_df['case_id']:
            slide_ids = slide_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            # my_slide_ids = []
            wsi_num += len(slide_ids)
            
            # for my_slide_id in slide_ids:
            #     mypath = os.path.join('/data_local2/ljjdata/TCGA/GBM/graph_files/',my_slide_id.replace('.svs','.pt'))
            #     if os.path.exists(mypath):
            #         my_slide_ids.append(my_slide_id)
            #         wsi_num += 1
            #     else:
            #         not_exists.append(my_slide_id)
            # if len(my_slide_ids)>0:
            #     my_slide_ids = np.array(my_slide_ids)
            patient_dict.update({patient:slide_ids})
            # else:
            #     # print('my_slide_ids len is 0',patient)
            #     pass

        print('wsi num is ', wsi_num)
        print('patient num is ', len(patient_dict.keys()))
        
        self.patient_dict = patient_dict
    
        slide_data = patients_df  # 1023->957
        slide_data.reset_index(drop=True, inplace=True)
        slide_data = slide_data.assign(slide_id=slide_data['case_id'])  # 生成新列

        label_dict = {}
        key_count = 0
        for i in range(len(q_bins)-1):
            for c in [0, 1]:
                print('{} : {}'.format((i, c), key_count))
                label_dict.update({(i, c):key_count})
                key_count+=1

        self.label_dict = label_dict
        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = key
            censorship = slide_data.loc[i, 'censorship']
            key = (key, int(censorship))
            slide_data.at[i, 'label'] = label_dict[key]

        self.bins = q_bins
        self.num_classes=len(self.label_dict)
        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {'case_id':patients_df['case_id'].values, 'label':patients_df['label'].values}

        new_cols = list(slide_data.columns[-2:]) + list(slide_data.columns[:-2])
        slide_data = slide_data[new_cols]
        self.slide_data = slide_data
        self.metadata = slide_data.columns[:11]
        self.mode = mode
        self.cls_ids_prep()

        if print_info:
            self.summarize()


        # if print_info:
        #     self.summarize()

    


    def cls_ids_prep(self):
        self.patient_cls_ids = [[] for i in range(self.num_classes)]        
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]


    def patient_data_prep(self):
        patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
        patient_labels = []
        
        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations[0]] # get patient label
            patient_labels.append(label)
        
        self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}


    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        disc_labels, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)//self.batch_size

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
        for i in range(self.num_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))


    def get_split_from_df(self, all_splits: dict, split_key: str='train', scaler=None):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())  # 是否包含split(train/val)字符串
            df_slice = self.slide_data[mask].reset_index(drop=True)

            #tmp 
            tmp = self.patient_dict.keys()
            mask = df_slice['slide_id'].isin(list(tmp))  # 是否包含split(train/val)字符串
            df_slice = df_slice[mask].reset_index(drop=True)
            # tmp

            split = Generic_Split(df_slice, batch_size=self.batch_size,metadata=self.metadata, mode=self.mode,  data_dir=self.data_dir, label_col=self.label_col, patient_dict=self.patient_dict, num_classes=self.num_classes)
        else:
            split = None
        
        return split


    def return_splits(self, from_id: bool=True, csv_path: str=None):
        if from_id:
            raise NotImplementedError
        else:
            assert csv_path 
            csv_path = csv_path.replace('splits_','fold_')
            all_splits = pd.read_csv(csv_path)
            train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')
            val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')

        return train_split, val_split


    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None

class Generic_MIL_Survival_Dataset(Generic_WSI_Survival_Dataset):
    def __init__(self, data_dir, batch_size, mode: str='path', **kwargs):
        super(Generic_MIL_Survival_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.use_h5 = False
        self.batch_size = batch_size

    def load_from_h5(self, toggle):
        self.use_h5 = toggle


    def __getitem__(self, index):
        idxs = self.batch_sample[index]
        
        survival_state = []
        survival_time = []
        survival_label = []
        batch_slide_ids = []

        for idx in idxs:
            case_id = self.slide_data['case_id'][idx]
            label = self.slide_data['disc_label'][idx]
            event_time = self.slide_data[self.label_col][idx]
            c = 1- self.slide_data['censorship'][idx]
            slide_ids = self.patient_dict[case_id]
            
            survival_state.extend([c])
            survival_time.extend([event_time])
            survival_label.extend([label])
            batch_slide_ids.extend(slide_ids)

        
    



        if type(self.data_dir) == dict:
            source = self.slide_data['oncotree_code'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir
        
        if not self.use_h5:
            if self.data_dir:
                if self.mode == 'path':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    return (path_features, label, event_time, c)

                elif self.mode == 'cluster':
                    path_features = []
                    cluster_ids = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'feat_dir/pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                        cluster_ids.extend(self.fname2ids[slide_id[:-4]+'.pt'])
                    path_features = torch.cat(path_features, dim=0)
                    cluster_ids = torch.Tensor(cluster_ids)
                    return (path_features, cluster_ids, label, event_time, c)

                elif self.mode == 'graph':
                    path_features = []
                    from datasets.BatchWSI import BatchWSI
                    for slide_id in batch_slide_ids:
                        wsi_path = os.path.join(data_dir, 'graph_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)

                    path_features = BatchWSI.from_data_list(path_features, update_cat_dims={'edge_latent': 1})
                    # return (path_features, label, event_time, c)
                    return (path_features, survival_label, survival_time, survival_state)
                    # return {'festures':path_features, 'label':label, 'event_time':event_time, 'state':c}

                else:
                    raise NotImplementedError('Mode [%s] not implemented.' % self.mode)
                ### <--
            else:
                return slide_ids, label, event_time, c


class Generic_Split(Generic_MIL_Survival_Dataset):
    def __init__(self, slide_data, batch_size,metadata, mode, data_dir=None, label_col=None, patient_dict=None, num_classes=2):
        self.use_h5 = False
        self.slide_data = slide_data
        self.metadata = metadata
        self.batch_size = batch_size
        self.mode = mode
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        self.batch_sample = self.get_batch_sample()
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]
        
        if os.path.isfile(os.path.join(data_dir, 'fast_cluster_ids.pkl')):
            with open(os.path.join(data_dir, 'fast_cluster_ids.pkl'), 'rb') as handle:
                self.fname2ids = pickle.load(handle)

        def series_intersection(s1, s2):
            return pd.Series(list(set(s1) & set(s2)))

    def get_batch_sample(self):
        num = len(self.slide_data)
        choice_num = num // self.batch_size
        batch_sample = np.random.choice(num,[choice_num,self.batch_size],replace=False)

        return batch_sample


    def __len__(self):
        return len(self.slide_data)//self.batch_size