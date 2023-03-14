import pickle
import torch
import numpy as np
import torch.nn as nn
import pdb
import subprocess,os
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
# from torch_geometric.data import DataLoader
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from torch.utils.data import Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler
import torch.optim as optim
import pdb
import torch.nn.functional as F
import math
from itertools import islice
import collections

from torch.utils.data.dataloader import default_collate
import torch_geometric
from torch_geometric.data import Batch

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SubsetSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def collate_MIL(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    label = torch.LongTensor([item[1] for item in batch])
    return [img, label]

def collate_features(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    coords = np.vstack([item[1] for item in batch])
    return [img, coords]

def collate_MIL_survival(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    label = torch.LongTensor([item[1] for item in batch])
    event_time = np.array([item[2] for item in batch])
    c = torch.FloatTensor([item[3] for item in batch])
    return [img, label, event_time, c]

def collate_MIL_survival_cluster(batch):
    img = torch.cat([item[0] for item in batch], dim = 0)
    cluster_ids = torch.cat([item[1] for item in batch], dim = 0).type(torch.LongTensor)
    label = torch.LongTensor([item[2] for item in batch])
    event_time = np.array([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])
    return [img, cluster_ids, label, event_time, c]

def collate_MIL_survival_graph(batch):
    elem = batch[0]
    elem_type = type(elem)        
    transposed = zip(*batch)
    return [samples[0] if isinstance(samples[0], torch_geometric.data.Batch) else default_collate(samples) for samples in transposed]


def get_simple_loader(dataset, batch_size=1):
    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    loader = DataLoader(dataset, batch_size=batch_size, sampler = sampler.SequentialSampler(dataset), collate_fn = collate_MIL, **kwargs)
    return loader 

def get_split_loader(split_dataset, training = False, testing = False, weighted = False, mode='coattn', batch_size=1):
    """
        return either the validation loader or training loader 
    """
    if mode == 'graph':
        print("asdf")
        collate = collate_MIL_survival_graph
    elif mode == 'cluster':
        collate = collate_MIL_survival_cluster
    else:
        collate = collate_MIL_survival

    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    if not testing:
        if training:
            if weighted:
                weights = make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(split_dataset, batch_size=1, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate, **kwargs)    
            else:
                loader = DataLoader(split_dataset, batch_size=1, sampler = RandomSampler(split_dataset), collate_fn = collate, **kwargs)
        else:
            loader = DataLoader(split_dataset, batch_size=1, sampler = SequentialSampler(split_dataset), collate_fn = collate, **kwargs)
    
    else:
        ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
        loader = DataLoader(split_dataset, batch_size=1, sampler = SubsetSequentialSampler(ids), collate_fn = collate, **kwargs )

    return loader

def get_optim(model, args):
    if args.opt == "adam":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    else:
        raise NotImplementedError
    return optimizer

def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)
    
    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
    
    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)


def generate_split(cls_ids, val_num, test_num, samples, n_splits = 5,
    seed = 7, label_frac = 1.0, custom_test_ids = None):
    indices = np.arange(samples).astype(int)
    
    pdb.set_trace()
    if custom_test_ids is not None:
        indices = np.setdiff1d(indices, custom_test_ids)

    np.random.seed(seed)
    for i in range(n_splits):
        all_val_ids = []
        all_test_ids = []
        sampled_train_ids = []
        
        if custom_test_ids is not None: # pre-built test split, do not need to sample
            all_test_ids.extend(custom_test_ids)

        for c in range(len(val_num)):
            possible_indices = np.intersect1d(cls_ids[c], indices) #all indices of this class
            remaining_ids = possible_indices

            if val_num[c] > 0:
                val_ids = np.random.choice(possible_indices, val_num[c], replace = False) # validation ids
                remaining_ids = np.setdiff1d(possible_indices, val_ids) #indices of this class left after validation
                all_val_ids.extend(val_ids)

            if custom_test_ids is None and test_num[c] > 0: # sample test split

                test_ids = np.random.choice(remaining_ids, test_num[c], replace = False)
                remaining_ids = np.setdiff1d(remaining_ids, test_ids)
                all_test_ids.extend(test_ids)

            if label_frac == 1:
                sampled_train_ids.extend(remaining_ids)
            
            else:
                sample_num  = math.ceil(len(remaining_ids) * label_frac)
                slice_ids = np.arange(sample_num)
                sampled_train_ids.extend(remaining_ids[slice_ids])

        yield sorted(sampled_train_ids), sorted(all_val_ids), sorted(all_test_ids)


def nth(iterator, n, default=None):
    if n is None:
        return collections.deque(iterator, maxlen=0)
    else:
        return next(islice(iterator,n, None), default)

def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

    return error

def make_weights_for_balanced_classes_split(dataset):
    N = float(len(dataset))                                           
    weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        y = dataset.getlabel(idx)                        
        weight[idx] = weight_per_class[y]                                  

    return torch.DoubleTensor(weight)

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


def dfs_unfreeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        dfs_unfreeze(child)

def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    '''
    Y is label
    '''
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    S_padded = torch.cat([torch.ones_like(c), S], 1) #S(-1) = 0, all patients are alive from (-inf, 0) by definition
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    uncensored_loss = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y.long()).clamp(min=eps)) + torch.log(torch.gather(hazards, 1, Y.long()).clamp(min=eps))) # gather收集输入的特定维度指定位置的数值
    censored_loss = - c * torch.log(torch.gather(S_padded, 1, (Y+1).long()).clamp(min=eps)) # gather收集输入的特定维度指定位置的数值,clamp将input 的值控制在min 和 max 之间,
    neg_l = censored_loss + uncensored_loss
    loss = (1-alpha) * neg_l + alpha * uncensored_loss
    loss = loss.mean()
    return loss


def ce_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    batch_size = len(Y)
    Y = Y.view(batch_size, 1) # ground truth bin, 1,2,...,k
    c = c.view(batch_size, 1).float() #censorship status, 0 or 1
    if S is None:
        S = torch.cumprod(1 - hazards, dim=1) # surival is cumulative product of 1 - hazards
    # without padding, S(0) = S[0], h(0) = h[0]
    # after padding, S(0) = S[1], S(1) = S[2], etc, h(0) = h[0]
    #h[y] = h(1)
    #S[1] = S(1)
    S_padded = torch.cat([torch.ones_like(c), S], 1)
    reg = -(1 - c) * (torch.log(torch.gather(S_padded, 1, Y.long())+eps) + torch.log(torch.gather(hazards, 1, Y.long()).clamp(min=eps)))
    ce_l = - c * torch.log(torch.gather(S, 1, Y.long()).clamp(min=eps)) - (1 - c) * torch.log(1 - torch.gather(S, 1, Y.long()).clamp(min=eps))
    loss = (1-alpha) * ce_l + alpha * reg
    loss = loss.mean()
    return loss


class CrossEntropySurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None): 
        if alpha is None:
            return ce_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return ce_loss(hazards, S, Y, c, alpha=alpha)

# loss_fn(hazards=hazards, S=S, Y=Y_hat, c=c, alpha=0)
class NLLSurvLoss(object):
    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, hazards, S, Y, c, alpha=None):
        if alpha is None:
            return nll_loss(hazards, S, Y, c, alpha=self.alpha)
        else:
            return nll_loss(hazards, S, Y, c, alpha=alpha)
    # h_padded = torch.cat([torch.zeros_like(c), hazards], 1)
    #reg = - (1 - c) * (torch.log(torch.gather(hazards, 1, Y)) + torch.gather(torch.cumsum(torch.log(1-h_padded), dim=1), 1, Y))


class CoxSurvLoss(object):
    # def __call__(self, hazards, S, Y,c):
    def __call__(self, S, Y, c):
        # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
        # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
        # current_batch_len = len(S)
        # R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        # for i in range(current_batch_len):
        #     for j in range(current_batch_len):
        #         R_mat[i,j] = S[j] >= S[i]

        # R_mat = torch.FloatTensor(R_mat).to(device)
        # theta = hazards.reshape(-1)
        # exp_theta = torch.exp(theta)
        # loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * (1-c))
        # return loss_cox
        '''
        参考论文公式(4-2)
        predict:预测的风险 shape batch size
        label_os:随访时间
        label_oss:是否发生损伤
        '''
        predict = 1-torch.sum(S, dim=1)
        index = torch.argsort(Y, dim=0, descending=True)  # 在Batch Size 获取降序的索引
        # 匹配降序的索引
        # Xi
        risk = torch.gather(input=predict, dim=0, index=index) # 根据OS的降序 改变predict 、oss 的顺序    患者 的生存时间 与风险值不对应
        status = torch.gather(input=c, dim=0, index=index)
        # label_os = torch.sort(label_os,dim=0,descending=True).values
        # torch.exp(x) y=e^x
        # 风险
        hazard_ratio = torch.exp(risk)
        '''
        a=[1,2,3]
        torch.cumsum:累加,b=[1,3,6]
        '''
        #XI,log e^x = x
        # log i:ti≥tj的样本的风险累加和
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0)) # 包括本身的风险值
        # temp = torch.ones_like(hazard_ratio).cuda()
        # temp = torch.cumsum(temp, dim=0)
        # log_risk = torch.log(torch.cumsum(hazard_ratio / temp, dim=0)) # 包括本身的风险值

        # Rj-log i:ti≥tj的样本的风险累加和 参考硕士论文公式(4-2)
        partial_likelihood = risk - log_risk
        # 观察到复发的样本label_oss=1
        uncensored_likelihood = partial_likelihood * status  # 除去无结局的样本
    
        num_observed_events = torch.sum(status)
        # 如果mini-batch 都为无结局

        num_observed_events = num_observed_events.float()
        #合并a,b两个tensor a>0的地方保存，防止分母为0
        # 如果全为负样本，0->1e-7
        num_observed_events = torch.where(num_observed_events > 0, num_observed_events,
                                        torch.tensor(1e-7, device=num_observed_events.device,))
        # 类似除以batch size
        loss = -torch.sum(uncensored_likelihood) / num_observed_events

        return loss



class CISurvLoss(object):
    def __call__(predict_risk, survive_time, dead_status, **kwargs):
        index = torch.argsort(survive_time, dim=0, descending=False)  # 在Batch Size 获取升序
        # 匹配降序的索引
        risk = torch.gather(input=predict_risk, dim=0, index=index) # 根据OS的降序 改变predict 、oss 的顺序    患者 的生存时间 与风险值不对应
        status = torch.gather(input=dead_status, dim=0, index=index)

        n = 0
        ci_loss = 0
        for i, l in enumerate(status):
            if l == 1 and i<len(status)-1:
                dr = risk-risk[i]
                ci_loss += torch.sum(1-torch.exp(dr[i+1:]))
                n += len(dr[i+1:])

        ci_loss = ci_loss / n

        return -ci_loss


class WCISurvLoss(object):
    def __call__(predict_risk, survive_time, dead_status, **kwargs):
        index = torch.argsort(survive_time, dim=0, descending=False)  # 在Batch Size 获取升序
        # 匹配降序的索引
        risk = torch.gather(input=predict_risk, dim=0, index=index) # 根据OS的降序 改变predict 、oss 的顺序    患者 的生存时间 与风险值不对应
        status = torch.gather(input=dead_status, dim=0, index=index)
        # 风险
        wci_loss = []
        for i, l in enumerate(status):
            if l == 1 and i<len(status)-1:
                dr = risk[i] - risk
                loss = torch.exp(-(dr[i:])/0.1).mean()
                wci_loss.append(loss)

        return torch.stack(wci_loss, dim=0).mean()


class QCISurvLoss(object):
    def __call__(predict_risk, survive_time, dead_status, **kwargs):
        index = torch.argsort(survive_time, dim=0, descending=False)  # 在Batch Size 获取升序
        status_time = torch.gather(input=survive_time, dim=0, index=index)
        # 匹配降序的索引
        risk = torch.gather(input=predict_risk, dim=0, index=index) # 根据OS的降序 改变predict 、oss 的顺序    患者 的生存时间 与风险值不对应
        status = torch.gather(input=dead_status, dim=0, index=index)
        # 风险
        qci_loss = []
        for i, l in enumerate(status):
            if l == 1 and i<len(status)-1:
                gamma = status_time - status_time[i]
                dr = risk[i] - risk
                loss = torch.exp(-(dr[i:])*gamma[i:]).mean()
                qci_loss.append(loss)

        return torch.stack(qci_loss, dim=0).mean()


def l1_reg_all(model, reg_type=None):
    l1_reg = None

    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)
    return l1_reg

def get_custom_exp_code(args):
    ### New exp_code
    exp_code = '_'.join(args.split_dir.split('_')[:2])
    dataset_path = 'datasets_csv'
    param_code = ''

    if args.model_type == 'amil':
        param_code += 'AMIL'
    elif args.model_type == 'deepset':
        param_code += 'DS'
    elif args.model_type == 'mi_fcn':
        param_code += 'MIFCN'
    elif args.model_type == 'dgc':
        agg = 'latent' if args.edge_agg == 'latent' else 'spatial'
        param_code += 'DGC_%s%s' % (agg)
    elif args.model_type == 'patchgcn':
        param_code += 'PatchGCN'
    else:
        raise NotImplementedError

    if args.resample > 0:
        param_code += '_resample'

    param_code += '_%s' % args.bag_loss

    param_code += '_a%s' % str(args.alpha_surv)

    if args.lr != 2e-4:
        param_code += '_lr%s' % format(args.lr, '.0e')

    if args.reg_type != 'None':
        param_code += '_reg%s' % format(args.lambda_reg, '.0e')

    # param_code += '_%s' % args.which_splits.split("_")[0]

    if args.batch_size != 1:
        param_code += '_b%s' % str(args.batch_size)

    if args.gc != 1:
        param_code += '_gc%s' % str(args.gc)

    args.exp_code = exp_code + '_' + param_code
    args.param_code = param_code
    args.dataset_path = dataset_path

    return args


def model_select_save(args, val_cindex, model, epoch, trained_epochs,k):

    index_ranking = np.argsort(-np.array(val_cindex))  # np.argsort从小到大排序,返回排序后的下标
    epoch_ranking = np.array(trained_epochs)[index_ranking]
    
    # check if current epoch is among the top-k epchs.
    if epoch in epoch_ranking[:5]:
        model_path = os.path.join(args.results_dir, "k_{}_epoch_{}_checkpoint.pt".format(k,epoch))
        
        torch.save(model.state_dict(), model_path)

        # delete params of the epoch that just fell out of the top-k epochs.
        #删除c-index最小的模型
        if len(epoch_ranking) > 5:
            epoch_rm = epoch_ranking[5]
            subprocess.call('rm {}'.format(os.path.join(args.results_dir, "k_{}_epoch_{}_checkpoint.pt".format(k,epoch_rm))), shell=True)