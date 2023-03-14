import torch
from pycox.models.utils import pad_col

class CELoss(object):

    def __init__(self, alpha=0.15):
        self.alpha = alpha

    def __call__(self, results_dict, survival_time,state_label,interval_label): 

        return ce_loss(results_dict, survival_time, state_label,interval_label)


def ce_loss(results_dict, survival_time,state_label,interval_label, eps=1e-7):
    hazards = results_dict['logits']
    hazards = pad_col(hazards).softmax(1)
    # .softmax(1)
    batch_size = len(survival_time)
    surv_time = survival_time.view(batch_size, 1) # ground truth bin, 1,2,...,k
    sta_label = state_label.view(batch_size, 1).float() #censorship status, 0 or 1
    inter_label = interval_label.view(batch_size, 1)

    
    # death = (1 - censoring) * (torch.log(torch.gather(hazards, 1, survival_time.long()).clamp(min=eps)))
    death = sta_label * torch.log(torch.gather(hazards, 1, inter_label.long()))

    cumsum = hazards.cumsum(1)
    censor = torch.log(1-torch.gather(cumsum, 1, inter_label.long()))

    # censor = []
    # for b in range(batch_size):
    #     indices = interval_label[b].long()+1
    #     sample = torch.sum(torch.log(hazards[b,indices:]))
    #     censor.append(sample)
    # censor = torch.stack(censor, dim=0)

    censor = (1-state_label) * censor
    # loss = - (death.view(-1) + censor)
    loss = - (death + censor)
    loss = loss.mean()
    return loss