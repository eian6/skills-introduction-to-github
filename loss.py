import torch
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F


def kd_loss(logits, logits_t, alpha=1.0, T=10.0):
    ce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    kl_loss_fn = nn.KLDivLoss()
    labels_t = torch.where(logits_t > 0.0, torch.ones(logits_t.shape).to(logits_t.device),
                           torch.zeros(logits_t.shape).to(logits_t.device))
    ce_loss = ce_loss_fn(logits, labels_t)
    d_s = torch.log(torch.cat((torch.sigmoid(logits / T), 1 - torch.sigmoid(logits / T)), dim=1))
    d_t = torch.cat((torch.sigmoid(logits_t / T), 1 - torch.sigmoid(logits_t / T)), dim=1)
    kl_loss = kl_loss_fn(d_s, d_t) * T * T
    return ce_loss * alpha + (1 - alpha) * kl_loss


def KLDiv(edgex, edgey):
    eps = 1e-8
    edgex = edgex + eps
    edgey = edgey + eps
    diff = edgey * (torch.log(edgey) - torch.log(edgex))
    return torch.mean(torch.flatten(diff))



def graphKL_loss1(middle_feats_s, middle_feats_t):

    dist_t = F.softmax(middle_feats_t, dim=1)
    dist_s = F.softmax(middle_feats_s, dim=1)
    loss = F.kl_div(dist_s.log(), dist_t, reduction='batchmean')
    return loss


def graphKL_loss(models, middle_feats_s, train_mask,feats,adj, epoch, args):
    if epoch < args.s_epochs:
        t_model = models['t1_model']['model']
        middle_feats_t = t_model(feats,adj)
        middle_feats_t = middle_feats_t[train_mask]

    elif args.s_epochs <= epoch < 2 * args.s_epochs:
        t_model = models['t2_model']['model']
        middle_feats_t = t_model(feats,adj)
        middle_feats_t = middle_feats_t[train_mask]

    else:
        t_model = models['t3_model']['model']
        middle_feats_t = t_model(feats,adj)
        middle_feats_t = middle_feats_t[train_mask]


    dist_t = F.softmax(middle_feats_t, dim=1)
    dist_s = F.softmax(middle_feats_s, dim=1)
    loss = F.kl_div(dist_s.log(), dist_t, reduction='batchmean')
    return loss

def optimizing(models, loss, model_list):
    for model in model_list:
        models[model]['optimizer'].zero_grad()
    loss.backward()
    for model in model_list:
        models[model]['optimizer'].step()