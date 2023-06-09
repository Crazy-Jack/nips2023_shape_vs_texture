"""
Module for TopK operation. 
Topk Module is chosen via find_topk_operation_using_name function by specifying
the argument --sp_hw_policy_name to match with each Topk Module class name in lowercase.
Each module will return [topk_sparsified_activation, loss_to_optimize_for_this_layer],
    if loss_to_optimize_for_this_layer is None, it won't be optimized.
"""
import torch.nn as nn 
import torch

import importlib
from torch.nn import Parameter as P



def find_topk_operation_using_name(model_name):
    """
    Find TopK module to use
    """
    model_filename = "TopKGen.TopkLoss.topk_loss_module"
    modellib = importlib.import_module(model_filename)
    model = None
    target_model_name = model_name
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, nn.Module):
            model = cls

    if model is None:
        print("In %s.py, there should be a class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model

#############################
#        Topk function      #
#############################
def sparse_hw(x, tau, topk_keep_num, device='cuda'):

    n, c, h, w = x.shape
    if topk_keep_num == h * w:
        return x
    x_reshape = x.view(n, c, h * w)
    
    _, index = torch.topk(x_reshape.abs(), topk_keep_num, dim=2)
    mask = torch.zeros_like(x_reshape).scatter_(2, index, 1).to(device)
    sparse_x = mask * x_reshape
    sparsity_x = 1.0 - torch.where(sparse_x == 0.0)[0].shape[0] / (n * c * h * w)
    if tau == 1.0:
        return sparse_x.view(n, c, h, w)
    
    tau_x = x * torch.FloatTensor([1. - tau]).to(device)
    return sparse_x.view(n, c, h, w) * torch.FloatTensor([tau]).to(device) + tau_x


def sparse_hw_replace_w_mean(x, topk_keep_num):
    b, c, h, w = x.shape
    x_reshape = x.reshape(b, c, h * w)
    value, index = torch.topk(x_reshape, topk_keep_num, dim=2)
    value_mean = value.mean(2, keepdim=True).repeat(1, 1, topk_keep_num)
    out = torch.zeros_like(x_reshape).scatter_(2, index, value_mean)
    out = out.reshape(b, c, h, w)
    return out


#############################
#        Topk Module        #
#############################

class IdentityBase(nn.Module):
    def __init__(self):
        super(IdentityBase, self).__init__()
    
    def forward(self, x):
        # do nothing
        return x


class TopKMaskBase(nn.Module):
    def __init__(self, topk, topk_keep_num, topk_apply_dim, args, reso):
        super(TopKMaskBase, self).__init__()
        self.topk_keep_num = max(topk_keep_num, 1)
        self.topk_apply_dim = topk_apply_dim
        self.args = args
    
    def forward(self, x, tau, evaluation=False):
        raise NotImplementedError


class TopKMaskHW(TopKMaskBase):
    def __init__(self, topk, topk_keep_num, topk_apply_dim, args, reso):
        super(TopKMaskHW, self).__init__(topk, topk_keep_num, topk_apply_dim, args, reso)
    
    def forward(self, x, tau, evaluation=False):
        return sparse_hw(x, tau, self.topk_keep_num), None # return None for normal topk without loss function


class TopKMaskHWMeanReplace(TopKMaskBase):
    def __init__(self, topk, topk_keep_num, topk_apply_dim, args, reso):
        super(TopKMaskHWMeanReplace, self).__init__(topk, topk_keep_num, topk_apply_dim, args, reso)
    
    def forward(self, x, tau, evaluation=False):
        return sparse_hw_replace_w_mean(x, self.topk_keep_num), None # return None for normal topk without loss function