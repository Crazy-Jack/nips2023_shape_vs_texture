import torch
import torch.nn as nn
from torchvision import models

class TopKLayer(nn.Module):
    def __init__(self, topk=0.1, revert=False):
        super(TopKLayer, self).__init__()
        self.revert=revert
        self.topk=topk

    def sparse_hw(self, x, tau, topk, device='cuda'):
        n, c, h, w = x.shape
        if topk == 1:
            return x
        x_reshape = x.view(n, c, h * w)
        topk_keep_num = int(max(1, topk * h * w))
        _, index = torch.topk(x_reshape.abs(), topk_keep_num, dim=2)
        if self.revert:
            # Useless
            mask = (torch.ones_like(x_reshape) - torch.zeros_like(x_reshape).scatter_(2, index, 1)).to(device)
        else:
            mask = torch.zeros_like(x_reshape).scatter_(2, index, 1).to(device)
        # print("mask percent: ", mask.mean().item())
        sparse_x = mask * x_reshape
        sparsity_x = 1.0 - torch.where(sparse_x == 0.0)[0].shape[0] / (n * c * h * w)
        # print("sparsity -- ({}): {}".format((n, c, h, w), sparsity_x)) ## around 9% decrease to 4% fired eventually this way
        if tau == 1.0:
            return sparse_x.view(n, c, h, w) 
        # print("--- tau", tau)
        tau_x = x * torch.FloatTensor([1. - tau]).to(device)
        # print("sum of x used", tau_x.sum())
        return sparse_x.view(n, c, h, w) * torch.FloatTensor([tau]).to(device) + tau_x

    def forward(self, x):
        return self.sparse_hw(x,1,self.topk)

    
def topK_VGG(pretrain_weigth,topk, **kwargs):
    if pretrain_weigth=="":
        vgg16 = models.vgg16(pretrained=True)
        new_features = nn.Sequential(
            # layers up to the point of insertion
            *(list(vgg16.features.children())[:5]), # 4 is MaxPool2d
            TopKLayer(topk),
            *(list(vgg16.features.children())[5:10]),
            TopKLayer(topk),
            *(list(vgg16.features.children())[10:17]),
            TopKLayer(topk),
            *(list(vgg16.features.children())[17:24]),
            TopKLayer(topk),
            *(list(vgg16.features.children())[24:]),
            TopKLayer(topk),
        )
        vgg16.features = new_features
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vgg16 = vgg16.to(device)
        vgg16 = torch.nn.DataParallel(vgg16)
        return vgg16

    vgg16 = models.vgg16()
    new_features = nn.Sequential(
        # layers up to the point of insertion
        *(list(vgg16.features.children())[:5]), # 4 is MaxPool2d
        TopKLayer(topk),
        *(list(vgg16.features.children())[5:10]),
        TopKLayer(topk),
        *(list(vgg16.features.children())[10:17]),
        TopKLayer(topk),
        *(list(vgg16.features.children())[17:24]),
        TopKLayer(topk),
        *(list(vgg16.features.children())[24:]),
        TopKLayer(topk),
    )
    vgg16.features = new_features
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vgg16 = vgg16.to(device)
    vgg16 = torch.nn.DataParallel(vgg16)
    vgg16.load_state_dict(torch.load(pretrain_weigth))
    return vgg16

def topK_AlexNet(pretrain_weigth,topk, **kwargs):
    if pretrain_weigth=="":
        alexnet = models.alexnet(pretrained=True)
        new_features = nn.Sequential(
            # layers up to the point of insertion
            *(list(alexnet.features.children())[:3]), 
            TopKLayer(topk),
            *(list(alexnet.features.children())[3:6]),
            TopKLayer(topk),
            *(list(alexnet.features.children())[6:8]),
            TopKLayer(topk),
            *(list(alexnet.features.children())[8:10]),
            TopKLayer(topk),
            *(list(alexnet.features.children())[10:]),
            TopKLayer(topk),
        )
        alexnet.features = new_features
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        alexnet = alexnet.to(device)
        alexnet = torch.nn.DataParallel(alexnet)
        return alexnet