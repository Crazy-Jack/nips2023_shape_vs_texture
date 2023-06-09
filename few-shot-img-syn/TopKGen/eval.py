import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import utils as vutils

import os
import random
import argparse
from tqdm import tqdm


import models, models_hier

def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

def resize(img):
    return F.interpolate(img, size=256)

def batch_generate(zs, netG, batch=8):
    g_images = []
    with torch.no_grad():
        for i in range(len(zs)//batch):
            g_images.append( netG(zs[i*batch:(i+1)*batch]).cpu() )
        if len(zs)%batch>0:
            g_images.append( netG(zs[-(len(zs)%batch):]).cpu() )
    return torch.cat(g_images)

def batch_save(images, folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    for i, image in enumerate(images):
        vutils.save_image(image.add(1).mul(0.5), folder_name+'/%d.jpg'%i)


if __name__ == "__main__":
    from TopKGen.options import get_parser
    args = get_parser() 

    noise_dim = 256
    device = torch.device('cuda:%d'%(args.cuda))
    using_noise = not args.nonoise
    ngf = 64
    nz = args.nz
    im_size = args.im_size
    
    net_ig = models.Generator(args, ngf=ngf, nz=nz, im_size=im_size, noise=using_noise)

    net_ig.to(device)


    checkpoint = torch.load(args.ckpt, map_location=lambda a,b: a)
    net_ig.load_state_dict(checkpoint['g'])
    avg_param_G = checkpoint['g_ema']
    load_params(net_ig, avg_param_G)
    iteration = checkpoint['iter']

    print('load checkpoint success, iteration %d'%iteration)

    net_ig.to(device)

    del checkpoint

    dist = os.path.join(args.name, 'eval_img')  
    os.makedirs(dist, exist_ok=True)

    # get current sample status
    existed_samples = os.listdir(dist)
    if existed_samples: 
        sampled_i = [int(str(name).split('.')[0]) for name in existed_samples]
        max_i = max(sampled_i)
        starting_batch = max_i // args.batch_size + 1
        print(f"Starting from {max_i}.png...")
        
    else:
        starting_batch = 0
    

    myrange = range(starting_batch, args.n_sample//args.batch_size)
    with torch.no_grad():
        for i in tqdm(myrange):
            noise = torch.randn(args.batch_size, noise_dim).to(device)
            g_imgs = net_ig(noise, evaluation=True)[0]
            g_imgs = F.interpolate(g_imgs, 512)
            for j, g_img in enumerate( g_imgs ):
                vutils.save_image(g_img.add(1).mul(0.5), 
                    os.path.join(dist, '%d.png'%(i*args.batch_size+j)))#, normalize=True, range=(-1,1))
