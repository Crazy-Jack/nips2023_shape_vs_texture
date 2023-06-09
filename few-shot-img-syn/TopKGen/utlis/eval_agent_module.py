"""
Eval agent for dynamically computing FID and KID
"""

import torch
import torch_fidelity
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import numpy as np

import os
import argparse
from tqdm import tqdm
from datetime import datetime

from TopKGen.benchmarking.operation import ImageFolder


class EvalFakeDataset(Dataset):
    def __init__(self, image_tensor, transform=None):
        
        super(EvalFakeDataset, self).__init__()
        self.image_tensor = image_tensor
        self.transform = transform

    
    def __getitem__(self, idx):
        data = self.image_tensor[idx, :]
        # print(self.image_tensor.shape)
        if self.transform:
            data = self.transform(data)

        return data 
    
    def __len__(self):
        return self.image_tensor.shape[0]



class Evaler:
    def __init__(self, real_image_path, log_path, version_name, sample_batch_size, n_sample, eval_size, z_dim, z_type, test_times=3, fid_feature_extractor_name='inception-v3-compat'):
        # real image
        self.real_image_path = real_image_path
        self.eval_size = eval_size
        self.transform = transforms.Compose(
            [
                transforms.Resize( (eval_size, eval_size) ),
                transforms.ToTensor(),
                lambda x: (255 * (x.clamp(-1, 1) * 0.5 + 0.5)).to(torch.uint8),
            ]
        )
        self.dset_real = ImageFolder(self.real_image_path, transform=self.transform)

        # noise
        self.z_type = z_type # ['normal', 'random']
        self.z_dim = z_dim

        # create log file
        print(f"fid_feature_extractor_name {fid_feature_extractor_name}")
        self.log_path = os.path.join(log_path, f'{version_name}_{fid_feature_extractor_name}.log')
        self.version_name = version_name
        with open(self.log_path, 'a') as f:
            f.write(f"\n\n ---- Train for version {self.version_name} ---- \n\n")

        self.sample_batch_size = sample_batch_size
        self.n_sample = n_sample

        # reproducibility
        self.test_times = test_times

        # best result
        self.best_results = {
            'fid': 999999,
            'kid_mean': 999999,
            'kid_std': 999999
        }

        # shape bias networks
        self.fid_feature_extractor_name = fid_feature_extractor_name

    @torch.no_grad()
    def sample(self, netG):
        # function that sample from the network G and create a torch dataset from it
        myrange = range(self.n_sample//self.sample_batch_size)
        images = []
        for i in tqdm(myrange):
            noise = torch.randn(self.sample_batch_size, self.z_dim).to(next(netG.parameters()).device)
            g_imgs = netG(noise, evaluation=True)[0]
            g_imgs = F.interpolate(g_imgs, 512).add(1).mul(0.5)
            g_imgs = F.interpolate(g_imgs, self.eval_size)
            g_imgs = (255 * (g_imgs.clamp(-1, 1) * 0.5 + 0.5)).to(torch.uint8)
            g_imgs = g_imgs.cpu()
            images.append(g_imgs)
        images = torch.cat(images, dim=0)
        fake_dataset = EvalFakeDataset(images)
        return fake_dataset
    

    def eval_basic(self, netG):
        """compute FID and KID score here"""
        fake_dataset = self.sample(netG)
        metrics = torch_fidelity.calculate_metrics(
            input1=fake_dataset,
            input2=self.dset_real,
            fid=True,
            kid=True,
            verbose=True,
            kid_subset_size=100,
            feature_extractor=self.fid_feature_extractor_name,
        )
        del fake_dataset
        return metrics

    def eval(self, netG, iteration):
        fids = []
        kids = []
        kids_std = []
        for _ in range(self.test_times):
            metrics = self.eval_basic(netG)
            fids.append(metrics['frechet_inception_distance'])
            kids.append(metrics['kernel_inception_distance_mean'])
            kids_std.append(metrics['kernel_inception_distance_std'])

        local_best_ind = fids.index(min(fids))
        eval_results = {
            'fid': round(fids[local_best_ind], 6),
            'kid_mean': round(kids[local_best_ind], 6),
            'kid_std': round(kids_std[local_best_ind], 6),
        }

        is_new_best_results = self.is_better_result(eval_results)
        if is_new_best_results:
            self.best_results = eval_results

        # log
        self.log(eval_results, iteration)

        return is_new_best_results, eval_results
    
    def is_better_result(self, new_results):
        if self.best_results['fid'] > new_results['fid']:
            return True 
        else:
            return False

    def log(self, results, iteration):
        """
        log the results into formatted strings
        """
        log_str = f"{str(datetime.now()):30} Iter: {iteration:10} | Best FID: {self.best_results['fid']:10} | Current: FID: {results['fid']:10} | KID: {results['kid_mean']:10} +- {results['kid_std']:7}\n"
        print(log_str)
        with open(self.log_path, 'a') as f:
            f.write(log_str)

    def analysis(self, netG, iteration):
        myrange = range(self.n_sample//self.sample_batch_size)
        images = []
        intermediate = []
        for i in tqdm(myrange):
            noise = torch.randn(self.sample_batch_size, self.z_dim).to(next(netG.parameters()).device)
            g_imgs = netG(noise, evaluation=True)[0]
            g_imgs = F.interpolate(g_imgs, 512).add(1).mul(0.5)
            g_imgs = F.interpolate(g_imgs, self.eval_size)
            g_imgs = (255 * (g_imgs.clamp(-1, 1) * 0.5 + 0.5)).to(torch.uint8)
            g_imgs = g_imgs.cpu()
            images.append(g_imgs)
        images = torch.cat(images, dim=0)
        fake_dataset = EvalFakeDataset(images)
        return fake_dataset
        

            


