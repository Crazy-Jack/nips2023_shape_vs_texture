import argparse
import pickle

import torch
from torch import nn
import numpy as np
from scipy import linalg
from tqdm import tqdm

from torchvision import transforms

from torch.utils.data import DataLoader

from calc_inception import load_patched_inception_v3
from operation import ImageFolder
import os
import torch_fidelity



from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True



@torch.no_grad()
def extract_features(loader, inception, device):
    pbar = tqdm(loader)

    feature_list = []

    for img in pbar:
        img = img.to(device)
        feature = inception(img)[0].view(img.shape[0], -1)
        feature_list.append(feature.to('cpu'))
    # print(feature_list[0])
    features = torch.cat(feature_list, 0)

    return features


def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--path_a', type=str)
    parser.add_argument('--path_b', type=str)

    args = parser.parse_args()



    path_b = os.path.join(args.path_b, 'img')

    transform = transforms.Compose(
        [
            transforms.Resize( (args.size, args.size) ),
            transforms.ToTensor(),
            lambda x: (255 * (x.clamp(-1, 1) * 0.5 + 0.5)).to(torch.uint8),
        ]
    )


    dset_a = ImageFolder(args.path_a, transform=transform)
  
    dset_b = ImageFolder(path_b, transform=transform)
    metrics = torch_fidelity.calculate_metrics(
        input1=dset_b,
        input2=dset_a,
        fid=True,
        kid=True,
        kid_subset_size=100,
    )
    

    

