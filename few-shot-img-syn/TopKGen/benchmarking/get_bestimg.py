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
from scipy.spatial.distance import cdist


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


def get_best(feat_real, feat_fake, top_percent=0.1):
    dist_mat = cdist(feat_fake, feat_real)
    low_value = np.quantile(dist_mat, top_percent, axis=0)
    fake_id = np.unique(np.where(dist_mat < low_value)[0])
    return fake_id

    

    



if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--path_a', type=str)
    parser.add_argument('--path_b', type=str)
    parser.add_argument('--iter', type=int, default=3)
    parser.add_argument('--select_topk_percent', type=float, default=0.1)

    args = parser.parse_args()

    inception = load_patched_inception_v3().eval().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize( (args.size, args.size) ),
            #transforms.RandomHorizontalFlip(p=0.5 if args.flip else 0),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dset_a = ImageFolder(args.path_a, transform=transform)
    loader_a = DataLoader(dset_a, batch_size=args.batch, num_workers=4)

    features_a = extract_features(loader_a, inception, device).numpy()
    print(f'extracted {features_a.shape[0]} features')

    real_mean = np.mean(features_a, 0)
    real_cov = np.cov(features_a, rowvar=False)
    
    #for folder in os.listdir(args.path_b):
    folder = args.iter
    folder = 'eval_%d'%(folder*10000)
    if os.path.exists(os.path.join( args.path_b, folder )):
        print(folder)
        dset_b = ImageFolder( os.path.join( args.path_b, folder, 'img' ), transform=transform)
        loader_b = DataLoader(dset_b, batch_size=args.batch, num_workers=4, shuffle=False)

        features_b = extract_features(loader_b, inception, device).numpy()
        print(f'extracted {features_b.shape[0]} features')

        select_fake_id = get_best(features_a, features_b, top_percent=args.select_topk_percent)
        dump_dir = os.path.join(args.path_b, folder, f'select_best_{args.select_topk_percent}')
        

        with open("select_best.sh", 'w') as f:
            f.write(f"mkdir {dump_dir}\n")
            for i in select_fake_id:
                path = dset_b.frame[i]
                f.write(f"cp {path} {dump_dir}\n")


                    

            


            
