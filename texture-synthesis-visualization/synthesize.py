from typing import Optional
import argparse
import os

import torch
import matplotlib.pyplot as plt     # type: ignore

import utilities
import model
import optimize

def main(args: Optional[argparse.Namespace] = None):
    if args is None:
        args = parse_arguments()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for seed in range(args.num_texture_per_image):
        for img_path_i in os.listdir(args.img_path):
            # image name 
            img_name = img_path_i.split(".")[0]
            # load model & data
            target_image = utilities.preprocess_image(
                utilities.load_image(os.path.join(args.img_path, img_path_i))
            )

            print(f"...synthesizing for {img_name} seed {seed} ")
            net = model.Model(args.model_path, device, target_image, 
                                topk = args.topk, reverse_topk = args.reverse_topk)

            # synthesize
            optimizer = optimize.Optimizer(net, args, device, seed)
            result = optimizer.optimize()

            # save result
            final_image = utilities.postprocess_image(
                result, utilities.load_image(os.path.join(args.img_path, img_path_i))
            )
            final_image.save(os.path.join(args.out_dir, f'seed_{seed}_{img_name}.jpg'))

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-o',
        dest='out_dir',
        default='output',
        help=(
            'Output directory. All output will be save into a subdirectory '
            'within this directory which will be called after the input image '
            'and will be created if it does not exist yet.'
        )
    )

    parser.add_argument(
        '-i',
        dest='img_path',
        default='../few-shot-img-syn/data/jeep',
        help='Path to a directory of target images.'
    )

    parser.add_argument(
        '-m',
        dest='model_path',
        default='models/VGG19_normalized_avg_pool_pytorch',
        help='Path to the model file.'
    )

    parser.add_argument(
        "--samples",
        default=10,
        type=int,
        help='seed.'
    )

    parser.add_argument(
        '-n',
        dest='n_steps',
        type=int,
        default=100,
        help='The maximum number of optimizer steps to be performed.'
    )

    parser.add_argument(
        '--iter',
        dest='max_iter',
        type=int,
        default=20,
        help='The maximum number of iterations within one optimization step.'
    )

    parser.add_argument(
        '--lr',
        dest='lr',
        type=float,
        default=1.0,
        help='Optimizer learning rate.'
    )

    parser.add_argument(
        '--num_texture_per_image',
        type=int,
        default=1,
        help='number of synthesis texture per images'
    )

    parser.add_argument(
        '--topk',
        type=float,
        default=0.05,
        help='topk sparsity'
    )

    parser.add_argument(
        '--reverse_topk',
        action='store_true',
        help='number of synthesis texture per images'
    )

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    return args


if __name__ == '__main__':
    main()
