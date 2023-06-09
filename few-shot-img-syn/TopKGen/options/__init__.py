
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument('--path', type=str, default='../../../data/table', help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--cuda', type=int, default=0, help='index of gpu to use')
    parser.add_argument('--seed', type=int, default=0, help='seed for init')
    parser.add_argument('--name', type=str, default='test1', help='experiment name')
    parser.add_argument('--iter', type=int, default=50000, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=1024, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='Default', help='Use restart to get retrain the model without init from the best model. checkpoint weight path if have one; if restart will retrain from beginning, default is to resume')

    parser.add_argument("--nonoise", default=False, action='store_true')
    
    # latent dimension
    parser.add_argument("--nz", type=int, default=256, help="limit nz to a small number")
    # backup scheduler and network perturbation
    parser.add_argument("--backup_scheduler_patience", type=int, default=10)
    parser.add_argument("--eps_D", type=float, default=0.,help="how much to perturb after each time rollback for D")
    parser.add_argument("--eps_G", type=float, default=0, help="how much to perturb after each time rollback for G")

    # add sparsity on activations
    parser.add_argument("--sparse_hw_info", type=str, default="None", help="sparse_hw_info: 16-32-64_10-10-20") # default is 1. in case -- need to be put in .sh
    parser.add_argument("--sp_hw_policy_name", type=str, default="IdentityBase")
     
    # sparse_loss weight
    parser.add_argument("--sparse_layer_loss_weight", type=float, default=1e-2, help="topk sparse layer loss weight")
    
    # evaluation encoders
    parser.add_argument("--fid_feature_extractor_name", default='inception-v3-compat', help="what encoder to use to compute FID")
    
    # eval
    parser.add_argument("--n_sample", type=int, default=3000)

    args = gather_and_process_args(parser)

    return args


def gather_and_process_args(parser):

    args = parser.parse_args() 
    print(args)
    print(f"nonoise {args.nonoise}")

    return args