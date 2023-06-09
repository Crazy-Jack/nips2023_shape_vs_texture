import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
torch.autograd.set_detect_anomaly(True)
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

import models
from operation import copy_G_params, load_params, get_dir, perturb_params
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment
import os 
from TopKGen.options import get_parser

policy = 'color,translation'
import lpips
percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)

import TopKGen.utlis.eval_agent_module as eval_agent


def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]

def train_d(net, data, label="real"):
    """Train function of discriminator"""
    if label=="real":
        pred, [rec_all, rec_small, rec_part], part = net(data, label)
        err = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean() + \
            percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
            percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() +\
            percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
        err.backward()
        return pred.mean().item(), rec_all, rec_small, rec_part
    else:
        pred = net(data, label)
        err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()
        return pred.mean().item()
    

def train(args):

    data_root = args.path
    total_iterations = args.iter
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    ndf = 64
    ngf = 64
    nz = args.nz
    nlr = 0.0002
    nbeta1 = 0.5
    use_cuda = True
    multi_gpu = False
    dataloader_workers = 8
    current_iteration = 0
    save_interval = 100
    saved_model_folder, saved_image_folder = get_dir(args)
    
    device = torch.device("cpu")
    if use_cuda:
        device = torch.device("cuda:0")

    transform_list = [
            transforms.Resize((int(im_size),int(im_size))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    trans = transforms.Compose(transform_list)
    
    
    dataset = ImageFolder(root=data_root, transform=trans)

    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers, pin_memory=True))
    
    
    #from model_s import Generator, Discriminator
    using_noise = not args.nonoise
    
    netG = models.Generator(args, ngf=ngf, nz=nz, im_size=im_size, noise=using_noise)
    netG.apply(models.weights_init)

    netD = models.Discriminator(args, ndf=ndf, im_size=im_size)
    netD.apply(models.weights_init)

    netG.to(device)
    netD.to(device)

    avg_param_G = copy_G_params(netG)

    fixed_noise = torch.FloatTensor(8, nz).normal_(0, 1).to(device)

    if multi_gpu:
        netG = nn.DataParallel(netG.cuda())
        netD = nn.DataParallel(netD.cuda())
    
    # evaler
    real_image_path = args.path
    log_path = saved_image_folder
    log_path = os.path.join(log_path, "log") 
    os.makedirs(log_path, exist_ok=True)
    
    save_eval_folder = os.path.join('train_results/', args.name, 'eval')
    dump_img_path = os.path.join(save_eval_folder, "dump_folder", "img")
    best_img_path = os.path.join(save_eval_folder, "best_result_folder", "img")
    sample_batch_size = args.batch_size
    n_sample = 3000
    eval_size = args.im_size
    z_type = 'normal'
    z_dim = args.nz
    version_name = args.name
    evalers_all = {}
    # extract fid_feature_extractor_name
    evalers_all_names = args.fid_feature_extractor_name.split("+")
    
    assert len(evalers_all_names) > 0
    primary_evaler_name = evalers_all_names[0]

    for eval_name_i in evalers_all_names:
        evalers_all[eval_name_i] = eval_agent.Evaler(real_image_path, log_path, version_name, sample_batch_size, n_sample, eval_size, z_dim, z_type, fid_feature_extractor_name=eval_name_i)
    
    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    
    best_ckpts = {}
    if checkpoint != 'restart':
        if checkpoint == 'Default':
            for evaler_name in evalers_all_names:
                best_default_ckpt = f"train_results/{args.name}/models/best_all_{evaler_name}.pth"
                if os.path.isfile(best_default_ckpt):
                    ckpt = torch.load(best_default_ckpt)

                    print(f"Load from {best_default_ckpt}")
                else:
                    ckpt = None
                best_ckpts[evaler_name] = ckpt
        else:
            load_ckpt_dir_path = checkpoint # pointing to the directory that contains the checkpoint
            for evaler_name in evalers_all_names:
                best_ckpt = os.path.join(load_ckpt_dir_path, f"best_all_{evaler_name}.pth")
                if os.path.isfile(best_ckpt):
                    ckpt = torch.load(best_ckpt)

                    print(f"Load from {best_ckpt}")
                else:
                    ckpt = None
                best_ckpts[evaler_name] = ckpt
        
        if best_ckpts[primary_evaler_name]:
            ckpt = best_ckpts[primary_evaler_name]
            netG.load_state_dict(ckpt['g'])
            netD.load_state_dict(ckpt['d'])
            avg_param_G = ckpt['g_ema']
            optimizerG.load_state_dict(ckpt['opt_g'])
            optimizerD.load_state_dict(ckpt['opt_d'])
            current_iteration = int(ckpt['iter']) + 1
            del ckpt
        
            # load the evaler in the loop 
            for evaler_name in evalers_all_names:
                if best_ckpts[evaler_name]:
                    evalers_all[evaler_name].best_results = best_ckpts[evaler_name]['performance']


    # setup backup scheduler
    bad_behavior_num = 0

    iteration = current_iteration
    while iteration <= total_iterations:
    
        real_image = next(dataloader)
        real_image = real_image.cuda(non_blocking=True)
        current_batch_size = real_image.size(0)
        noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(device)

        fake_images = netG(noise)
        ## 1. DiffAug
        real_image = DiffAugment(real_image, policy=policy)
        fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]
        ## 2. train Discriminator
        netD.zero_grad()

        err_dr, rec_img_all, rec_img_small, rec_img_part = train_d(netD, real_image, label="real")
        train_d(netD, [fi.detach() for fi in fake_images], label="fake")
        optimizerD.step()
        
        ## 3. train Generator
        netG.zero_grad()
        pred_g = netD(fake_images, "fake")
        err_g = -pred_g.mean()
        
        ## add loss for sparse_recon_loss
        if hasattr(netG, 'sparse_layer_loss') and len(netG.sparse_layer_loss) > 0:
            sparse_layer_loss = torch.nan_to_num(torch.cat(netG.sparse_layer_loss).mean())
            err_g += args.sparse_layer_loss_weight * sparse_layer_loss
            if iteration % 1 == 0:
                with torch.no_grad():
                    print(f"loss sparse layer loss: {[i.detach().item() for i in netG.sparse_layer_loss]}")
        
        err_g.backward()
        optimizerG.step()

        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        if iteration % 100 == 0:
            print(f"Iteration {iteration}")
            print("GAN: loss d: %.5f    loss g: %.5f ;"%(err_dr, -err_g.item()))

        if iteration % (save_interval*10) == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                vutils.save_image(netG(fixed_noise)[0].add(1).mul(0.5), saved_image_folder+'/%d.jpg'%iteration, nrow=4)
                vutils.save_image( torch.cat([
                        F.interpolate(real_image, 128), 
                        rec_img_all, rec_img_small,
                        rec_img_part]).add(1).mul(0.5), saved_image_folder+'/rec_%d.jpg'%iteration )
            save_gradient = False
            if save_gradient:
                netG.zero_grad()
                fake_images = netG(fixed_noise)
                pred_g = - netD(fake_images, "fake").mean()
                gradient = torch.autograd.grad(
                    # Note: You need to take the gradient of outputs with respect to inputs.
                    #### START CODE HERE ####
                    inputs = fake_images[0],
                    outputs = pred_g,
                    #### END CODE HERE ####
                    # These other parameters have to do with how the pytorch autograd engine works
                    # grad_outputs=torch.ones_like(fake_images[0]), 
                    create_graph=True,
                    retain_graph=False,
                )[0]
                # gradient = torch.tanh(gradient)
                gradient_f = gradient.flatten()
                plt.clf()
                plt.scatter(np.arange(len(gradient_f)), gradient_f.sort(descending=True)[0].detach().cpu().numpy())
                plt.savefig(saved_image_folder+'/%d_grad_stats.jpg'%iteration)
                gradient = gradient.abs() * 100
                gradient = (gradient - gradient.view(gradient.shape[0], -1).min(1)[0].view(gradient.shape[0], 1, 1, 1)) / (gradient.view(gradient.shape[0], -1).max(1)[0] - gradient.view(gradient.shape[0], -1).min(1)[0]).view(gradient.shape[0], 1, 1, 1)
                
                print(f"gradient max {gradient.max()}, gradient min {gradient.min()}, gradient {gradient.mean()}")
                beta = 0.8
                overlay_grad = torch.FloatTensor([beta]).cuda() * gradient + \
                                torch.FloatTensor([1 - beta]).cuda() * fake_images[0]
                vutils.save_image(overlay_grad, saved_image_folder+'/%d_grad.jpg'%iteration, nrow=4)
                
            load_params(netG, backup_para)



        store_intermediate = False
        if store_intermediate:
            if iteration % (save_interval*50) == 0 or iteration == total_iterations:
                backup_para = copy_G_params(netG)
                load_params(netG, avg_param_G)
                torch.save({'g':netG.state_dict(),'d':netD.state_dict()}, saved_model_folder+'/%d.pth'%iteration)
                load_params(netG, backup_para)
                torch.save({'g':netG.state_dict(),
                            'd':netD.state_dict(),
                            'g_ema': avg_param_G,
                            'opt_g': optimizerG.state_dict(),
                            'opt_d': optimizerD.state_dict()}, saved_model_folder+'/all_%d.pth'%iteration)


        # eval every 1000 epochs
        if ((iteration % 1000 == 0) and (iteration > 0)) or (iteration == total_iterations):
            with torch.no_grad():
                
                backup_para = copy_G_params(netG)
                load_params(netG, avg_param_G)
                # begin inv evaler
                for evaler_name in evalers_all:
                    evaler = evalers_all[evaler_name]
                    save_or_not, eval_results = evaler.eval(netG, iteration)

                    if save_or_not:
                        # update best results
                        torch.save({'g':netG.state_dict(),'d':netD.state_dict()}, saved_model_folder+'/best.pth')
                        load_params(netG, backup_para)
                        save_state_dict = {'iter': iteration,
                                    'performance': evaler.best_results,
                                    'g':netG.state_dict(),
                                    'd':netD.state_dict(),
                                    'g_ema': avg_param_G,
                                    'opt_g': optimizerG.state_dict(),
                                    'opt_d': optimizerD.state_dict(), 
                                    }
        
                        torch.save(save_state_dict, saved_model_folder+f'/best_all_{evaler_name}.pth')
                        # clear bad behavior
                        if evaler_name == primary_evaler_name:
                            bad_behavior_num = 0
                    else:
                        if evaler_name == primary_evaler_name:
                            bad_behavior_num += 1
                    
                load_params(netG, backup_para)

                    # consider parameter reset in G based on the history of performance 
                if bad_behavior_num >= args.backup_scheduler_patience:
                    evaler = evalers_all[primary_evaler_name] # chane to the primary evaler
                    # rollback the g/d parameter into the checkpoints of the best model
                    ckpt = torch.load(saved_model_folder+f'/best_all_{primary_evaler_name}.pth')
                    netG.load_state_dict(ckpt['g'])
                    netD.load_state_dict(ckpt['d'])
                    avg_param_G = ckpt['g_ema']
                    optimizerG.load_state_dict(ckpt['opt_g'])
                    optimizerD.load_state_dict(ckpt['opt_d'])
                    iteration = int(ckpt['iter'])
                    evaler.best_results = ckpt['performance']
                    

                    del ckpt
                    with open(evaler.log_path, 'a') as f:
                        f.write(f"------ Rollback to the recent best results from iter {iteration} based on {primary_evaler_name} performance check ... ------")
                    
                    # perturb D and G
                    try:
                        if args.eps_D > 0:
                            perturb_params(netD, eps=args.eps_D)
                        if args.eps_G > 0:
                            perturb_params(netG, eps=args.eps_G)
                        with open(evaler.log_path, 'a') as f:
                            f.write(f"------ D perturbed eps:{args.eps_D} and G perturbed eps:{args.eps_G}... ------\n")
                    
                    except Exception as e:
                        with open(evaler.log_path, 'a') as f:
                            f.write(f"------ Perturbation failed: {e}.. move on\n")
                    
                    bad_behavior_num = 0
        iteration += 1

if __name__ == "__main__":
    args = get_parser()
    torch.manual_seed(args.seed)
    train(args)