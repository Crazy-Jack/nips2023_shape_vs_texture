
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import numpy as np 
from models.resnet_imgnette import *
from utils import progress_bar
from datetime import datetime

parser = argparse.ArgumentParser(description='PyTorch ImageNette Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--pretrained', type=str, default="none",
                    help='resume from checkpoint')
parser.add_argument("--name", type=str)
parser.add_argument("--topk", type=str, default="")
parser.add_argument("--patch_size", type=int, default=8)
parser.add_argument("--data_folder_path", type=str, required=True)
parser.add_argument("--total_epochs", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--use_sin_val_folder", type=str, default="l_a0.5")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')

# the magic normalization parameters come from the example
transform_mean = np.array([ 0.485, 0.456, 0.406 ])
transform_std = np.array([ 0.229, 0.224, 0.225 ])

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = transform_mean, std = transform_std),
])


transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = transform_mean, std = transform_std),
])


trainset = torchvision.datasets.ImageFolder(
    root=os.path.join(args.data_folder_path, 'train'), transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

testset = torchvision.datasets.ImageFolder(
    root=os.path.join(args.data_folder_path, 'val'), transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=args.batch_size, shuffle=False, num_workers=4)


testset_sin = torchvision.datasets.ImageFolder(
    root=args.use_sin_val_folder, transform=transform_test)
testloader_sin = torch.utils.data.DataLoader(
    testset_sin, batch_size=args.batch_size, shuffle=False, num_workers=4)

# Model
print('==> Building model..')

if not args.topk:
    print("Use original no topk")
    net = ResNet18()
else:
    topk_level = [int(i) for i in args.topk.split("_")[0].split("-")]
    topk_sparsity = float(args.topk.split("_")[1]) * 0.01
    net = ResNet18_topk(topk_level, topk_sparsity)
    print(f"use topk with level {topk_level} and sparsity {topk_sparsity}")
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(f'./checkpoint/{args.name}'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'./checkpoint/{args.name}/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

elif args.pretrained != 'none':
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(f'{args.pretrained}')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_epochs)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    all_train_loss.append(train_loss / total * 1.)

def test(epoch, testloader, save_title, best_acc):
    print("Evaluating...")
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
            
        torch.save(state, f'./checkpoint/{args.name}/{save_title}ckpt.pth')
        best_acc = acc
    return best_acc, acc

best_acc = 0
best_sin_acc = 0
all_train_loss = []
accs = []
sin_accs = []
os.makedirs('checkpoint', exist_ok=True)
if not os.path.isdir(f'checkpoint/{args.name}'):
    os.mkdir(f'checkpoint/{args.name}')
with open(f'./checkpoint/{args.name}/log.txt', 'a') as f:
    f.write(f"\n---- Start New Training: {datetime.now()} ----\n")
    for arg in vars(args):
        f.write(f"{arg}: {getattr(args, arg)}\n")

for epoch in range(start_epoch, start_epoch+args.total_epochs):
    train(epoch)
    scheduler.step()
    best_acc, normal_acc = test(epoch, testloader, '', best_acc)
    accs.append(normal_acc)

    best_sin_acc, normal_sin_acc = test(epoch, testloader_sin, '', best_sin_acc)
    sin_accs.append(normal_sin_acc)
    np.save(f'./checkpoint/{args.name}/eval.npy', np.array(accs))
    np.save(f'./checkpoint/{args.name}/eval_sin.npy', np.array(sin_accs))
    np.save(f'./checkpoint/{args.name}/all_train_loss.npy', np.array(all_train_loss))

    with open(f'./checkpoint/{args.name}/log.txt', 'a') as f:
        f.write(f"{datetime.now()} | epoch {epoch} | best_acc {best_acc} | current acc {normal_acc} | best sin acc {best_sin_acc} | sin acc {normal_sin_acc} | loss {all_train_loss[-1]}\n")