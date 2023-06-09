"""
Model for TopKGen
Dynamically install Top-K sparse layers inside generator
Layer installation is controlled via --sparse_hw_info, with the following format specifying the sparsity:
    --sparse_hw_info [sparselayer1resolution-sparselayer2resolution-sparselayer3resolution_sparselayer1sparsity-sparselayer2sparsity-sparselayer3sparsity]
    e.g. to install 5% sparsity on 32x32 as used in the paper experiment, use --sparse_hw_info 32_5
         to install 5% sparsity on 32x32 and 16x16 layer, use --sparse_hw_info 16-32_5-5
         to install 5% sparsity on 32x32 and 10% sparsity on 16x16 layer, use --sparse_hw_info 16-32_10-5

Adapted from https://github.com/odegeasslbc/FastGAN-pytorch/blob/main/models.py
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

import random

import torchvision.models as vision_models
from TopKGen.TopkLoss.topk_loss_module import find_topk_operation_using_name
seq = nn.Sequential

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def conv2d(*args, **kwargs):
    return spectral_norm(nn.Conv2d(*args, **kwargs))

def convTranspose2d(*args, **kwargs):
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))

def batchNorm2d(*args, **kwargs):
    return nn.BatchNorm2d(*args, **kwargs)

def linear(*args, **kwargs):
    return spectral_norm(nn.Linear(*args, **kwargs))

class PixelNorm(nn.Module):
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.target_shape = shape

    def forward(self, feat):
        batch = feat.shape[0]
        return feat.view(batch, *self.target_shape)


class GLU(nn.Module):
    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        if False:
            if noise is None:
                batch, _, height, width = feat.shape
                noise = torch.randn(batch, 1, height, width).to(feat.device)

            return feat + self.weight * noise
        else:
            return feat


class Swish(nn.Module):
    def forward(self, feat):
        return feat * torch.sigmoid(feat)


class SEBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()

        self.main = nn.ModuleList([ nn.AdaptiveAvgPool2d(4),
                                    conv2d(ch_in, ch_out, 4, 1, 0, bias=False), Swish(),
                                    conv2d(ch_out, ch_out, 1, 1, 0, bias=False), nn.Sigmoid() ])
        
    def forward(self, feat_small, feat_big, v=False):
        
        x = self.main[0](feat_small)
        x = self.main[1](x)
        x = self.main[2](x)
        x = self.main[3](x)

        return feat_big * x


class InitLayer(nn.Module):
    def __init__(self, nz, channel):
        super().__init__()

        self.init = nn.Sequential(
                        convTranspose2d(nz, channel*2, 4, 1, 0, bias=False),
                        batchNorm2d(channel*2), GLU() )

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)


def UpBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        batchNorm2d(out_planes*2), GLU())
    return block


class UpblockComp_control(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(UpblockComp_control, self).__init__()
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False)
        self.noise1 = NoiseInjection()
        self.bn1 = batchNorm2d(out_planes*2)
        self.glu1 = GLU()
        self.conv2 = conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False)
        self.noise2 = NoiseInjection()
        self.bn2 = batchNorm2d(out_planes*2)
        self.glu2 = GLU()

    def forward(self, x, noise=False):
        x = self.up1(x)
        x = self.conv1(x)
        if noise:
            x = self.noise1(x)
        x = self.bn1(x)
        x = self.glu1(x)
        x = self.conv2(x)
        if noise:
            x = self.noise2(x)
        x = self.bn2(x)
        x = self.glu2(x)

        return x


def UpBlockComp(in_planes, out_planes, noise=True):
    if noise:
        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
            NoiseInjection(),
            batchNorm2d(out_planes*2), GLU(),
            conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False),
            NoiseInjection(),
            batchNorm2d(out_planes*2), GLU()
        )
    else:
        block = UpblockComp_control(in_planes, out_planes)
    return block



class Generator(nn.Module):
    def __init__(self, args, ngf=64, nz=100, nc=3, im_size=1024, noise=True):
        super(Generator, self).__init__()
        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ngf)

        self.im_size = im_size

        self.init = InitLayer(nz, channel=nfc[4])
        print(f"Use noise in generator: {noise}")
        self.feat_8   = UpBlockComp(nfc[4], nfc[8], noise=noise)
        self.feat_16  = UpBlock(nfc[8], nfc[16])
        self.feat_32  = UpBlockComp(nfc[16], nfc[32], noise=noise)
        self.feat_64  = UpBlock(nfc[32], nfc[64])

        self.feat_128 = UpBlockComp(nfc[64], nfc[128], noise=noise)
        self.feat_256 = UpBlock(nfc[128], nfc[256])

        # sparse layers
        if args.sparse_hw_info != "None":
            self.sparse_hw_reso, self.sparse_hw_topk = args.sparse_hw_info.split("_")
            self.sparse_hw_reso = [int(i) for i in self.sparse_hw_reso.split("-")]
            self.sparse_hw_topk = [int(i) * 0.01 for i in self.sparse_hw_topk.split("-")]
            self.sparse_hw_topk_info = {self.sparse_hw_reso[i]: self.sparse_hw_topk[i] for i in range(len(self.sparse_hw_reso))}
            for reso in self.sparse_hw_reso:
                topk_keep_num = max(int(self.sparse_hw_topk_info[reso] * reso * reso), 1)
                setattr(self, f"sparse_hw_layer_{reso}", find_topk_operation_using_name(args.sp_hw_policy_name)(self.sparse_hw_topk_info[reso], topk_keep_num, reso * reso, args, reso))
        else:
            self.sparse_hw_topk_info = "None"

        self.sparse_layer_loss = []   

        # SEBlocks
        self.se_64  = SEBlock(nfc[4], nfc[64])
        self.se_128 = SEBlock(nfc[8], nfc[128])
        self.se_256 = SEBlock(nfc[16], nfc[256])

        self.to_128 = conv2d(nfc[128], nc, 1, 1, 0, bias=False)
        self.to_big = conv2d(nfc[im_size], nc, 3, 1, 1, bias=False)

        if im_size > 256:
            self.feat_512 = UpBlockComp(nfc[256], nfc[512], noise=noise)
            self.se_512 = SEBlock(nfc[32], nfc[512])
        if im_size > 512:
            self.feat_1024 = UpBlock(nfc[512], nfc[1024])
        
    def sparse_layer(self, reso, x):
        tau = 1.

        if self.sparse_hw_topk_info != "None" and reso in self.sparse_hw_topk_info:
            sparse_layer_reso = getattr(self, f"sparse_hw_layer_{reso}")
            x, loss = sparse_layer_reso(x, tau = tau)
            if torch.is_tensor(loss):
                self.sparse_layer_loss.append(loss.unsqueeze(0))
        return x



    def forward(self, input, output_attention_layer=False, inject_noise=0, evaluation=False):
        self.sparse_layer_loss = []

        # print("This fo")
        feat_4   = self.init(input)
        
        feat_4 = self.sparse_layer(4, feat_4)

        feat_8   = self.feat_8(feat_4)
        
        feat_8 = self.sparse_layer(8, feat_8)

        feat_16  = self.feat_16(feat_8)
        
        feat_16 = self.sparse_layer(16, feat_16)

        feat_32  = self.feat_32(feat_16)
        
        feat_32 = self.sparse_layer(32, feat_32)

        feat_64  = self.se_64( feat_4, self.feat_64(feat_32))

        if inject_noise:
            eval_time_noise = torch.rand(feat_64.shape).cuda() * inject_noise
            feat_64 += eval_time_noise

        feat_64 = self.sparse_layer(64, feat_64)

        feat_128 = self.se_128( feat_8, self.feat_128(feat_64) )

        feat_128 = self.sparse_layer(128, feat_128)


        feat_256 = self.se_256( feat_16, self.feat_256(feat_128) )
        
        feat_256 = self.sparse_layer(256, feat_256)


        if self.im_size == 256:
            return [self.to_big(feat_256), self.to_128(feat_128)]

        feat_512 = self.se_512( feat_32, self.feat_512(feat_256) )

        feat_512 = self.sparse_layer(512, feat_512)

        if self.im_size == 512:
            return [self.to_big(feat_512), self.to_128(feat_128)]

        feat_1024 = self.feat_1024(feat_512)

        feat_1024 = self.sparse_layer(1024, feat_1024)

        im_128 = torch.tanh(self.to_128(feat_128))
        im_1024 = torch.tanh(self.to_big(feat_1024))

        return [im_1024, im_128]


class DownBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlock, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, feat):
        return self.main(feat)


class DownBlockComp(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DownBlockComp, self).__init__()

        self.main = nn.Sequential(
            conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2, inplace=True),
            conv2d(out_planes, out_planes, 3, 1, 1, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2)
            )

        self.direct = nn.Sequential(
            nn.AvgPool2d(2, 2),
            conv2d(in_planes, out_planes, 1, 1, 0, bias=False),
            batchNorm2d(out_planes), nn.LeakyReLU(0.2))

    def forward(self, feat):
        return (self.main(feat) + self.direct(feat)) / 2


class Discriminator(nn.Module):
    def __init__(self, args, ndf=64, nc=3, im_size=512):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.im_size = im_size

        nfc_multi = {4:16, 8:16, 16:8, 32:4, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ndf)

        if im_size == 1024:
            self.down_from_big = nn.Sequential(
                                    conv2d(nc, nfc[1024], 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True),
                                    conv2d(nfc[1024], nfc[512], 4, 2, 1, bias=False),
                                    batchNorm2d(nfc[512]),
                                    nn.LeakyReLU(0.2, inplace=True))
        elif im_size == 512:
            self.down_from_big = nn.Sequential(
                                    conv2d(nc, nfc[512], 4, 2, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True) )
        elif im_size == 256:
            self.down_from_big = nn.Sequential(
                                    conv2d(nc, nfc[512], 3, 1, 1, bias=False),
                                    nn.LeakyReLU(0.2, inplace=True) )

        self.down_4  = DownBlockComp(nfc[512], nfc[256])
        self.down_8  = DownBlockComp(nfc[256], nfc[128])
        self.down_16 = DownBlockComp(nfc[128], nfc[64])
        self.down_32 = DownBlockComp(nfc[64],  nfc[32])
        self.down_64 = DownBlockComp(nfc[32],  nfc[16])

        self.rf_big = nn.Sequential(
                            conv2d(nfc[16] , nfc[8], 1, 1, 0, bias=False),
                            batchNorm2d(nfc[8]), nn.LeakyReLU(0.2, inplace=True),
                            conv2d(nfc[8], 1, 4, 1, 0, bias=False))

        self.se_2_16 = SEBlock(nfc[512], nfc[64])
        self.se_4_32 = SEBlock(nfc[256], nfc[32])
        self.se_8_64 = SEBlock(nfc[128], nfc[16])

        self.down_from_small = nn.Sequential(
                                            conv2d(nc, nfc[256], 4, 2, 1, bias=False),
                                            nn.LeakyReLU(0.2, inplace=True),
                                            DownBlock(nfc[256],  nfc[128]),
                                            DownBlock(nfc[128],  nfc[64]),
                                            DownBlock(nfc[64],  nfc[32]), )

        self.rf_small = conv2d(nfc[32], 1, 4, 1, 0, bias=False)

        self.decoder_big = SimpleDecoder(nfc[16], nc)
        self.decoder_part = SimpleDecoder(nfc[32], nc)
        self.decoder_small = SimpleDecoder(nfc[32], nc)


    def forward(self, imgs, label, evaluation=False):
        if type(imgs) is not list:
            imgs = [F.interpolate(imgs, size=self.im_size), F.interpolate(imgs, size=128)]

        feat_2 = self.down_from_big(imgs[0])     # bz, 16, 256, 256
        
        feat_4 = self.down_4(feat_2)             # bz, 32, 128, 128
        
        feat_8 = self.down_8(feat_4)             # bz, 64, 64, 64

        feat_16 = self.down_16(feat_8)           # bz, 128, 32, 32
        
        feat_16 = self.se_2_16(feat_2, feat_16)

        feat_32 = self.down_32(feat_16)          # bz, 256, 16, 16
        
        feat_32 = self.se_4_32(feat_4, feat_32)

        feat_last = self.down_64(feat_32)        # bz, 512, 8, 8

        feat_last = self.se_8_64(feat_8, feat_last)

        rf_0 = self.rf_big(feat_last)

        feat_small = self.down_from_small(imgs[1])
        
        rf_1 = self.rf_small(feat_small)

        if label=='real':
            rec_img_big = self.decoder_big(feat_last)
            rec_img_small = self.decoder_small(feat_small)

            part = random.randint(0, 3)
            rec_img_part = None
            if part==0:
                rec_img_part = self.decoder_part(feat_32[:,:,:8,:8])
            if part==1:
                rec_img_part = self.decoder_part(feat_32[:,:,:8,8:])
            if part==2:
                rec_img_part = self.decoder_part(feat_32[:,:,8:,:8])
            if part==3:
                rec_img_part = self.decoder_part(feat_32[:,:,8:,8:])

            return torch.cat([rf_0, rf_1]) , [rec_img_big, rec_img_small, rec_img_part], part

        return torch.cat([rf_0, rf_1])


class SimpleDecoder(nn.Module):
    """docstring for CAN_SimpleDecoder"""
    def __init__(self, nfc_in=64, nc=3):
        super(SimpleDecoder, self).__init__()

        nfc_multi = {4:16, 8:8, 16:4, 32:2, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*32)

        def upBlock(in_planes, out_planes):
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
                batchNorm2d(out_planes*2), GLU())
            return block

        self.main = nn.Sequential(  nn.AdaptiveAvgPool2d(8),
                                    upBlock(nfc_in, nfc[16]) ,
                                    upBlock(nfc[16], nfc[32]),
                                    upBlock(nfc[32], nfc[64]),
                                    upBlock(nfc[64], nfc[128]),
                                    conv2d(nfc[128], nc, 3, 1, 1, bias=False),
                                    nn.Tanh() )

    def forward(self, input):
        # input shape: c x 4 x 4
        return self.main(input)

from random import randint
def random_crop(image, size):
    h, w = image.shape[2:]
    ch = randint(0, h-size-1)
    cw = randint(0, w-size-1)
    return image[:,:,ch:ch+size,cw:cw+size]

class TextureDiscriminator(nn.Module):
    def __init__(self, ndf=64, nc=3, im_size=512):
        super(TextureDiscriminator, self).__init__()
        self.ndf = ndf
        self.im_size = im_size

        nfc_multi = {4:16, 8:8, 16:8, 32:4, 64:2, 128:1, 256:0.5, 512:0.25, 1024:0.125}
        nfc = {}
        for k, v in nfc_multi.items():
            nfc[k] = int(v*ndf)

        self.down_from_small = nn.Sequential(
                                            conv2d(nc, nfc[256], 4, 2, 1, bias=False),
                                            nn.LeakyReLU(0.2, inplace=True),
                                            DownBlock(nfc[256],  nfc[128]),
                                            DownBlock(nfc[128],  nfc[64]),
                                            DownBlock(nfc[64],  nfc[32]), )
        self.rf_small = nn.Sequential(
                            conv2d(nfc[16], 1, 4, 1, 0, bias=False))

        self.decoder_small = SimpleDecoder(nfc[32], nc)

    def forward(self, img, label):
        img = random_crop(img, size=128)

        feat_small = self.down_from_small(img)
        rf = self.rf_small(feat_small).view(-1)

        if label=='real':
            rec_img_small = self.decoder_small(feat_small)

            return rf, rec_img_small, img

        return rf
