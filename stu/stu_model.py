from email.mime import base
import torch
import torch.nn as nn
from einops import rearrange
import scipy.io as sio
import numpy as np
import torch.nn.functional as F
import torch.autograd as autograd
from timm.models.layers import trunc_normal_, DropPath
from math import exp
from torch.autograd import Variable
import os
from GWFE import GWFE
import argparse
from dataset import Get_SDataset, Get_MEF_Dataset_RGB,Get_Test_Dataset
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from PIL import Image
import glob
from feat_fuse import CAFM
from utils import RGB2YCrCb, YCbCr2RGB
from losses_kd import Fusion_loss, calculate_cosine_similarity_loss
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
from torchvision import transforms

class Former(nn.Module):
    def __init__(self, base_filter):
        super(Former, self).__init__()
        self.gwfe = GWFE(base_filter,base_filter)

    def forward(self, I):
        return self.gwfe(I)

class Encoder(nn.Module):
    def __init__(self, num_channel, base_filter):
        super(Encoder, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.pool = torch.nn.MaxPool2d(3,stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.eformer1 = Former(base_filter)
        self.eformer2 = Former(base_filter)
        self.eformer4 = Former(base_filter)
        self.eformer8 = Former(base_filter)
        self.in_conv = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(num_channel, base_filter, 3, 1, 0, bias=False))
    def forward(self, I):
        I = self.in_conv(I)
        I = self.eformer1(I)
        I2 = self.pool(self.pad(I))
        I2 = self.eformer2(I2)
        I4 = self.pool(self.pad(I2))
        I4 = self.eformer4(I4)
        I8 = self.pool(self.pad(I4))
        I8 = self.eformer8(I8)
        Ires = I - self.up(I2)
        I2res = I2 - self.up(I4) 
        I4res = I4 - self.up(I8) 
        return Ires,I2res,I4res,I8

class Decoder(nn.Module):
    def __init__(self, num_channel, base_filter):
        super(Decoder, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dformer1 = Former(base_filter)
        self.dformer2 = Former(base_filter)
        self.dformer4 = Former(base_filter)
        self.dformer8 = Former(base_filter)
        self.in_conv = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(num_channel, base_filter, 3, 1, 0, bias=False))
        self.out_conv = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(base_filter, num_channel, 3, 1, 0, bias=False))
    def forward(self, Ires,I2res,I4res,I8):
        I8_d = self.dformer8(I8)
        I4 = self.dformer4(I4res + self.up(I8_d))
        I2 = self.dformer2(I2res + self.up((I4)))
        I1 = self.dformer1(Ires + self.up((I2)))
        I = self.out_conv(I1)
        return I, I8_d, I4, I2, I1
    
class Stu_Decoder(nn.Module):
    def __init__(self, num_channel, base_filter):
        super(Stu_Decoder, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dformer1 = Former(base_filter)
        self.dformer2 = Former(base_filter)
        self.dformer4 = Former(base_filter)
        self.dformer8 = Former(base_filter)

        self.in_conv = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(num_channel, base_filter, 3, 1, 0, bias=False))
        self.out_conv = nn.Sequential(nn.ReflectionPad2d(1),
                                     nn.Conv2d(base_filter, num_channel, 3, 1, 0, bias=False))

    def forward(self, Ires,I2res,I4res,I8, tea_dec_feat, rate):
        if tea_dec_feat is not None:
            I8_d = self.dformer8(I8)
            I4 = self.dformer4(I4res + self.up(I8_d + tea_dec_feat[0] * rate))
            I2 = self.dformer2(I2res + self.up(I4 + tea_dec_feat[1] * rate))
            I1 = self.dformer1(Ires + self.up(I2 + tea_dec_feat[2] * rate))
            I = self.out_conv(I1 + tea_dec_feat[3] * rate)
        else:
            I8_d = self.dformer8(I8)
            I4 = self.dformer4(I4res + self.up(I8_d))
            I2 = self.dformer2(I2res + self.up((I4)))
            I1 = self.dformer1(Ires + self.up((I2)))
            I = self.out_conv(I1)
        return I, I8_d, I4, I2, I1

class Laplace_Pyramid_Former(nn.Module):
    def __init__(self, num_channel, base_filter):
        super(Laplace_Pyramid_Former, self).__init__()
        self.encoder = Encoder(num_channel,base_filter)
        self.decoder = Decoder(num_channel,base_filter)
        
    def forward(self, I1):
        I1res,I1res2,I1res4,I18 = self.encoder(I1)
        rI1, I8_d, I4, I2, I1_d = self.decoder(I1res,I1res2,I1res4,I18)
        return rI1, [I1res, I1res2, I1res4, I18], [I8_d, I4, I2, I1_d]
    

class Learned_Fuse(nn.Module):
    def __init__(self, base_filter):
        super(Learned_Fuse, self).__init__()
        self.fl_former = CAFM(base_filter, base_filter)
        self.fh_former = CAFM(base_filter, base_filter)
        self.fh2_former = CAFM(base_filter, base_filter)
        self.fh4_former = CAFM(base_filter, base_filter)
    def forward(self, I1res,I1res2,I1res4,I18,I2res,I2res2,I2res4,I28):
        I8 = self.fl_former(I18,I28)
        Ires = self.fh_former(I1res,I2res)
        Ires2  = self.fh2_former(I1res2,I2res2)
        Ires4 = self.fh4_former(I1res4,I2res4)
        return Ires,Ires2,Ires4,I8

class stu_Laplace_Pyramid_Former(nn.Module):
    def __init__(self, num_channel, base_filter):
        super(stu_Laplace_Pyramid_Former, self).__init__()
        self.encoder_vi = Encoder(num_channel,base_filter)
        self.encoder_ir = Encoder(num_channel,base_filter)
        self.adpt_fuse = Learned_Fuse(base_filter)
        self.decoder = Stu_Decoder(num_channel,base_filter)
        
    def forward(self, vi, ir, tea_enc_feat, tea_dec_feat, rate):
        I1res,I1res2,I1res4,I18 = self.encoder_vi(vi)
        I2res,I2res2,I2res4,I28 = self.encoder_ir(ir)
        fuse1,fuse2,fuse4,fuse8 = self.adpt_fuse(I1res,I1res2,I1res4,I18,I2res,I2res2,I2res4,I28)
        fuse1 = fuse1 + tea_enc_feat[0] * rate
        fuse2 = fuse2 + tea_enc_feat[1] * rate
        fuse4 = fuse4 + tea_enc_feat[2] * rate
        fuse8 = fuse8 + tea_enc_feat[3] * rate
        rI1, I8_d, I4, I2, I1_d = self.decoder(fuse1,fuse2,fuse4,fuse8, tea_dec_feat, rate)
        return rI1,[fuse1,fuse2,fuse4,fuse8], [I8_d, I4, I2, I1_d]
    
    def test(self, vi, ir):
        I1res,I1res2,I1res4,I18 = self.encoder_vi(vi)
        I2res,I2res2,I2res4,I28 = self.encoder_ir(ir)
        fuse1,fuse2,fuse4,fuse8 = self.adpt_fuse(I1res,I1res2,I1res4,I18,I2res,I2res2,I2res4,I28)
        rI1, _, _, _, _ = self.decoder(fuse1,fuse2,fuse4,fuse8, None, 0)
        return rI1

device = torch.device('cuda:0')
l1_loss = torch.nn.L1Loss().to(device)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='stu_pyramid_model', help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--ema_decay', default=0.999, type=float)
    parser.add_argument('--use_ema', action='store_true', default=True, help='use EMA model')
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--gamma', default=0.5, type=int)
    parser.add_argument('--sbatch_size', default=8, type=int)
    parser.add_argument('--total_epochs', default=10, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float)
    parser.add_argument('--weight', default=[1,1,0.0001, 0.0002], type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)
    args = parser.parse_args()
    return args

from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

def YCbCr2RGB(Y, Cb, Cr):
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0,1.0)
    return out

import math
def cosin_decay(epoch, total_epochs):
    if total_epochs <= 1:
        return 0.0
    progress = min((epoch + 1) / (total_epochs), 1.0)   
    rate = 0.5 * (1 + math.cos(math.pi * progress))
    if epoch == total_epochs - 1:
        rate = 0.0
    return rate

def train(args, train_loader_ir, model_tea, model_stu, optimizer, epoch):
    model_stu.train()
    model_tea.eval()
    rate = cosin_decay(epoch, args.total_epochs) 

    print("rate: ", rate)
    for i, (batch_s) in tqdm(enumerate(zip(train_loader_ir))):
        ir_img = Variable(batch_s[0][0])
        ir_img = ir_img.to(device)
        vi_img = Variable(batch_s[0][1])
        vi_img = vi_img.to(device)
        Y_vi, _, _ = RGB2YCrCb(vi_img)
        tea_img = Variable(batch_s[0][2])
        tea_img = tea_img.to(device)
        optimizer.zero_grad()
        with torch.no_grad(): 
            _, tea_enc_feat, tea_dec_feat = model_tea(tea_img)
        stu_fused_img, stu_fuse_feat, stu_dec_feat = model_stu(Y_vi, ir_img, tea_enc_feat, tea_dec_feat, rate)
        fuse_loss, _, _ =  Fusion_loss(vi_img, ir_img, stu_fused_img,device=device)
        kd_gt_loss = l1_loss(stu_fused_img, tea_img)
        kd_feat_loss = (calculate_cosine_similarity_loss(stu_fuse_feat, tea_enc_feat) + calculate_cosine_similarity_loss(stu_dec_feat, tea_dec_feat)) / 2
        total_loss = fuse_loss + kd_gt_loss * 100 + kd_feat_loss * 100
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model_stu.parameters(), max_norm=1e-4, norm_type=2)
        optimizer.step()
        model_stu.zero_grad()
        print("fusion_Loss: {:.2e},kd_gt_loss: {:.2e},kd_feat_loss: {:.2e}".format(fuse_loss.item(), kd_gt_loss.item() * 100, kd_feat_loss.item() * 100))

def main():
    args = parse_args()
 
    print('Config -----')
    cudnn.benchmark = True

    train_dir_ir = "./images/ir/train/" # forground
    train_dir_vi = "./images/vi/train/" # forground
    train_dir_tea_img = "../tea/train_img/"
    train_name_list = os.listdir(train_dir_ir)

    transform_train = transforms.Compose([transforms.ToTensor()])

    dataset_train_ir = Get_MEF_Dataset_RGB(train_dir_ir, train_dir_vi, train_dir_tea_img, train_name_list,
                                    transform=transform_train)

    train_loader_ir = DataLoader(dataset_train_ir,
                              shuffle=True,
                              num_workers=6,
                              batch_size=args.sbatch_size)
    
    model_tea = Laplace_Pyramid_Former(num_channel=1, base_filter=16).to(device)
    model_tea.load_state_dict(torch.load('../tea/models/tea_model.pth'))
    
    model_stu = stu_Laplace_Pyramid_Former(num_channel=1, base_filter=16).to(device)

    milestones = []
    for i in range(1, args.epochs+1):
        if i == 100:
            milestones.append(i)
        if i == 200:
            milestones.append(i)
    
    optimizer = optim.Adam(model_stu.parameters(), lr=args.lr,
                           betas=args.betas, eps=args.eps)
    scheduler_f = lrs.MultiStepLR(optimizer, milestones, args.gamma)

    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch+1, args.epochs))
        model_stu.zero_grad()
        train(args,train_loader_ir, model_tea,model_stu,optimizer,epoch)
        ckt['a'] = ckt['a'] + 2e-4
        scheduler_f.step()
        if (epoch+1) % 10 == 0:
            torch.save(model_stu.state_dict(), 'models/model_{}.pth'.format(epoch+1))

def tensor2img(img):
  img = img.cpu().float().numpy()
  if img.shape[0] == 1:
    img = np.tile(img, (3, 1, 1))
  img = (img - img.min()) / (img.max() - img.min())
  img = np.transpose(img, (1, 2, 0))  * 255.0
  return img.astype(np.uint8)

def save_img_single(img, name, width=None, height=None):
  img = tensor2img(img)
  img = Image.fromarray(img)
  if not width==None and not height==None:
    img = img.resize((width , height))
  img.save(name)


def test_model():
    test_dir_ir = "./images/ir/test/" # forground
    test_dir_vi = "./images/vi/test/" # forground
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          ])
    dataset_train_ir = Get_Test_Dataset(test_dir_ir, test_dir_vi,
                                    transform=transform_train)
    test_loader_ir = DataLoader(dataset_train_ir,
                              shuffle=True,
                              batch_size=1)
    model_stu = stu_Laplace_Pyramid_Former(num_channel=1, base_filter=16).to(device)
    model_stu.load_state_dict(torch.load("models/stu_model.pth"))
    model_stu.eval()
    save_folder_path = './fused_imgs/MSRS/'
    for i, (batch_s) in tqdm(enumerate(zip(test_loader_ir))):
        ir_img = Variable(batch_s[0][0])
        ir_img = ir_img.to(device)
        vi_img = Variable(batch_s[0][1])
        vi_img = vi_img.to(device)
        Y_vi, Cb_vi, Cr_vi = RGB2YCrCb(vi_img)
        h, w = Y_vi.shape[2], Y_vi.shape[3]
        if h % 16 == 0 and w % 16 == 0:
            with torch.no_grad():
                stu_fused_img  = model_stu.test(Y_vi, ir_img)
                stu_fused_img = torch.clamp(stu_fused_img, min=0.0, max=1.0)
                save_img_single(stu_fused_img[0,::], save_folder_path + batch_s[0][2][0], Y_vi.shape[-1], Y_vi.shape[-2])
        else:
            new_h = h
            new_w = w
            if h % 16 !=  0:
                new_h = ((h + 15) // 16) * 16
            if w % 16 != 0:
                new_w = ((w + 15) // 16) * 16
            with torch.no_grad():
                resize = transforms.Resize((new_h, new_w), interpolation=transforms.InterpolationMode.BILINEAR)
                Y_vi_resized = resize(Y_vi)
                ir_resized = resize(ir_img)
                stu_fused_img = model_stu.test(Y_vi_resized, ir_resized)
                stu_fused_img = F.interpolate(stu_fused_img, size=(h,w), mode='bilinear', align_corners=False)
                stu_fused_img = torch.clamp(stu_fused_img, min=0.0, max=1.0)
                save_img_single(stu_fused_img[0,::], 
                            save_folder_path + batch_s[0][2][0], 
                            w, h)

if __name__ == '__main__':
    ckt = {'epoch':0, 'a':0.0} 
    i_iter = 0
    test_model()