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
import argparse
from dataset import Get_SDataset, Get_MEF_Dataset_RGB,Get_Test_Dataset
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from PIL import Image
import glob
from utils import RGB2YCrCb, YCbCr2RGB
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
from torchvision import transforms
from WTConv import WTConv2d


def Fusion_loss(vi, ir, fu, weights=[10, 10], device=None):

    vi_gray = torch.mean(vi, 1, keepdim=True)
    fu_gray = torch.mean(fu, 1, keepdim=True)
    sobelconv=Sobelxy(device) 

    vi_grad_x, vi_grad_y = sobelconv(vi_gray)
    ir_grad_x, ir_grad_y = sobelconv(ir)
    fu_grad_x, fu_grad_y = sobelconv(fu_gray)
    grad_joint_x = torch.max(vi_grad_x, ir_grad_x)        
    grad_joint_y = torch.max(vi_grad_y, ir_grad_y)
    loss_grad = F.l1_loss(grad_joint_x, fu_grad_x) + F.l1_loss(grad_joint_y, fu_grad_y)

    loss_intensity = torch.mean(torch.pow((fu - vi), 2)) + torch.mean((fu_gray < ir) * torch.abs((fu_gray - ir)))

    loss_total = weights[0] * loss_grad + weights[1] * loss_intensity
    return loss_total, loss_intensity, loss_grad


def calculate_cosine_similarity_loss(intermediate_outputs_student, intermediate_outputs_teacher):
	assert len(intermediate_outputs_student) == len(intermediate_outputs_teacher), "Input tuples must have the same length"
	total_loss = 0.0
	for student, teacher in zip(intermediate_outputs_student, intermediate_outputs_teacher):
		cosine_sim = F.cosine_similarity(student.flatten(1), teacher.flatten(1), dim=1)
		cosine_loss = 1 - cosine_sim.mean()
		total_loss += cosine_loss
	return total_loss


class GWFE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GWFE, self).__init__()
        self.groups = 4
        group_channels = in_channels // self.groups
        
        self.wtconv1 = WTConv2d(group_channels, group_channels, kernel_size=1)
        self.wtconv3 = WTConv2d(group_channels, group_channels, kernel_size=3)
        self.wtconv5 = WTConv2d(group_channels, group_channels, kernel_size=5)
        self.wtconv7 = WTConv2d(group_channels, group_channels, kernel_size=7)
        
        self.fusion_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        x_groups = torch.chunk(x, self.groups, dim=1)

        group1 = F.leaky_relu(self.wtconv1(x_groups[0]), negative_slope=0.1)
        group2 = F.leaky_relu(self.wtconv3(x_groups[1]), negative_slope=0.1)
        group3 = F.leaky_relu(self.wtconv5(x_groups[2]), negative_slope=0.1)
        group4 = F.leaky_relu(self.wtconv7(x_groups[3]), negative_slope=0.1)
        
        x = torch.cat([group1, group2, group3, group4], dim=1)
        return self.fusion_conv(x)


class CBR(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, 
                     padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act:  
            x = self.relu(x)
        return x
    


class CSAM(nn.Module):
    def __init__(
            self,
            in_chans: int,
            reduction: int = 16,
            min_channels: int = 8,
    ):
        super(CSAM, self).__init__()
        hidden_chans = max(in_chans // reduction, min_channels)
        self.mlp_chans = nn.Sequential(
            nn.Conv2d(in_chans, hidden_chans, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_chans, in_chans, kernel_size=1, bias=False),
        )
        self.space_attn_1 = nn.Conv2d(in_chans, 1, kernel_size=3, padding=1, bias=False)     
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        avg_x_s = x.mean((2, 3), keepdim=True)
        max_x_s = x.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        avg_x_s_new = (self.mlp_chans(avg_x_s)).squeeze(-1)
        max_x_s_new = (self.mlp_chans(max_x_s)).squeeze(-1)
        attn = self.gate(torch.matmul(avg_x_s_new, max_x_s_new.transpose(1, 2)))
        x_flatten = x.view(B, C, -1)  # [1,128,126*126]
        x_flatten = torch.matmul(attn, x_flatten)  # [1,128,126*126]
        x = x_flatten.view(B, C, H, W) # [1,128,126,126]

        x_space_1 = self.space_attn_1(x)
        spital_attn = self.gate(x_space_1)
        return spital_attn


class CAFM(nn.Module):
    def __init__(self, in_c, dim, num_heads=4, kernel_size=3, padding=1, stride=1,
                 attn_drop=0.):
        super().__init__()

        self.dim = dim        
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5 
        
        self.attn_fg = CSAM(in_chans=in_c)
        self.attn_bg = CSAM(in_chans=in_c)
        
        self.attn_drop = nn.Dropout(attn_drop)
        
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)
        
        self.input_cbr = nn.Sequential(
            CBR(dim, dim, kernel_size=3, padding=1)   
        )

        self.fg_cbr = nn.Sequential(
            CBR(dim, dim, kernel_size=3, padding=1) 
        )
        
        self.bg_cbr = nn.Sequential(
            CBR(dim, dim, kernel_size=3, padding=1) 
        )
        
        self.relu = nn.ReLU(inplace=True)

        self.feat_fuse = nn.Sequential(
            nn.Conv2d(in_c*2, in_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
        )

        self.output_cbr = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, fg, bg):
        x = self.feat_fuse(torch.cat([fg, bg], dim=1))
        B, H, W, C = x.shape
        v = x

        fg = self.fg_cbr(fg)
        attn_fg = self.compute_attention(fg, B, H, W, C, 'fg')  
        bg = self.bg_cbr(bg)
        attn_bg = self.compute_attention(bg, B, H, W, C, 'bg')

        attn = attn_fg + attn_bg
        x_weighted_bg = attn * v
        out = self.output_cbr(x_weighted_bg) 

        return out

    def compute_attention(self, feature_map, B, H, W, C, feature_type):
        attn_layer = self.attn_fg if feature_type == 'fg' else self.attn_bg
        attn = attn_layer(feature_map)
        attn = attn * self.scale
        attn = self.attn_drop(attn)  
        return attn


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

    

class teacher_model(nn.Module):
    def __init__(self, num_channel, base_filter):
        super(teacher_model, self).__init__()
        self.encoder = Encoder(num_channel,base_filter)
        self.decoder = Decoder(num_channel,base_filter)
        
    def forward(self, I1):
        I1res,I1res2,I1res4,I18 = self.encoder(I1)
        # decoder
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

class stu_model(nn.Module):
    def __init__(self, num_channel, base_filter):
        super(stu_model, self).__init__()
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

        # decoder
        rI1, I8_d, I4, I2, I1_d = self.decoder(fuse1,fuse2,fuse4,fuse8, tea_dec_feat, rate)
        return rI1,[fuse1,fuse2,fuse4,fuse8], [I8_d, I4, I2, I1_d]
    
    def test(self, vi, ir):
        I1res,I1res2,I1res4,I18 = self.encoder_vi(vi)
        I2res,I2res2,I2res4,I28 = self.encoder_ir(ir)
        fuse1,fuse2,fuse4,fuse8 = self.adpt_fuse(I1res,I1res2,I1res4,I18,I2res,I2res2,I2res4,I28)
 
        # decoder
        rI1, _, _, _, _ = self.decoder(fuse1,fuse2,fuse4,fuse8, None, 0)
        return rI1

device = torch.device('cuda:0')
l1_loss = torch.nn.L1Loss().to(device)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='stu_pyramid_model', help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=250, type=int)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--gamma', default=0.5, type=int)
    parser.add_argument('--sbatch_size', default=8, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float)
    parser.add_argument('--weight', default=[1,1,0.0001, 0.0002], type=float)
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)

    args = parser.parse_args()
    return args

from tqdm import tqdm
import torch.backends.cudnn as cudnn
seed = 555
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark  = False
import torchvision.transforms as transforms
import joblib
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
    progress = min(epoch/total_epochs, 1.0)   
    rate = 0.5 * (1 + math.cos(math.pi * progress))
    return rate


def train(args, train_loader_ir, model_tea, model_stu, optimizer, epoch, loss_history):
    model_stu.train()
    model_tea.eval()
    rate = cosin_decay(epoch+1,args.epochs)
    for i, (batch_s) in tqdm(enumerate(zip(train_loader_ir))):
        ir_img = Variable(batch_s[0][0])
        ir_img = ir_img.to(device)
        vi_img = Variable(batch_s[0][1])
        vi_img = vi_img.to(device)
        Y_vi, Cb_vi, Cr_vi = RGB2YCrCb(vi_img)
        tea_img = Variable(batch_s[0][2])
        tea_img = tea_img.to(device)
        optimizer.zero_grad()
        with torch.no_grad(): 
            _, tea_enc_feat, tea_dec_feat = model_tea(tea_img)
        stu_fused_img, stu_fuse_feat, stu_dec_feat = model_stu(Y_vi, ir_img, tea_enc_feat, tea_dec_feat, rate)
        fuse_loss, _, _ =  Fusion_loss(vi_img, ir_img, stu_fused_img,device=device)
        kd_gt_loss = l1_loss(stu_fused_img, tea_img)
        kd_feat_loss = calculate_cosine_similarity_loss(stu_fuse_feat, tea_enc_feat)
        total_loss = fuse_loss + kd_gt_loss * 100 + kd_feat_loss * 100
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model_stu.parameters(), max_norm=1e-4, norm_type=2)
        optimizer.step()
        model_stu.zero_grad()
    
def main():
    args = parse_args()
    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)
    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))

    joblib.dump(args, 'models/%s/args.pkl' %args.name)
    cudnn.benchmark = True

    # supervised mfif data
    train_dir_ir = "./datasets/MSRS/train/ir/" 
    train_dir_vi = "./datasets/MSRS/train/vi/" 
    train_dir_tea_img = "./MKDFusion_tea_Images/"
    train_name_list = os.listdir(train_dir_ir)
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          ])
    dataset_train_ir = Get_MEF_Dataset_RGB(train_dir_ir, train_dir_vi, train_dir_tea_img, train_name_list,
                                    transform=transform_train)
    train_loader_ir = DataLoader(dataset_train_ir,
                              shuffle=True,
                              batch_size=args.sbatch_size)
    model_tea = teacher_model(num_channel=1, base_filter=16).to(device)
    model_tea.load_state_dict(torch.load('./stu_kd/tea_recon_model.pth'))
    model_stu = stu_model(num_channel=1, base_filter=16).to(device)
    milestones = []
    for i in range(1, args.epochs+1):
        if i == 100:
            milestones.append(i)
        if i == 200:
            milestones.append(i)
    optimizer = optim.Adam(model_stu.parameters(), lr=args.lr,
                           betas=args.betas, eps=args.eps)
    scheduler_f = lrs.MultiStepLR(optimizer, milestones, args.gamma)
    loss_history = dict((k, []) for k in ["epoch", "fusion_loss", "kd_gt_loss", "kd_feat_loss"])
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch+1, args.epochs))
        model_stu.zero_grad()
        train(args,train_loader_ir, model_tea,model_stu,optimizer,epoch, loss_history)
        ckt['a'] = ckt['a'] + 2e-4

        scheduler_f.step()
        if (epoch+1) % 1 == 0:
            torch.save(model_stu.state_dict(), 'models_pl/model_{}.pth'.format(epoch+1))
 

if __name__ == '__main__':
    ckt = {'epoch':0, 'psnr':0.0, 'a':0.0} 
    i_iter = 0
    main()
