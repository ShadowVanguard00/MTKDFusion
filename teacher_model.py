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
from dataset import Get_SDataset
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from PIL import Image
import glob
from WTConv import WTConv2d


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
                                     nn.Conv2d(base_filter, num_channel, 3, 1, 0, bias=False),
                                     nn.Tanh())
    def forward(self, Ires,I2res,I4res,I8):
        I4 = I4res + self.up(self.dformer8(I8))
        I2 = I2res + self.up(self.dformer4(I4))
        I = Ires + self.up(self.dformer2(I2)) 
        I = self.dformer1(I)
        I = self.out_conv(I)
        return I
    

class teacher_model(nn.Module):
    def __init__(self, num_channel, base_filter):
        super(teacher_model, self).__init__()
        self.encoder = Encoder(num_channel,base_filter)
        self.decoder = Decoder(num_channel,base_filter)
        
    def forward(self, I1):
        I1res,I1res2,I1res4,I18 = self.encoder(I1)
        # decoder
        rI1 = self.decoder(I1res,I1res2,I1res4,I18)
        return rI1

device = torch.device('cuda:0')
l1_loss = torch.nn.L1Loss().to(device)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='pyramid_model', help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=100, type=int)
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


def train(args, train_loader_ir, model, optimizer,epoch):
    model.train()
    for i, (batch_s) in tqdm(enumerate(zip(train_loader_ir))):
        nf = Variable(batch_s[0])
        nf = nf.to(device)
        
        optimizer.zero_grad()
        
        out_nf = model(nf)

        mfif_loss =  l1_loss(out_nf, nf)
        total_loss = mfif_loss
        print("Loss: {:.2e}".format(total_loss.item()))

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e-4, norm_type=2)
        optimizer.step()
        model.zero_grad()


def main():
    args = parse_args()

    if not os.path.exists('models/%s' %args.name):
        os.makedirs('models/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))

    joblib.dump(args, 'models/%s/args.pkl' %args.name)
    cudnn.benchmark = True

    train_dir_f = "./tea_recon/MKDFusion_tea_Images/"
    train_name_list = os.listdir(train_dir_f)

    transform_train = transforms.Compose([transforms.ToTensor(),
                                          ])
    dataset_train_ir = Get_SDataset(train_dir_f, train_name_list,
                                    transform=transform_train)

    train_loader_ir = DataLoader(dataset_train_ir,
                              shuffle=True,
                              batch_size=args.sbatch_size)
    
    model = teacher_model(num_channel=1, base_filter=16).to(device)

    milestones = []
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=args.betas, eps=args.eps)
    scheduler_f = lrs.MultiStepLR(optimizer, milestones, args.gamma)

    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' % (epoch+1, args.epochs))
        model.zero_grad()
        train(args,train_loader_ir,model,optimizer,epoch)
        ckt['a'] = ckt['a'] + 2e-4

        scheduler_f.step()
        if (epoch+1) % 1 == 0:
            torch.save(model.state_dict(), 'models_ch16/%s/model_{}.pth'.format(epoch+1) %args.name)


if __name__ == '__main__':
    ckt = {'epoch':0, 'psnr':0.0, 'a':0.0} 
    i_iter = 0
    main()