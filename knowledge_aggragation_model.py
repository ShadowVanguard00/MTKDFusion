import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import time
import torch
from torch.nn import functional as F
import torch.nn as nn
import os
from options import *
from create_dataset import *
from utils import *
from time import time
from tqdm import tqdm as tqdm
import datetime
import math
import argparse
from GIFNet_model import TwoBranchesFusionNet
from args import Args as args
from net import *
from FusionNet import FusionNet
from FFViT import FFVit
import glob
import re
import sys

import shutil 

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='./tea_weights/Final.model', help='fusion network weight')
opt_GIFNet = parser.parse_args()


def Fusion_loss(vi, ir, fu, weights=[10, 10], device=None):

    vi_gray = torch.mean(vi, 1, keepdim=True)
    fu_gray = torch.mean(fu, 1, keepdim=True)
    sobelconv=Sobelxy(device) 

    # 梯度损失
    vi_grad_x, vi_grad_y = sobelconv(vi_gray)
    ir_grad_x, ir_grad_y = sobelconv(ir)
    fu_grad_x, fu_grad_y = sobelconv(fu_gray)
    grad_joint_x = torch.max(vi_grad_x, ir_grad_x)        
    grad_joint_y = torch.max(vi_grad_y, ir_grad_y)
    loss_grad = F.l1_loss(grad_joint_x, fu_grad_x) + F.l1_loss(grad_joint_y, fu_grad_y)

    ## 强度损失
    loss_intensity = torch.mean(torch.pow((fu - vi), 2)) + torch.mean((fu_gray < ir) * torch.abs((fu_gray - ir)))

    loss_total = weights[0] * loss_grad + weights[1] * loss_intensity
    return loss_total, loss_intensity, loss_grad



def Ref_loss(ref, fu, weights=[10, 10], device=None):

    ref_gray = torch.mean(ref, 1, keepdim=True)
    fu_gray = torch.mean(fu, 1, keepdim=True)
    sobelconv=Sobelxy(device) 

    # 梯度损失
    ref_grad_x, ref_grad_y = sobelconv(ref)
    fu_grad_x, fu_grad_y = sobelconv(fu_gray)
    loss_grad = F.l1_loss(ref_grad_x, fu_grad_x) + F.l1_loss(ref_grad_y, fu_grad_y)

    ## 强度损失
    loss_intensity = torch.mean(torch.pow((fu - ref), 2))

    loss_total = weights[0] * loss_grad + weights[1] * loss_intensity
    return loss_total, loss_intensity, loss_grad

def YCbCr2RGB(Y, Cb, Cr):
    Y = (Y - Y.min()) / (Y.max() - Y.min())
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

def trainer(train_loader, 
                       net_g_tea1,
                       net_g_tea2,
                       net_g_tea3,
                       net_tea_fuse,
                       device, optimizer, opt, logger, start_ep=0, total_it=0):
    total_epoch = opt.n_ep
    saver = Saver(opt)    

    start = glob_st = time()
    for ep in range(start_ep, total_epoch):

        for it, (img_ir, img_vi, label) in enumerate(train_loader):
            total_it += 1
            img_ir = img_ir.to(device)
            img_vi = img_vi.to(device)
            Y_vi, Cb_vi, Cr_vi = RGB2YCrCb(img_vi)

            with torch.no_grad():
                tea1_fused_img = net_g_tea1(Y_vi, img_ir)


            with torch.no_grad():
                tea2_fused_img = run(net_g_tea2, img_ir, Y_vi)

            net_g_tea3_model_E = net_g_tea3[0]
            net_g_tea3_model_D = net_g_tea3[1]
            net_g_tea3_model_F = net_g_tea3[2] 

            with torch.no_grad(): 
                F_ir, emb_ir, F_vis, emb_vis= net_g_tea3_model_E(Y_vi,img_ir)
                F, emb_F = net_g_tea3_model_F(F_ir,F_vis, [emb_ir, emb_vis] )
                tea3_fused_img, F_fea= net_g_tea3_model_D(F, emb_F)
                     
            optimizer.zero_grad()

            tea_fused_img = net_tea_fuse(torch.cat([tea1_fused_img, tea2_fused_img, tea3_fused_img], dim=1))
                
            ## content and gradient loss            
            fusion_loss, int_loss, grad_loss = Fusion_loss(Y_vi, img_ir, tea_fused_img, device=device)
            ref1_loss, _, _ = Ref_loss(tea1_fused_img, tea_fused_img, device=device)
            ref2_loss, _, _ = Ref_loss(tea2_fused_img, tea_fused_img, device=device) 
            ref3_loss, _, _ = Ref_loss(tea3_fused_img, tea_fused_img, device=device)
            train_loss = 0.1 * fusion_loss +  (ref1_loss + ref2_loss + ref3_loss) / 3
            print('fusion_los:{}, ref1_loss:{}, ref2_loss:{}, ref3_loss:{}'.format(0.1 * fusion_loss,  ref1_loss, ref2_loss, ref3_loss))
            
            train_loss.backward()
            optimizer.step()
  
        lr = optimizer.get_lr()
        end = time()
        training_time, glob_t_intv = end - start, end - glob_st
        now_it = total_it + 1
        eta = int((total_epoch * len(train_loader) - now_it) * (glob_t_intv / (now_it)))
        eta = str(datetime.timedelta(seconds=eta))
        logger.info('ep: [{}/{}], learning rate: {:.6f}, time consuming: {:.2f}s, fusion loss: {:.4f}'.format(ep+1, total_epoch, lr, training_time, fusion_loss.item()))
        logger.info('grad loss: [{:.4f}], int loss: [{:.4f}] \n'.format(grad_loss, int_loss.item())) #, corr loss: [{:.4f}]
        start = time()
        ## save Visualization results
        if (ep + 1) % opt.img_save_freq == 0:
            input = [img_ir, img_vi,  tea1_fused_img, tea2_fused_img, tea3_fused_img, tea_fused_img]
            output = [img_ir, img_vi, YCbCr2RGB(tea1_fused_img, Cb_vi, Cr_vi) , YCbCr2RGB(tea2_fused_img, Cb_vi, Cr_vi), YCbCr2RGB(tea3_fused_img, Cb_vi, Cr_vi), YCbCr2RGB(tea_fused_img, Cb_vi, Cr_vi)]
            saver.write_img(ep, input, output)
        ## save model
        if (ep + 1) % opt.model_save_freq == 0:                       
            torch.save(
                net_tea_fuse.state_dict(),
                os.path.join(opt.result_dir, f"net_tea_fuse_epoch_{ep+1}.pth"))


def resume(model, optimizer=None, model_save_path=None, device=None, is_train=True):
    # weight
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint['MTAN'])
    if is_train:
        optimizer.load_state_dict(checkpoint['optimizer'])
        ep = checkpoint['ep']
        total_it = checkpoint['total_it']
        return model, optimizer, ep, total_it
    else:
        return model


def load_model(model_path_twoBranches):
    model = TwoBranchesFusionNet(args.s, args.n, args.channel, args.stride)

    model.load_state_dict(torch.load(model_path_twoBranches))

    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))
    
    total = sum([param.nelement() for param in model.parameters()])
    print('Number    of    parameter: {:4f}M'.format(total / 1e6))
    
    model.eval()
    if (args.cuda):
        model.cuda()

    return model

def run(model, ir_test_batch, vis_test_batch):
    img_ir = ir_test_batch
    img_vi = vis_test_batch
    
    img_ir = Variable(img_ir, requires_grad=False)
    img_vi = Variable(img_vi, requires_grad=False)

    fea_com = model.forward_encoder(img_ir, img_vi)    
    fea_fused = model.forward_MultiTask_branch(fea_com_ivif = fea_com, fea_com_mfif = fea_com)            
    out_y_or_gray = model.forward_mixed_decoder(fea_com, fea_fused);    
    
    out_y_or_gray = out_y_or_gray[:,:,:,:]
    
    return out_y_or_gray


class ka_model(nn.Module):
    """Base SR model for single image super-resolution."""
    def __init__(self):
        super(ka_model, self).__init__()
        self.device = torch.device('cuda')

        # teacher1 network (Seafusion)
        fusionmodel = FusionNet(output=1)
        fusionmodel.load_state_dict(torch.load("./tea_weights/fusionmodel_final.pth"))
        self.net_g_tea1 = fusionmodel.to(self.device)
        self.net_g_tea1.eval()

 
        # teacher2 network (GIFNet)
        model_path_twoBranches = opt_GIFNet.checkpoint
        self.net_g_tea2 = load_model(model_path_twoBranches).to(self.device)
        self.net_g_tea2.eval()        


        # teacher3 network (DAFusion)
        model_path_E =r'./tea_weights/model_E.pth'
        model_path_D =r'./tea_weights/model_D.pth'
        model_path_F =r'./tea_weights/model_F.pth'

        self.net_g_tea3_model_E = torch.load(model_path_E, map_location=self.device)
        self.net_g_tea3_model_D = torch.load(model_path_D, map_location=self.device) 
        self.net_g_tea3_model_F = torch.load(model_path_F, map_location=self.device)
 
        # 图像融合结构
        upscale = 2
        window_size = 8
        height = (48 // upscale // window_size + 1) * window_size
        width = (48 // upscale // window_size + 1) * window_size

        model = FFVit(in_chans=3, num_out_ch=1, num_layers=6, embed_dim=24, upscale=2)
        self.net_tea_fuse = model
        self.net_tea_fuse.train()
        self.net_tea_fuse.to(self.device)

    def optimize_parameters(self):
        # parse options    
        parser = TrainOptions()
        opts = parser.parse()

        device = torch.device("cuda:{}".format(opts.gpu) if torch.cuda.is_available() else "cpu")
  
        train_dataset = MSRSData(opts, is_train=True)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=opts.batch_size,
            num_workers = opts.nThreads,
            shuffle=True)

        ep_iter = len(train_loader)
        max_iter = opts.n_ep * ep_iter
        print('Training iter: {}'.format(max_iter))
        momentum = 0.9
        weight_decay = 5e-4
        lr_start = 1e-3
        power = 0.9
        warmup_steps = 10
        warmup_start_lr = 1e-5
        optimizer = Optimizer(
            model = self.net_tea_fuse,
            lr0 = lr_start,
            momentum = momentum,
            wd = weight_decay,
            warmup_steps = warmup_steps,
            warmup_start_lr = warmup_start_lr,
            max_iter = max_iter,
            power = power)

        ep += 1
        log_dir = os.path.join(opts.display_dir, 'logger', opts.name)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'log.txt')
        if os.path.exists(log_path):
            os.remove(log_path)
        logger = logger_config(log_path=log_path, logging_name='Timer')
        logger.info('Parameter: {:.6f}M'.format(count_parameters(self.net_tea_fuse) / 1024 * 1024))
        total_it = 0
    
        # Train and evaluate multi-task network
        trainer(train_loader,
                        self.net_g_tea1,
                        self.net_g_tea2,
                        [self.net_g_tea3_model_E, self.net_g_tea3_model_D, self.net_g_tea3_model_F],
                        self.net_tea_fuse,
                        device,
                        optimizer,
                        opts,
                        logger,
                        ep,
                        total_it)

    def test_one_dataset_weight(self):
        device = torch.device("cuda:0")   
        # define dataset    
        test_dataset = FusionData()
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=1,
            shuffle=False)

        test_bar= tqdm(test_loader)
        Fusion_save_dir = './fused_imgs_MSRS/knowledge_distillation'
        print(Fusion_save_dir)
        os.makedirs(Fusion_save_dir, exist_ok=True)

        self.net_tea_fuse.load_state_dict(torch.load("./net_tea_fuse_epoch_50.pth"))

        with torch.no_grad():  # operations inside don't track history
            for it, (img_ir, img_vi, img_names, widths, heights) in enumerate(test_bar):
                img_ir = img_ir.to(device)
                img_vi = img_vi.to(device)
                vi_Y, vi_Cb, vi_Cr = RGB2YCrCb(img_vi)
                vi_Y = vi_Y.to(device)
                vi_Cb = vi_Cb.to(device)
                vi_Cr = vi_Cr.to(device) 

                with torch.no_grad():
                    tea1_fused_img = self.net_g_tea1(vi_Y, img_ir)

                with torch.no_grad():
                    tea2_fused_img = run(self.net_g_tea2, img_ir, vi_Y)

                net_g_tea3_model_E = self.net_g_tea3_model_E
                net_g_tea3_model_D = self.net_g_tea3_model_D
                net_g_tea3_model_F = self.net_g_tea3_model_F

                with torch.no_grad(): 
                    F_ir, emb_ir, F_vis, emb_vis= net_g_tea3_model_E(vi_Y,img_ir)
                    F, emb_F = net_g_tea3_model_F(F_ir,F_vis, [emb_ir, emb_vis])
                    tea3_fused_img, F_fea = net_g_tea3_model_D(F, emb_F)

                with torch.no_grad(): 
                    tea_fused_img = self.net_tea_fuse(torch.cat([tea1_fused_img, tea2_fused_img, tea3_fused_img], dim=1))
                    tea_fused_img = torch.clamp(tea_fused_img, 0, 1)  # 新增的归一化操作
                    fused_img = YCbCr2RGB(tea_fused_img, vi_Cb, vi_Cr)

                # fused_img = YCbCr2RGB(fused_img, vi_Cb, vi_Cr)
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    fusion_save_name = os.path.join(Fusion_save_dir, img_name)
                    save_img_single(tea_fused_img[i, ::], fusion_save_name, widths[i], heights[i])
                    test_bar.set_description('Image: {} '.format(img_name))

if __name__ == '__main__':
   ka_instance = ka_model()
   ka_instance.optimize_parameters()