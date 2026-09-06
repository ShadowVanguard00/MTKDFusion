import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import kornia.filters as KF
import torch
import torch.nn.functional as F
import numpy as np
from math import exp

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



def Ref_loss(ref, fu, weights=[10, 10], device=None):

    ref_gray = torch.mean(ref, 1, keepdim=True)
    fu_gray = torch.mean(fu, 1, keepdim=True)
    sobelconv=Sobelxy(device) 

    ref_grad_x, ref_grad_y = sobelconv(ref)
    fu_grad_x, fu_grad_y = sobelconv(fu_gray)
    loss_grad = F.l1_loss(ref_grad_x, fu_grad_x) + F.l1_loss(ref_grad_y, fu_grad_y)

    loss_intensity = torch.mean(torch.pow((fu - ref), 2))

    loss_total = weights[0] * loss_grad + weights[1] * loss_intensity
    return loss_total, loss_intensity, loss_grad

class Sobelxy(nn.Module):
    def __init__(self, device):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).to(device=device)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).to(device=device)
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx), torch.abs(sobely)