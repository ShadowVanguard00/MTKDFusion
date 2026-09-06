from operator import index
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import random
from random import randrange
import cv2
import numpy as np

from itertools import cycle
import os

class Get_SDataset(Dataset):
    def __init__(self, train_dir_f,train_name_list, is_patch=True, transform=None):
        super(Get_SDataset, self).__init__()
        
        self.train_name_list = train_name_list
        self.train_dir_f = train_dir_f
        self.transform = transform
        self.is_patch = is_patch

    def __getitem__(self, index):
        train_name = self.train_name_list[index]
        f = cv2.imread(os.path.join(self.train_dir_f, train_name), cv2.IMREAD_GRAYSCALE)
        f = f[..., np.newaxis]  # shape (H,W) -> (H,W,1)

        if self.is_patch:
            f = self.get_patch(f, patch_size=256)
        # ------------------To tensor------------------#
        if self.transform is not None:
            tran = transforms.ToTensor()
            f = tran(f)
            return f

    def __len__(self):
        return len(self.train_name_list)
    
    def get_patch(self, img_in, patch_size):
        h, w = img_in.shape[:2]

        stride = patch_size

        x = random.randint(0, w - stride)
        y = random.randint(0, h - stride)

        img_in = img_in[y:y + stride, x:x + stride, :]
        return img_in
    
from PIL import Image

class Get_MEF_Dataset_RGB(Dataset):
    def __init__(self, train_dirs_ir, train_dirs_vi, train_dirs_gt, is_patch=True, transform=None):
        super(Get_MEF_Dataset_RGB, self).__init__()

        if isinstance(train_dirs_ir, str):
            train_dirs_ir = [train_dirs_ir]
        if isinstance(train_dirs_vi, str):
            train_dirs_vi = [train_dirs_vi]
        if isinstance(train_dirs_gt, str):
            train_dirs_gt = [train_dirs_gt]

        assert len(train_dirs_ir) == len(train_dirs_vi) == len(train_dirs_gt)

        self.train_dirs_ir = train_dirs_ir
        self.train_dirs_vi = train_dirs_vi
        self.train_dirs_gt = train_dirs_gt
        self.transform = transform
        self.is_patch = is_patch
        self.samples = []
        for ir_dir, vi_dir, gt_dir in zip(train_dirs_ir, train_dirs_vi, train_dirs_gt):
            ir_names = sorted(os.listdir(ir_dir))
            vi_names = sorted(os.listdir(vi_dir))
            gt_names = sorted(os.listdir(gt_dir))

            common_names = sorted(list(set(ir_names) & set(vi_names) & set(gt_names)))

            for name in common_names:
                self.samples.append({
                    "ir": os.path.join(ir_dir, name),
                    "vi": os.path.join(vi_dir, name),
                    "gt": os.path.join(gt_dir, name)
                })

        print(f"共构建 {len(self.samples)} 个样本")

    def __getitem__(self, index):
        sample = self.samples[index]

        ir_img = cv2.imread(sample["ir"], cv2.IMREAD_GRAYSCALE)
        vi_img = cv2.imread(sample["vi"])
        tea_img = cv2.imread(sample["gt"], cv2.IMREAD_GRAYSCALE)

        if ir_img is None:
            raise FileNotFoundError(f"读取失败: {sample['ir']}")
        if vi_img is None:
            raise FileNotFoundError(f"读取失败: {sample['vi']}")
        if tea_img is None:
            raise FileNotFoundError(f"读取失败: {sample['gt']}")

        ir_img = ir_img[..., np.newaxis]
        vi_img = cv2.cvtColor(vi_img, cv2.COLOR_BGR2RGB)
        tea_img = tea_img[..., np.newaxis]

        ir_img, vi_img, tea_img = self.crop_to_multiple_of_16(ir_img, vi_img, tea_img)

        if self.is_patch:
            ir_img, vi_img, tea_img = self.get_patch(ir_img, vi_img, tea_img, patch_size=256)

        if self.transform is not None:
            tran = transforms.ToTensor()
            ir_img = tran(ir_img)
            vi_img = tran(vi_img)
            tea_img = tran(tea_img)

        return ir_img, vi_img, tea_img

    def __len__(self):
        return len(self.samples)

    def crop_to_multiple_of_16(self, img_ir, img_vi, img_gt):
        h, w = img_ir.shape[:2]
        new_h = (h // 16) * 16
        new_w = (w // 16) * 16

        img_ir = img_ir[:new_h, :new_w, :]
        img_vi = img_vi[:new_h, :new_w, :]
        img_gt = img_gt[:new_h, :new_w, :]

        return img_ir, img_vi, img_gt

    def get_patch(self, img_in, img_in1, img_tar, patch_size):
        h, w = img_in.shape[:2]

        if h < patch_size or w < patch_size:
            return img_in, img_in1, img_tar

        x = random.randint(0, w - patch_size)
        y = random.randint(0, h - patch_size)

        img_in = img_in[y:y + patch_size, x:x + patch_size, :]
        img_in1 = img_in1[y:y + patch_size, x:x + patch_size, :]
        img_tar = img_tar[y:y + patch_size, x:x + patch_size, :]

        return img_in, img_in1, img_tar

class Get_Test_Dataset(Dataset):
    def __init__(self, train_dir_ir , train_dir_vi, transform=None):
        super(Get_Test_Dataset, self).__init__()
        self.train_folder_ir = train_dir_ir
        self.train_folder_vi = train_dir_vi
        self.transform = transform
        self.gt_name_list = sorted(os.listdir(train_dir_ir))

    def __getitem__(self, index):
        image_name = self.gt_name_list[index]
        
        ir_img = cv2.imread(os.path.join(self.train_folder_ir, image_name), cv2.IMREAD_GRAYSCALE)
        ir_img = ir_img[..., np.newaxis]
        
        vi_img = cv2.imread(os.path.join(self.train_folder_vi, image_name))
        vi_img = cv2.cvtColor(vi_img, cv2.COLOR_BGR2RGB)
        
        if self.transform is not None:
            tran = transforms.ToTensor()
            ir_img = tran(ir_img)
            vi_img = tran(vi_img)
            return ir_img, vi_img, image_name

    def __len__(self):
        return len(self.gt_name_list)