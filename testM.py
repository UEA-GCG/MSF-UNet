import argparse
import os

from thop import profile
import cv2

from utils import *
import numpy as np
import torchvision.transforms as transforms
from torchvision import utils
from torch import Tensor
from PIL import Image
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from option import args
from models import MSFUNet
from data.nyu_dataloader import *
from data.rgbdd_dataloader import *
from data.middlebury_dataloader import Middlebury_dataset

import torch.nn as nn
import torch.nn.functional as F
import torch
# os.environ['CUDA_VISIBLE_DEVICE']="4"

def depth_to_color(depth_image):
    # 将深度值缩放到0-255范围
    scaled_depth = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 应用伪彩色映射
    color_map = cv2.applyColorMap(scaled_depth, cv2.COLORMAP_JET)

    return color_map


if __name__ == '__main__':
    net = MSFUNet.make_model(args)
    net.load_state_dict(torch.load("experiment/best_model.pth", map_location='cuda:0'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    data_transform = transforms.Compose([transforms.ToTensor()])
    dataset = Middlebury_dataset(root_dir='./data/SRData/Middlebury', scale=args.scale[-1], transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    data_num = len(dataloader)

    upsample = nn.Upsample(scale_factor=16,
                           mode='bicubic', align_corners=False)

    rmse = np.zeros(data_num)
    mad = 0.0
    with torch.no_grad():
        net.eval()


        for idx, data in enumerate(dataloader):
            guidance, lr, gt = data['guidance'].to(device), data['lr'].to(device), data['gt'].to(device)



            out = net(lr, guidance)

            lr_bic = upsample(lr)
            lr_bic = lr_bic.detach().cpu().numpy()
            out_n = out.detach().cpu().numpy()
            gt_n = gt.detach().cpu().numpy()
            lr_bic = np.squeeze(lr_bic)
            out_n = np.squeeze(out_n)
            gt_n = np.squeeze(gt_n)
            lr_bic = depth_to_color(lr_bic)
            out_n = depth_to_color(out_n)
            gt_n = depth_to_color(gt_n)
            cv2.imwrite(f"./test/RDCA6.0/Middlebury/Middlebury_X{args.scale[-1]}_{idx}Bic.png", lr_bic)
            cv2.imwrite(f"./test/RDCA6.0/Middlebury/Middlebury_X{args.scale[-1]}_{idx}depth.png", out_n)
            cv2.imwrite(f"./test/RDCA6.0/Middlebury/Middlebury_X{args.scale[-1]}_{idx}GT.png", gt_n)

            utils.save_image(guidance, f"./test/RDCA6.0/Middlebury/Middlebury_X{args.scale[-1]}_{idx}RGB.png", nrow=1,
                             normalize=False)


            rmse[idx] = midd_calc_rmse(gt[0,0], out[0,0])
        print(f"Middlebury_X{args.scale[-1]} 平均RMSE：{rmse.mean()}")

