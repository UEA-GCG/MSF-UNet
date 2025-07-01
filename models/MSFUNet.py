import math

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from scipy import io
from torchvision import transforms

from torchvision import utils
from models.common import *
from models import common
import torch.fft


def make_model(opt):
    return MSFUNet(opt)


class MSFUNet(nn.Module):
    def __init__(self, opt, conv=common.default_conv):
        super(MSFUNet, self).__init__()
        self.opt = opt
        self.scale = opt.scale
        self.phase = len(opt.scale)
        n_blocks = opt.n_blocks
        n_feats = opt.n_feats
        kernel_size = 3

        act = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=max(opt.scale),
                                    mode='bicubic', align_corners=False)


        self.head = conv(opt.n_colors, n_feats*2, kernel_size)

        self.down = [
            common.DownBlock(opt, 2, n_feats * 2, n_feats * 2, n_feats * 2
            ) for p in range(self.phase)
        ]

        self.down = nn.ModuleList(self.down)


        up_body_blocks1 = [[
            common.RDCN(
                n_feats * 5, n_feats * 3, n_feats, n_blocks, 6
            )
        ] for p in range(self.phase, 1, -1)
        ]

        up_body_blocks1.insert(0, [
            common.RDCN(
                n_feats * 3, n_feats * 3, n_feats, n_blocks, 6
            )
        ])

        # up_body_blocks = [[
        #     common.RCAB(
        #         conv, n_feats * 5, kernel_size, act=act
        #     ) for _ in range(n_blocks)
        # ] for p in range(self.phase, 1, -1)
        # ]
        #
        # up_body_blocks.insert(0, [
        #     common.RCAB(
        #         conv, n_feats * 3, kernel_size, act=act
        #     ) for _ in range(n_blocks)
        # ])


        # up_body_blocks2 = [[
        #     DBPN(
        #         num_channels=5*n_feats, base_filter=64,  feat = 256, num_stages=7, scale_factor=2, out_channels=2*n_feats
        #     )
        # ] for p in range(self.phase, 1, -1)
        # ]
        # up_body_blocks2.insert(0, [
        #     DBPN(
        #         num_channels=3*n_feats, base_filter=64,  feat = 256, num_stages=7, scale_factor=2, out_channels=2*n_feats
        #     )
        # ])



        # The fisrt upsample block
        up = [[
            common.Upsampler(conv, 2, n_feats * 3, act=False),
            conv(n_feats * 3, n_feats * 2, kernel_size=1)
        ]]



        # The rest upsample blocks
        for p in range(self.phase - 1, 0, -1):
            up.append([
                common.Upsampler(conv, 2, n_feats * 3, act=False),
                conv(n_feats * 3, n_feats * 2, kernel_size=1)
            ])

        self.up_blocks = nn.ModuleList()
        for idx in range(self.phase):
            self.up_blocks.append(
                nn.Sequential(*up_body_blocks1[idx], *up[idx])
            )

        # tail conv that output sr imgs
        tail = [nn.Sequential(
                conv(n_feats * 5, 2*n_feats, kernel_size),
                nn.ReLU(True),
                common.RCAB(conv, 2*n_feats, kernel_size, act=act),
                # common.RDCN(
                #     n_feats * 5, n_feats * 2, n_feats, n_blocks, 6
                # ),
                conv(2*n_feats, opt.n_colors, kernel_size))]

        self.tail = nn.ModuleList(tail)



        # rgb downsample blocks
        self.down_rgb_blocks = nn.ModuleList()
        for idx in range(self.phase+1):
            self.down_rgb_blocks.append(
                nn.Sequential(
                    nn.AvgPool2d(2, 2),
                    conv(n_feats, 64, kernel_size=3),
                    nn.ReLU(True),
                    conv(64, 64, kernel_size=3),
                    nn.ReLU(True),
                    conv(64, n_feats, kernel_size=3),
                    nn.ReLU(True)
                )
            )
        self.rgb_conv1 = nn.Sequential(
            conv(n_feats, 64, kernel_size=3),
            nn.ReLU(True),
            conv(64, 64, kernel_size=3),
            nn.ReLU(True),
            conv(64, n_feats, kernel_size=3),
            nn.ReLU(True)
        )

        # self.res = common.ResBlock(conv, n_feats, 3)


        # self.fre_process = SDB(opt, n_feats, n_feats)

        self.conv_dp1 = nn.Conv2d(in_channels=1, out_channels=n_feats,
                                  kernel_size=3, padding=1)
        self.conv_rg1 = nn.Conv2d(in_channels=3, out_channels=n_feats,
                                  kernel_size=3, padding=1)
        self.attention = SUFT(dp_feats=n_feats)



    def forward(self, depth_lr, rgb):
        depth = self.upsample(depth_lr)


        # SUFT guid RGB image
        dp = self.conv_dp1(depth)

        # input_ = dp.cpu().detach().numpy()
        # io.savemat("dp.mat", {"dp": input_})

        rg = self.conv_rg1(rgb)

        # input_ = rg.cpu().detach().numpy()
        # io.savemat("rg.mat", {"rg": input_})

        rgb = self.attention(dp, rg)

        # input_ = rgb.cpu().detach().numpy()
        # io.savemat("rgb.mat", {"rgb": input_})

        # upsample x to target sr size
        # depth_lr = self.upsample(depth_lr)


        # preprocess
        x = self.head(depth)
        y = self.rgb_conv1(rgb)

        # down phases,
        copies = []
        for idx in range(self.phase):
            copies.append(x)
            x = self.down[idx](x)

        # rgb down
        copies_rgb = []
        for idx in range(self.phase+1):
            copies_rgb.append(y)
            y = self.down_rgb_blocks[idx](y)

        # up phases
        # sr = self.add_mean(sr)
        x = torch.cat((x, copies_rgb[self.phase]), 1)
        # sr = self.tail[0](x)
        results = []
        for idx in range(self.phase):
            # upsample to SR features
            x = self.up_blocks[idx](x)
            # concat down features and upsample features
            x = torch.cat((x, copies[self.phase - idx - 1], copies_rgb[self.phase - idx - 1]), 1)
            # output sr imgs
            # sr = self.tail[idx + 1](x)


            if idx == self.phase - 1:
                sr = self.tail[0](x)
                sr = sr + depth


                results.append(sr)

        return results[-1]












class SUFT(nn.Module):
    def __init__(self, dp_feats, conv=common.default_conv):
        super(SUFT, self).__init__()
        self.scb1 = conv(dp_feats, dp_feats, kernel_size=3)
        self.scb2 = conv(dp_feats, dp_feats, kernel_size=3)
        self.scb3 = conv(dp_feats, dp_feats, kernel_size=3)
        # self.dp_up = DenseProjection(dp_feats, dp_feats, scale, up=True, bottleneck=False)
        # self.dpf_up = DenseProjection(dp_feats, dp_feats, scale, up=True, bottleneck=False)
        self.conv_du = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True)

    def forward(self, depth, rgb):
        dpf = self.scb1(depth)

        dp_h = self.scb2(depth)

        # input_ = dp_h.cpu().detach().numpy()
        # io.savemat("dph.mat", {"dph": input_})

        dif = torch.abs(dp_h - dpf)

        dif_avg = torch.mean(dif, dim=1, keepdim=True)
        dif_max, _ = torch.max(dif, dim=1, keepdim=True)
        attention = self.conv_du(torch.cat([dif_avg, dif_max], dim=1))
        max = torch.max(torch.max(attention, -1)[0], -1)[0].unsqueeze(1).unsqueeze(2)
        min = torch.min(torch.min(attention, -1)[0], -1)[0].unsqueeze(1).unsqueeze(2)

        attention = (attention - min) / (max - min + 1e-12)

        # input_ = attention.cpu().detach().numpy()
        # io.savemat("att.mat", {"att": input_})

        rgb = self.scb3(rgb)
        rgb_h = rgb * attention
        return rgb_h











def projection_conv(in_channels, out_channels, scale, up=True):
    kernel_size, stride, padding = {
        2: (6, 2, 2),
        4: (8, 4, 2),
        8: (12, 8, 2),
        16: (20, 16, 2)
    }[scale]
    if up:
        conv_f = nn.ConvTranspose2d
    else:
        conv_f = nn.Conv2d

    return conv_f(
        in_channels, out_channels, kernel_size,
        stride=stride, padding=padding
    )





class DenseProjection(nn.Module):
    def __init__(self, in_channels, nr, scale, up=True, bottleneck=True):
        super(DenseProjection, self).__init__()
        self.up = up
        if bottleneck:
            self.bottleneck = nn.Sequential(*[
                nn.Conv2d(in_channels, nr, 1),
                nn.PReLU(nr)
            ])
            inter_channels = nr
        else:
            self.bottleneck = None
            inter_channels = in_channels

        self.conv_1 = nn.Sequential(*[
            projection_conv(inter_channels, nr, scale, up),
            nn.PReLU(nr)
        ])
        self.conv_2 = nn.Sequential(*[
            projection_conv(nr, inter_channels, scale, not up),
            nn.PReLU(inter_channels)
        ])
        self.conv_3 = nn.Sequential(*[
            projection_conv(inter_channels, nr, scale, up),
            nn.PReLU(nr)
        ])

    def forward(self, x):
        if self.bottleneck is not None:
            x = self.bottleneck(x)

        a_0 = self.conv_1(x)
        b_0 = self.conv_2(a_0)
        e = b_0.sub(x)
        a_1 = self.conv_3(e)

        out = a_0.add(a_1)
        return out








