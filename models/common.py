import math
import numpy as np
import torch
import torch.nn as nn
from option import args


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                # m.append(nn.ConvTranspose2d(n_feats, n_feats, 2, 2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class DownBlock(nn.Module):
    def __init__(self, opt, scale, nFeat=None, in_channels=None, out_channels=None):
        super(DownBlock, self).__init__()
        negval = opt.negval

        if nFeat is None:
            nFeat = opt.n_feats

        if in_channels is None:
            in_channels = opt.n_colors

        if out_channels is None:
            out_channels = opt.n_colors

        dual_block = [
            nn.Sequential(
                nn.Conv2d(in_channels, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(negative_slope=negval, inplace=True)
            )
        ]

        for _ in range(1, int(np.log2(scale))):
            dual_block.append(
                nn.Sequential(
                    nn.Conv2d(nFeat, nFeat, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.LeakyReLU(negative_slope=negval, inplace=True)
                )
            )

        dual_block.append(nn.Conv2d(nFeat, out_channels, kernel_size=3, stride=1, padding=1, bias=False))

        self.dual_module = nn.Sequential(*dual_block)

    def forward(self, x):
        x = self.dual_module(x)
        return x


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction=16, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res












class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDCB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDCB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])
        self.CALayer = CALayer(growth_rate, reduction= args.n_feats)

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.CALayer(self.lff(self.layers(x)))  # local residual learning





class RDCN(nn.Module):
    def __init__(self, num_channels, num_features, growth_rate, num_blocks, num_layers):
        super(RDCN, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # self.CALayer = CALayer(num_features, reduction=args.n_feats)

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, args.n_feats, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDCB(args.n_feats, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDCB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        return x















class DenseBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation='relu', norm='batch'):
        super(DenseBlock, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm1d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm1d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.fc(x))
        else:
            out = self.fc(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm='batch'):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(num_filter)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(num_filter)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        residual = x
        if self.norm is not None:
            out = self.bn(self.conv1(x))
        else:
            out = self.conv1(x)

        if self.activation is not None:
            out = self.act(out)

        if self.norm is not None:
            out = self.bn(self.conv2(out))
        else:
            out = self.conv2(out)

        out = torch.add(out, residual)
        return out


class UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class UpBlockPix(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, scale=4, bias=True, activation='prelu',
                 norm=None):
        super(UpBlockPix, self).__init__()
        self.up_conv1 = UpsamplerS(scale, num_filter)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = UpsamplerS(scale, num_filter)

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class D_UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu',
                 norm=None):
        super(D_UpBlock, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class D_UpBlockPix(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, scale=4, bias=True,
                 activation='prelu', norm=None):
        super(D_UpBlockPix, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.up_conv1 = UpsamplerS(scale, num_filter)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = UpsamplerS(scale, num_filter)

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class DownBlockS(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(DownBlockS, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class DownBlockPix(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, scale=4, bias=True, activation='prelu',
                 norm=None):
        super(DownBlockPix, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = UpsamplerS(scale, num_filter)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class D_DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu',
                 norm=None):
        super(D_DownBlock, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class D_DownBlockPix(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, scale=4, bias=True,
                 activation='prelu', norm=None):
        super(D_DownBlockPix, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = UpsamplerS(scale, num_filter)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class PSBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, scale_factor, kernel_size=3, stride=1, padding=1, bias=True,
                 activation='prelu', norm='batch'):
        super(PSBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size * scale_factor ** 2, kernel_size, stride, padding,
                                    bias=bias)
        self.ps = torch.nn.PixelShuffle(scale_factor)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.ps(self.conv(x)))
        else:
            out = self.ps(self.conv(x))

        if self.activation is not None:
            out = self.act(out)
        return out


class UpsamplerS(torch.nn.Module):
    def __init__(self, scale, n_feat, bn=False, act='prelu', bias=True):
        super(UpsamplerS, self).__init__()
        modules = []
        for _ in range(int(math.log(scale, 2))):
            modules.append(ConvBlock(n_feat, 4 * n_feat, 3, 1, 1, bias, activation=None, norm=None))
            modules.append(torch.nn.PixelShuffle(2))
            if bn: modules.append(torch.nn.BatchNorm2d(n_feat))
            # modules.append(torch.nn.PReLU())
        self.up = torch.nn.Sequential(*modules)

        self.activation = act
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.up(x)
        if self.activation is not None:
            out = self.act(out)
        return out


class Upsample2xBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, bias=True, upsample='deconv', activation='relu', norm='batch'):
        super(Upsample2xBlock, self).__init__()
        scale_factor = 2
        # 1. Deconvolution (Transposed convolution)
        if upsample == 'deconv':
            self.upsample = DeconvBlock(input_size, output_size,
                                        kernel_size=4, stride=2, padding=1,
                                        bias=bias, activation=activation, norm=norm)

        # 2. Sub-pixel convolution (Pixel shuffler)
        elif upsample == 'ps':
            self.upsample = PSBlock(input_size, output_size, scale_factor=scale_factor,
                                    bias=bias, activation=activation, norm=norm)

        # 3. Resize and Convolution
        elif upsample == 'rnc':
            self.upsample = torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                ConvBlock(input_size, output_size,
                          kernel_size=3, stride=1, padding=1,
                          bias=bias, activation=activation, norm=norm)
            )

    def forward(self, x):
        out = self.upsample(x)
        return out

