"""
Discriminator and Generator implementation from DCGAN paper
WGAN中的判别器被称为critic，为了让判别器函数f满足1李普希茨连续（f' <= 1）,
判别器的最后去掉了nn.Sigmoid()函数。
WGAN-GP在WGAN的基础上，将判别器中的BatchNorm层换成了LayerNorm层
"""

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channel_image, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # 输入形状： N x channels_image x 64 x 64
            nn.Conv2d(  # 判别器D的第一层不用BatchNorm
                channel_image, features_d, kernel_size=4, stride=2, padding=1
            ),  # 32x32
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d*2, 4, 2, 1),  # 16x16
            self._block(features_d*2, features_d*4, 4, 2, 1),  # 8x8
            self._block(features_d*4, features_d*8, 4, 2, 1),  # 4x4
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),  # 1x1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,  # 后面有BatchNorm，无需用bias
            ),
            # LayerNorm <---> InstanceNorm
            nn.InstanceNorm2d(out_channels, affine=True),  # affine=True是为了设置可学习的参数
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):  # f_g取64，则f_g*16=1024
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # 输入： N x z_dim x 1 x 1
            self._block(z_dim, features_g*16, 4, 1, 0),  # N x f_g*16 x 4 x 4 ( 4=(1-1)*1+4-2*0 )
            self._block(features_g*16, features_g*8, 4, 2, 1),  # 8x8 ( 8=(4-1)*2+4-2*1 )
            self._block(features_g*8, features_g*4, 4, 2, 1),  # 16x16 ( 16=(8-1)*2+4-2*1 )
            self._block(features_g*4, features_g*2, 4, 2, 1),   # 32x32
            nn.ConvTranspose2d(  # 生成器最后一层不用BatchNorm,与channel_img有关
                features_g*2, channels_img, kernel_size=4, stride=2, padding=1,  # 64x64
            ),
            # 输出： N x channels_img x 64 x 64
            nn.Tanh(),  # 像素值映射到-1~1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)


def initialize_weights(model):
    # Initialize weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)   # DCGAN原文将网络中的权重初始化为0均值0.02标准差的数

def test():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1)
    gen = Generator(z_dim, in_channels, 8)
    z = torch.randn((N, z_dim, 1, 1))
    initialize_weights(gen)
    assert gen(z).shape == (N, in_channels, H, W)
    print("Success")


if __name__ == '__main__':
    test()
