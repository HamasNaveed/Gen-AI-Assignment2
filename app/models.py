"""CycleGAN networks (ResNet generator + PatchGAN discriminator) shared by the notebook and the Flask app."""
import torch
import torch.nn as nn

IMG_SIZE = 256   # training / inference resolution
N_RES = 9        # residual blocks in the generator (9 for 256x256, as in the CycleGAN paper)


class ResnetBlock(nn.Module):
    """Residual block: conv-IN-ReLU-conv-IN with a skip connection and reflection padding."""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3), nn.InstanceNorm2d(dim))

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """c7s1-64, d128, d256, R256 x n_res, u128, u64, c7s1-3 + Tanh (Johnson et al. style, as in CycleGAN)."""
    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_res=N_RES):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(in_ch, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        ch = ngf
        for _ in range(2):  # downsample
            layers += [nn.Conv2d(ch, ch * 2, 3, 2, 1), nn.InstanceNorm2d(ch * 2), nn.ReLU(True)]
            ch *= 2
        layers += [ResnetBlock(ch) for _ in range(n_res)]
        for _ in range(2):  # upsample
            layers += [nn.ConvTranspose2d(ch, ch // 2, 3, 2, 1, output_padding=1), nn.InstanceNorm2d(ch // 2), nn.ReLU(True)]
            ch //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ch, out_ch, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """70x70 PatchGAN: C64-C128-C256-C512 then a 1-channel patch score map."""
    def __init__(self, in_ch=3, ndf=64):
        super().__init__()
        def blk(i, o, s, norm=True):
            return [nn.Conv2d(i, o, 4, s, 1)] + ([nn.InstanceNorm2d(o)] if norm else []) + [nn.LeakyReLU(0.2, True)]
        self.model = nn.Sequential(*blk(in_ch, ndf, 2, False), *blk(ndf, ndf * 2, 2), *blk(ndf * 2, ndf * 4, 2),
                                   *blk(ndf * 4, ndf * 8, 1), nn.Conv2d(ndf * 8, 1, 4, 1, 1))

    def forward(self, x):
        return self.model(x)


def init_weights(net, std=0.02):
    """Initialise conv weights from N(0, 0.02) as in the paper."""
    for m in net.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return net
