import torch
import torch.nn as nn
import torch.nn.functional as F

class se_block(nn.Module):
    def __init__(self, channel, kernel, stride, padding):
        super(se_block, self).__init__()
        self.channel = channel
        self.kernel = kernel
        self.strdie = stride
        self.padding = padding

        self.conv1 = nn.Conv2d(channel, channel, kernel, stride, padding)
        self.conv1_norm = nn.InstanceNorm2d(channel)
        self.conv2 = nn.Conv2d(channel, channel, kernel, stride, padding)
        self.conv2_norm = nn.InstanceNorm2d(channel)

        self.se_conv1 = nn.Conv2d(channel, channel//16, kernel_size=1)
        self.se_conv2 = nn.Conv2d(channel//16, channel, kernel_size=1)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        output = F.relu(self.conv1_norm(self.conv1(x)))
        output = self.conv2_norm(self.conv2(output))
        
        se = F.avg_pool2d(output, output.size(2))
        se = F.relu(self.se_conv1(se))
        se = F.sigmoid(self.se_conv2(se))

        output = output * se
        output += x
        output = F.relu(output)
        return output

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class Generator(nn.Module):
    def __init__(self, conf):
        super(Generator, self).__init__()
        noise = conf.noise_dim
        channel = conf.channel_num
        self.batch_size = conf.batch_size
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(noise, channel*16, 8, 1, 0),
            nn.InstanceNorm2d(channel*16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(channel*16, channel*8, 4, 2, 1),
            nn.InstanceNorm2d(channel*8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(channel*8, channel*4, 4, 2, 1),
            nn.InstanceNorm2d(channel*4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(channel*4, channel*2, 4, 2, 1),
            nn.InstanceNorm2d(channel*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(channel*2, channel*1, 4, 2, 1),
            nn.InstanceNorm2d(channel*1),
            nn.ReLU(inplace=True),
        )

        self.resnet_blocks = []
        for i in range(conf.block_num):
                self.resnet_blocks.append(se_block(channel*1, 3, 1, 1))
                self.resnet_blocks[i].weight_init(0, 0.02)
        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        self.final_conv = nn.Sequential(
            nn.Conv2d(channel*1, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1, 1, 1)
        x = self.deconv(x)
        x = self.resnet_blocks(x)
        x = self.final_conv(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, conf):
        super(Discriminator, self).__init__()
        channel = conf.channel_num
        self.pre_conv = nn.Sequential(
            nn.Conv2d(3, channel*1, 4, 2, 1),
            nn.InstanceNorm2d(channel*1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.resnet_blocks = []
        for i in range(conf.block_num):
                self.resnet_blocks.append(se_block(channel*1, 3, 1, 1))
                self.resnet_blocks[i].weight_init(0, 0.02)
        self.resnet_blocks = nn.Sequential(*self.resnet_blocks)

        self.conv = nn.Sequential(
            nn.Conv2d(channel*1, channel*2, 4, 2, 1),
            nn.InstanceNorm2d(channel*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(channel*2, channel*4, 4, 2, 1),
            nn.InstanceNorm2d(channel*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(channel*4, channel*8, 4, 2, 1),
            nn.InstanceNorm2d(channel*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(channel*8, channel*16, 4, 2, 1),
            nn.InstanceNorm2d(channel*16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(channel*16, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.resnet_blocks(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return x