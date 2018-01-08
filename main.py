import os
import time
import argparse

import util
import model

import torch
from torch.autograd import Variable
from torch.optim import Adam
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='DCGAN')
# Directory
parser.add_argument('--dataset_dir', type=str, default='./')
parser.add_argument('--result_path', type=str, default='result')
# Data
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--fixed_num', type=int, default=32)
# Network
parser.add_argument('--noise_dim', type=int, default=64)
parser.add_argument('--channel_num', type=int, default=64)
parser.add_argument('--block_num', type=int, default=4)
# Training
parser.add_argument('--learning_rate', type=int, default=0.00005)
parser.add_argument('--final_epoch', type=int, default=300)
parser.add_argument('--log_frequency', type=int, default=5)
parser.add_argument('--save_frequency', type=int, default=10)
# Resume
parser.add_argument('--resume', type=bool, default=False)

config = parser.parse_args()

use_cuda = torch.cuda.is_available()
loader = util.get_loader(config)

if config.resume:
    print('-- Resuming From Checkpoint')
    assert os.path.isdir('checkpoint'), '-- Error: No checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/dcgan.nn')
    net_d = checkpoint['net_d']
    net_g = checkpoint['net_g']
    start = checkpoint['epoch']
else:
    net_g = model.Generator(config)
    net_d = model.Discriminator(config)  
    start = 1

torch.manual_seed(long(time.time()))
fixed = Variable(torch.Tensor(config.fixed_num, config.noise_dim))
fixed.data.normal_(0.0, 1.0)

opt_g = Adam(net_g.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
opt_d = Adam(net_d.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))

bce = torch.nn.BCELoss()

if use_cuda:
    net_d = net_d.cuda()
    net_g = net_g.cuda()
    fixed = fixed.cuda()
    torch.cuda.manual_seed(long(time.time()))
    cudnn.benchmark = True

def train(start, epoch, config):
    last_time = time.time()
    epoch_time = time.time()
    for idx, (image, _) in enumerate(loader):
        net_d.zero_grad()
        net_g.zero_grad()

        # Discriminator
        real = Variable(image)
        noise = Variable(torch.Tensor(config.batch_size, config.noise_dim))
        noise.data.normal_(0.0, 1.0)
        if use_cuda:
            real = real.cuda()
            noise = noise.cuda()

        fake = net_g(noise)
        fake_d = net_d(fake.detach())
        real_d = net_d(real)

        real_label = Variable(torch.ones(real_d.size()))
        fake_label = Variable(torch.zeros(fake_d.size()))
        if use_cuda:
            real_label = real_label.cuda()
            fake_label = fake_label.cuda()
        
        cost_d = bce(real_d, real_label) + bce(fake_d, fake_label)
        cost_d.backward()
        opt_d.step()

        # Generator
        fake_g = net_d(fake)

        real_label = Variable(torch.ones(fake_g.size()))
        if use_cuda:
            real_label = real_label.cuda()

        cost_g = bce(fake_g, real_label)
        cost_g.backward()
        opt_g.step()

        # Log
        if idx % config.log_frequency == 0:
            speed = time.time() - last_time
            last_time = time.time()
            format_str = ('Epoch: %d, Step: %d, G-Loss: %.3f, D-Loss: %.3f, Speed: %.2f sec/step')
            print(format_str % (epoch, idx, cost_g, cost_d, speed/config.log_frequency))

    # Saving Data
    fake_fixed = net_g(fixed)
    state = {
            'net_g': net_g,
            'net_d': net_d,
            'epoch': epoch,
        }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/dcgan.nn')
    if (epoch > 15 and epoch % config.save_frequency == 0) or (epoch <= 15):
        save_image(util.denorm(fake_fixed).data, '%s/fixed_%d.jpg'%(config.result_path, epoch))
        save_image(util.denorm(fake).data, '%s/fake_%d.jpg'%(config.result_path, epoch))
    print('-- Models and test images saved.')
    epoch_time = (time.time() - epoch_time)/60
    time_remain = (epoch_time * (config.final_epoch - epoch))/60
    print('-- Epoch completed. Epoch Time: %.2f min, Time Est: %.2f hour.' %(epoch_time, time_remain))

if not os.path.exists(config.result_path):
    os.makedirs(config.result_path)

util.print_network(net_g)
util.print_network(net_d)
print('-- Start Training')
for epoch in range(start, config.final_epoch):
    train(start, epoch, config)