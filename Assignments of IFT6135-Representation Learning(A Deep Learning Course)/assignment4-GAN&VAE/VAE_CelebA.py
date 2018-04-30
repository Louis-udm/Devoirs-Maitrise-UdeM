#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 20:36:51 2018

@author: fanxiao
"""

import os
import matplotlib.pyplot as plt
from scipy.misc import imresize
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import itertools
#%%

# root path depends on your computer
root = '/Users/fanxiao/datasets/img_align_celeba/'
save_root = '/Users/fanxiao/datasets/resized_celebA/'
resize_size = 64

if not os.path.isdir(save_root):
    os.mkdir(save_root)
if not os.path.isdir(save_root + 'celebA'):
    os.mkdir(save_root + 'celebA')
img_list = os.listdir(root)

# ten_percent = len(img_list) // 10

for i in range(len(img_list)):
    img = plt.imread(root + img_list[i])
    img = imresize(img, (resize_size, resize_size))
    plt.imsave(fname=save_root + 'celebA/' + img_list[i], arr=img)

    if (i % 1000) == 0:
        print('%d images complete' % i)
        
#%%
class Encoder(torch.nn.Module):
    def __init__(self,input_size):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
                nn.Linear(input_size, 800),
                nn.ReLU()
                )
        self.input_size = input_size

    def forward(self, x):
        x = self.encoder(x)
        return x


class Decoder(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
                nn.Linear(input_size, 800),
                nn.ReLU(),
                nn.Linear(800,output_size),
                nn.Tanh()
                )

    def forward(self, x):
        return self.decoder(x)


class VAE(torch.nn.Module):

    def __init__(self, encoder, decoder, hidden_size=100):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.fc_mu = nn.Linear(400,hidden_size)
        self.fc_logvar = nn.Linear(400,hidden_size)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
#        x = self.encoder(x.view(-1, self.encoder.input_size))
        x = self.encoder(x)
        x = x.squeeze()
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        z.unsqueeze_(2)
        z.unsqueeze_(3)
        return self.decoder(z), mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
#    _,image_size = recon_x.size()
    loss = nn.MSELoss(size_average=False)
    BCE = loss(recon_x, x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def train(epoch, model, train_loader, loss_function, optimizer, log_interval):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if use_cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch, model, test_loader, loss_function):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if use_cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
#        if i == 0:
#            n = min(data.size(0), 8)
#            comparison = torch.cat([data[:n],
#                                  recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
#            save_image(comparison.data.cpu(),
#                     'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

#%%
# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(generator, self).__init__()
#        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0
        self.deconv1 = nn.ConvTranspose2d(100, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x))

        return x

class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(discriminator, self).__init__()
#        in_channels, out_channels, kernel_size, stride, padding
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1) # (64-4+2)/2+1 = 32
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1) # (32-4+2)/2+1= 16
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1) #(16-4+2)/2+1=8
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1) #(8-4+2)/2+1=4
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 400, 4, 1, 0) #(4-4)/1+1=1

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
        
#%%
def show_result(num_epoch, show = False, save = False, path = 'result.png'):
    z_ = Variable(torch.randn((5*5, 100)).view(-1, 100, 1, 1))
#    z_ = Variable(z_.cuda(), volatile=True)

    G.eval()
    test_images = G(z_)
    G.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla()
        ax[i, j].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
    
#%%
img_root = '/Users/fanxiao/datasets/resized_celebA/'
IMAGE_RESIZE = 64
train_sampler = range(2000)
BATCH_SIZE_TRAIN=100
#data_transform = transforms.Compose([
#    transforms.Resize((IMAGE_RESIZE, IMAGE_RESIZE), interpolation=Image.BILINEAR),
#    transforms.RandomHorizontalFlip(),
#    transforms.ToTensor(),
#    # transforms.Normalize(mean=[0], std=[1])
#    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
#                            0.229, 0.224, 0.225])
#])
data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
dataset = datasets.ImageFolder(root=img_root, transform=data_transform)

train_data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE_TRAIN, sampler=train_sampler, num_workers=10)


#%%

#encoder = Encoder(IMAGE_RESIZE*IMAGE_RESIZE)
#decoder = Decoder(100,IMAGE_RESIZE*IMAGE_RESIZE)
#model = VAE(encoder,decoder,100)
G = generator(128)
D = discriminator(128)
G.weight_init(mean=0.0, std=0.02)
D.weight_init(mean=0.0, std=0.02)
model = VAE(D,G)
#model = VAE()
use_cuda = torch.cuda.is_available()
if use_cuda:
    model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
EPOCHS=10

for epoch in range(1, EPOCHS + 1):
    train(epoch, model,train_data_loader,loss_function, optimizer, 20)
#    test(epoch, model, test_data_loader, loss_function)
#    sample = Variable(torch.randn(64, 20))
#    if use_cuda:
#        sample = sample.cuda()
#    sample = model.decode(sample).cpu()
#    save_image(sample.data.view(64, 1, 28, 28),
#               'results/sample_' + str(epoch) + '.png')
#%%
show_result(10, show=True,save=True, path='/Users/fanxiao/Google Drive/UdeM/IFT6135 Representation Learning/homework4/figures/result.pdf')