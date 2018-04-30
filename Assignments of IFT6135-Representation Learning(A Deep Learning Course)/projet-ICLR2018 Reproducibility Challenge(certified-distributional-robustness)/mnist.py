#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# # # #
# mnist.py
# @author Zhibin.LU
# @created Mon Apr 23 2018 17:19:42 GMT-0400 (EDT)
# @last-modified Sun Apr 29 2018 02:26:24 GMT-0400 (EDT)
# @website: https://louis-udm.github.io
# @description 
# # # #

#%%
import os
import importlib
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms
import torch.utils.data.sampler as sampler
import matplotlib.pyplot as plt

path="/Users/louis/Google Drive/M.Sc-DIRO-UdeM/IFT6135-Apprentissage de représentations/projet/"
if os.path.isdir(path):
    os.chdir(path)
else:
    os.chdir("./")
print(os.getcwd())

import  exp1
importlib.reload(exp1)

USE_CUDA=torch.cuda.is_available()
exp1.init_seed()

NO_CLASSES = 10
TRAIN_DATA_SIZE = 50000
TRAIN_EPOCH = 50 #10000
BATCH_SIZE = 128

MIN_LR0 = 0.001
MAX_LR0 = 0.001
GAMMA = 0.04 #0.01 #0.04 #0.01
#number of adversarial iterations
T_ADV = 15

'''
Load MNIST data
'''
mnist_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, transform=mnist_transforms, download=True)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, transform=mnist_transforms, download=True)
indices = list(range(len(mnist_train)))
np.random.shuffle(indices)
train_idx, valid_idx = indices[:TRAIN_DATA_SIZE], indices[TRAIN_DATA_SIZE:]
train_sampler = sampler.SubsetRandomSampler(train_idx)
valid_sampler = sampler.SubsetRandomSampler(valid_idx)
train_data_loader = torch.utils.data.DataLoader(
    mnist_train, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=10)
valid_data_loader = torch.utils.data.DataLoader(
    mnist_train, batch_size=BATCH_SIZE,  sampler=valid_sampler, num_workers=10)
test_data_loader = torch.utils.data.DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=10)
print('Loaded MNIST data, total',len(mnist_train)+len(mnist_test))

#? norm x?

#%%
'''
Arichitectur of estimateur for MNIST
'''
class Mnist_Estimateur(nn.Module):
    # initializers, d=num_filters
    def __init__(self, d=32, activation='relu'):
        super(Mnist_Estimateur, self).__init__()
        # in_channels, out_channels, kernel_size, stride, padding, dilation
        self.conv1 = nn.Conv2d(1, d, 8, 1, 0) # (28-8)+1 = 21
        self.conv2 = nn.Conv2d(d, d*2, 6, 1, 0) # (21-6)+1= 16
        self.conv3 = nn.Conv2d(d*2, d*4, 5, 1, 0) # (16-5)+1= 12
        self.fc1 = nn.Linear(18432,1024)
        self.fc2 = nn.Linear(1024,NO_CLASSES)
        if activation == 'relu':
            self.active = nn.ReLU() 
        else :
            self.active = nn.ELU()

    def init_weights(self, mean, std):
        for m in self._modules:
            if type(m) == nn.Linear:
                nn.init.xavier_uniform(m.weight)
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean, std)
                m.bias.data.zero_()


    def forward(self, input): 
        x = self.active(self.conv1(input))
        x = self.active(self.conv2(x))
        x = self.active( self.conv3(x) )
        x = x.view(x.size(0), -1)
        x = self.active(self.fc1(x))
        x = self.fc2(x)

        return x

#%%
if __name__=='__main__':

    loss_function=nn.CrossEntropyLoss()

    mnist_WRM=Mnist_Estimateur(activation='elu')
    if USE_CUDA:
        mnist_WRM=mnist_WRM.cuda()
    mnist_WRM.init_weights(mean=0.0, std=0.02)
    # optimizer = optim.Adam(mnist_WRM.parameters(), lr=LR0_MIN, betas=(0.5, 0.999))
    # optimizer = optim.RMSprop(mnist_WRM.parameters(), lr=LR0_MIN)
    optimizer = torch.optim.Adam(mnist_WRM.parameters(), lr=MIN_LR0)

    # exp1.train(mnist_WRM,optimizer,loss_function, train_data_loader,valid_data_loader, \
    #     TRAIN_EPOCH ,min_lr0=MIN_LR0,min_lr_adjust=False)

    exp1.train_WRM(mnist_WRM,optimizer,loss_function, train_data_loader,valid_data_loader, \
        TRAIN_EPOCH , GAMMA, max_lr0=MAX_LR0, min_lr0=MIN_LR0, min_lr_adjust=False, savepath='mnist_wrm_elu')
        
#%%
if __name__=='__main__':

    filename='mnist_wrm_elu_ep42'
    mnist_WRM,_=exp1.loadCheckpoint(mnist_WRM,filename)

    print('Accuracy on test data: ',exp1.evaluate(mnist_WRM,test_data_loader))
        