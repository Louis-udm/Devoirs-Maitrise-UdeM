#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function

#%%

import random
import time
import numpy as np

import os
import os.path
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torch.utils.data.sampler as sampler
import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image
#from tqdm import *

import matplotlib.pyplot as plt

#set path and load data
# os.chdir("/Users/louis/Google Drive/M.Sc-DIRO-UdeM/IFT6135-Apprentissage de représentations/assignment2")
# os.chdir("/Users/fanxiao/Google Drive/UdeM/IFT6135 Representation Learning/homework1/programming part ")
print(os.getcwd())


'''
# Set hyper parameters.
'''
IMAGES_ROOT = 'datasets/PetImages'
MY_TEST_ROOT = 'datasets/my_test'
MODEL_SAVE_PATH='output'
MODEL_SAVE_INTERVAL = 5

# hyper parameters of Deep learning
IMAGE_RESIZE=64
TRAIN_SIZE = 20000
VALID_SIZE = 4990
EPOCH_NUM = 100
BATCH_SIZE_TRAIN=25
BATCH_SIZE_VALID=50
LR_0=0.1
MOMENTUM=0.9
WEIGHT_DECAY=5e-4

cuda_available=True

#%%
'''
# Load the dog_vs_cat data
'''

indices = list(range(TRAIN_SIZE+VALID_SIZE))
np.random.seed(123)
np.random.shuffle(indices)

train_idx, valid_idx = indices[:TRAIN_SIZE], indices[TRAIN_SIZE:]

train_sampler = sampler.SubsetRandomSampler(train_idx)
valid_sampler = sampler.SubsetRandomSampler(valid_idx)

data_transform = transforms.Compose([
    transforms.Resize((IMAGE_RESIZE, IMAGE_RESIZE), interpolation=Image.BILINEAR),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0], std=[1])
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                            0.229, 0.224, 0.225])
])
dataset = datasets.ImageFolder(root=IMAGES_ROOT, transform=data_transform)

train_data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE_TRAIN, sampler=train_sampler, num_workers=10)

valid_data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=BATCH_SIZE_VALID,  sampler=valid_sampler, num_workers=10)

my_testset = datasets.ImageFolder(root=MY_TEST_ROOT, transform=data_transform)

my_data_loader = torch.utils.data.DataLoader(
    my_testset, batch_size=BATCH_SIZE_VALID, shuffle=False, num_workers=10)

print('Loaded images(train and valid), total',len(dataset))
print('Loaded my test images, total',len(my_testset))
print('Image size: ', dataset[0][0].size())

#%%
'''
a single Residual Block
'''
class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Conv Layer 1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Conv Layer 2
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=(3, 3), stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
    
        # Shortcut connection to downsample residual
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=(1, 1), stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

'''
# Define the Model A.
'''
class Model_A(nn.Module):
    def __init__(self):
        super(Model_A, self).__init__()

        # Initial input conv
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=(3, 3),
            stride=1, padding=1, bias=False #outpou64x64
        )
        self.bn1 = nn.BatchNorm2d(64)
        
        # Create stages 1-4
        self.stage1 = self._create_stage(64, 64, stride=1) #64x64
        self.stage2 = self._create_stage(64, 128, stride=2) #32x32 
        self.stage3 = self._create_stage(128, 256, stride=2) #16x16
        self.stage4 = self._create_stage(256, 512, stride=2) #8x8
        self.linear = nn.Linear(2048, 2)
    
    # A stage is just two residual blocks for ResNet18
    def _create_stage(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels, 1)
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = F.avg_pool2d(out, 4) #2x2
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


    def adjust_lr(self, optimizer, epoch, total_epochs):
        lr = LR_0 * (0.1 ** (epoch / float(total_epochs)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def init_weights_zero(self,m):
        if type(m) == nn.Linear:
            nn.init.constant(m.weight, 0.0)
    #         print(m.weight)

    def init_weights_normal(self,m):
        if type(m) == nn.Linear:
            nn.init.normal(m.weight)
            # print(m.weight)

    def init_weights_glorot(self,m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform(m.weight)
    #         print(m.weight)

    
    def save_model(self, ep):
        torch.save(self, '%s/modela_save_%d' % (MODEL_SAVE_PATH, ep))


    def evaluate(self, dataset_loader):

        # Evaluate,This has any effect only on modules such as Dropout or BatchNorm.
        self.eval()
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(dataset_loader):
            if cuda_available:
                inputs, targets = inputs.cuda(), targets.cuda()

            inputs, targets = Variable(inputs, volatile=True), Variable(targets, volatile=True)
            outputs = self(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
        return correct/total

    def predict(self, dataset_loader,display=False):

        self.eval()
        results=torch.Tensor()
        # for batch_idx, (inputs, targets) in enumerate(tqdm(dataset_loader)):
        for batch_idx, (inputs, targets) in enumerate(dataset_loader):
            if cuda_available:
                inputs, targets = inputs.cuda(), targets.cuda()

            inputs, targets = Variable(inputs, volatile=True), Variable(targets, volatile=True)
            outputs = self(inputs)
            _, predicted = torch.max(outputs.data, 1)

            if display:
                for x,t,p in zip(inputs.data,targets.data,predicted):
                    print(t,p)
                    img = transforms.ToPILImage()(x)
                    # img.show()

                    plt.imshow(img)
                    plt.show()

            if batch_idx==0:
                results=predicted
            else:
                torch.cat((results, predicted),0)

        return results

    def train_model(self, display=3):
        
        train_acc_list = list()
        valid_acc_list = list()

        if cuda_available:
            self = self.cuda()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=LR_0, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 200], gamma=0.1)
        
        for ep in range(EPOCH_NUM):

            # This has any effect only on modules such as Dropout or BatchNorm.
            self.train()

            losses = []
            scheduler.step()
            # Train
            start = time.time()
            # for batch_idx, (inputs, targets) in enumerate(tqdm(train_data_loader)):
            for batch_idx, (inputs, targets) in enumerate(train_data_loader):
                if cuda_available:
                    inputs, targets = inputs.cuda(), targets.cuda()
                
                # img = transforms.ToPILImage(inputs)

                optimizer.zero_grad()
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                losses.append(loss.data[0])
            end = time.time()

            # print the process of the training.  
            # if display!=0 and ep % display==0:
            #     print('Epoch : %d Loss : %.3f Time : %.3f seconds ' % (ep, np.mean(losses), end - start))
            
            # Evaluate,This has any effect only on modules such as Dropout or BatchNorm.
            self.eval()

            train_acc=self.evaluate(train_data_loader)
            train_acc_list.append(train_acc)
            valid_acc=self.evaluate(valid_data_loader)
            valid_acc_list.append(valid_acc)

            # print the process of the training.  
            if display!=0 and ep % display==0:
                print('Epoch : %d, Train Acc : %.3f, Test Acc : %.3f, Spend:%.3f minutes' % (ep, 100.*train_acc, 100.*valid_acc,(end-start)/60.0))
                print('--------------------------------------------------------------')

            if ep % MODEL_SAVE_INTERVAL == 0:
                self.save_model(ep)

        return train_acc_list, valid_acc_list



#%%
'''
Train Model A
'''
model = Model_A()
train_acc_list, valid_acc_list=model.train_model()

print('Accuracy of model A:')
print(train_acc_list)
print(valid_acc_list)

#%%
#plot accuracy as a function of epoch
plt.figure()
plt.plot(range(1,EPOCH_NUM+1),train_acc_list,label='Training')
plt.plot(range(1,EPOCH_NUM+1),valid_acc_list,label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('problem2-accuracy1.pdf')
plt.show()

#%%
# model = torch.load(MODEL_SAVE_PATH)
model.predict(my_data_loader,display=True)

#%%

'''
# Define the Model B.
'''
class Model_B(Model_A):
    def __init__(self):
        super(Model_B, self).__init__()

        # Initial input conv
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=(3, 3),
            stride=1, padding=1, bias=False #outpou64x64
        )
        self.bn1 = nn.BatchNorm2d(64)
        
        # Create stages 1-4
        self.stage1 = self._create_stage(64, 64, stride=1) #64x64
        self.stage2 = self._create_stage(64, 128, stride=2) #32x32 
        self.stage3 = self._create_stage(128, 256, stride=2) #16x16
        self.stage4 = self._create_stage(256, 512, stride=2) #8x8
        self.stage5 = self._create_stage(512, 1024, stride=2) #4x4
        self.linear = nn.Linear(1024, 2)
    
    # A stage is just two residual blocks for ResNet18
    def _create_stage(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels, 1)
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.stage5(out)
        out = F.avg_pool2d(out, 4) #1x1
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


    def save_model(self,ep):
        torch.save(self, '%s/modelb_save_%d' % (MODEL_SAVE_PATH, ep))

#%%
'''
Train Model B
'''
modelb = Model_B()
train_acc_list, valid_acc_list=modelb.train_model()

print('Accuracy of model B:')
print(train_acc_list)
print(valid_acc_list)

#%%
#plot accuracy as a function of epoch
plt.figure()
plt.plot(range(1,EPOCH_NUM+1),train_acc_list,label='Training')
plt.plot(range(1,EPOCH_NUM+1),valid_acc_list,label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('problem2-accuracy2.pdf')
plt.show()

#%%
# model = torch.load(MODEL_SAVE_PATH)
modelb.predict(my_data_loader,display=True)
