# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 18:08:31 2018

@author: lingyu.yue
"""

'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torch.utils.data.sampler as sampler

import os
#import argparse

#from models import *
from torch.autograd import Variable
import vgg
#%%
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
resume = True
best_acc = 0

cuda_available = torch.cuda.is_available()
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

# Model
if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/vgg16')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    net = vgg.VGG('VGG16')
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()

if cuda_available:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR_0, momentum=0.9, weight_decay=5e-4)
#optimizer = optim.Adam(net.parameters(), lr=LR_0, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4)

# Training
def train(ep,display=1):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_data_loader):
        if cuda_available:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        train_acc = correct/total
        adjust_lr(optimizer,ep,EPOCH_NUM)

#        utils.progress_bar(batch_idx, len(train_data_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    if display!=0 and ep % display==0:
        print('Epoch : %d, Train Acc : %.3f' % (ep, 100.*train_acc))
#            print('--------------------------------------------------------------')
    return train_acc
    

def test(ep, display=1):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(valid_data_loader):
        if cuda_available:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        test_acc = correct/total
#
#        utils.progress_bar(batch_idx, len(valid_data_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    if display!=0 and ep % display==0:
        print('Epoch : %d, Test Acc : %.3f' % (ep, 100.*test_acc))
        print('--------------------------------------------------------------')

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if cuda_available else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/vgg16')
        best_acc = acc
        
    return test_acc

def adjust_lr(optimizer, epoch, total_epochs):
    lr = LR_0 * (0.1 ** (epoch / float(total_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if resume==False : 
    train_acc_list = []
    valid_acc_list = []
for epoch in range(50, EPOCH_NUM):
    train_acc_list.append(train(epoch))
    valid_acc_list.append(test(epoch))
    
#%%
plt.figure()
plt.plot(range(1,EPOCH_NUM),train_acc_list,label='Training')
plt.plot(range(1,EPOCH_NUM),valid_acc_list,label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('problem2-accuracy1.pdf')
plt.show()