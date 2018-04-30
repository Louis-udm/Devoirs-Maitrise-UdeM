# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 19:26:06 2018

@author: fanxiao
"""

#%%
import os
import time
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

# path = 'C:/Users/lingyu.yue/Documents/Xiao_Fan/GAN'
path="/Users/louis/Google Drive/M.Sc-DIRO-UdeM/IFT6135-Apprentissage de représentations/assignment4/"
if os.path.isdir(path):
    os.chdir(path)
else:
    os.chdir("./")
print(os.getcwd())

#%%
from inception_score import inception_score
from inception_score import inception_score2

#%%
IMAGE_RESIZE = 64
use_cuda = torch.cuda.is_available()
torch.manual_seed(999)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(999)
    
#%%
'''
# Pretraite the images data to 64x64

# root path depends on your computer
# root = 'C:/Users/lingyu.yue/Documents/Xiao_Fan/GAN/img_align_celeba/img_align_celeba/'
root="/Users/louis/Google Drive/M.Sc-DIRO-UdeM/IFT6135-Apprentissage de représentations/assignment4/img_align_celeba/img_align_celeba/'
# save_root = 'C:/Users/lingyu.yue/Documents/Xiao_Fan/GAN/img_align_celeba/resized_celebA/' 
save_root = '/Users/louis/Google Drive/M.Sc-DIRO-UdeM/IFT6135-Apprentissage de représentations/assignment4/img_align_celeba/resized_celebA/' 

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
'''
#%%

def train(generator, discriminator, G_optimizer, D_optimizer,train_data_loader, Loss_fun, num_epochs, hidden_size=100, critic=1, score=False,savepath='GAN') :

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['Inc_score'] = []
    train_hist['total_ptime'] = []
    
    print('Training start!')
    start_time = time.time()
    
    if score : 
        test_z = torch.randn(10000,generator.hidden_size,1,1)
        
    for epoch in range(num_epochs):
        D_losses = []
        G_losses = []
    
        # learning rate decay
        if (epoch+1) == 11:
            G_optimizer.param_groups[0]['lr'] /= 10
            D_optimizer.param_groups[0]['lr'] /= 10
            print("learning rate change!")
    
        if (epoch+1) == 16:
            G_optimizer.param_groups[0]['lr'] /= 10
            D_optimizer.param_groups[0]['lr'] /= 10
            print("learning rate change!")
    
        num_iter = 0
    
        epoch_start_time = time.time()
        for x_, _ in train_data_loader:
            
            #For stability, update discriminator several times before updating generator
            D_train_loss_sum = 0
            mini_batch = x_.size()[0]
    
            y_real_ = torch.ones(mini_batch)
            y_fake_ = torch.zeros(mini_batch)
            
            if use_cuda :
                x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
            else :
                x_, y_real_, y_fake_ = Variable(x_), Variable(y_real_), Variable(y_fake_)
            
            # generator.eval()
            # discriminator.train()
            for n in range(critic) :
                
                # train discriminator D : maximize E[log(D(x))]+E[log(1-D(G(z)))], minimize -[]
                discriminator.zero_grad()
        

                D_result = discriminator(x_).squeeze() #(batch,100,1,1) => (batch,100)
                D_real_loss = Loss_fun(D_result, y_real_) #-log(D(x)) BEC_loss = -(ylogx+(1-y)log(1-x))
                D_real_loss.backward()

                z_ = torch.randn((mini_batch, hidden_size)).view(-1, hidden_size, 1, 1)
                if use_cuda : 
                    z_ = Variable(z_.cuda())
                else :
                    z_ = Variable(z_)
                G_result = generator(z_)
        
                D_result = discriminator(G_result).squeeze()
                D_fake_loss = Loss_fun(D_result, y_fake_) #-log(1-D(G(z)))
                D_fake_loss.backward()
                
                D_optimizer.step()

                # D_train_loss = D_real_loss + D_fake_loss
                # D_train_loss.backward()
                # D_optimizer.step()
                # D_train_loss_sum += D_train_loss.data[0]

                D_train_loss_sum += D_real_loss.data[0]+D_fake_loss.data[0]
    
            D_losses.append(D_train_loss_sum/critic)
    
            # train generator G : maximize E[log(D(G(z)))], minimize -[]
            # discriminator.eval()
            # generator.train()
            generator.zero_grad()
    
            z_ = torch.randn((mini_batch, hidden_size)).view(-1, hidden_size, 1, 1)
            if use_cuda : 
                z_ = Variable(z_.cuda())
            else :
                z_ = Variable(z_)
    
            G_result = generator(z_)
            D_result = discriminator(G_result).squeeze()
            G_train_loss = Loss_fun(D_result, y_real_) #-log(1-D(G(z)))
            G_train_loss.backward()
            G_optimizer.step()
    
            G_losses.append(G_train_loss.data[0])
    
            num_iter += 1
    
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
    
    
        print('[%d/%d], loss_D: %.3f, loss_G: %.3f - critic: %d, ptime: %.2fs' % ((epoch + 1), num_epochs, torch.mean(torch.FloatTensor(D_losses)),
                                                                  torch.mean(torch.FloatTensor(G_losses)), critic, per_epoch_ptime))
#        p = 'CelebA_DCGAN_results/Random_results/CelebA_DCGAN_' + str(epoch + 1) + '.png'
#        fixed_p = 'CelebA_DCGAN_results/Fixed_results/CelebA_DCGAN_' + str(epoch + 1) + '.png'
#        show_result((epoch+1), save=True, path=p, isFix=False)
#        show_result((epoch+1), save=True, path=fixed_p, isFix=True)
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
        
        if score :
            # score=inception_score(test_z,G,D,128,cuda=use_cuda,splits=10)
            score=inception_score(test_z, generator, batch_size=128, cuda=use_cuda, resize=True, splits=10)
            print("Inception score: ",score)
            train_hist['Inc_score'].append(score)
    
        if (epoch+1) % 3==0:
            end_time = time.time()
            total_ptime = end_time - start_time
            train_hist['total_ptime'].append(total_ptime)
            saveCheckpoint(generator,discriminator,train_hist,savepath+'_ep'+str(epoch+1),use_cuda)
        
    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)
    
    print("Avg per epoch ptime: %.2fs, total %d epochs ptime: %.2fs" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), num_epochs, total_ptime))
    print("Training finish!")
    return  train_hist

def train2(generator, discriminator, G_optimizer, D_optimizer,train_data_loader, Loss_fun, num_epochs, hidden_size=100, critic_max=10, score=False,savepath='GAN') :

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['Inc_score'] = []
    train_hist['total_ptime'] = []
    
    print('Training start!')
    start_time = time.time()
    
    if score : 
        test_z = torch.randn(10000,generator.hidden_size,1,1)
        
    for epoch in range(num_epochs):
        D_losses = []
        G_losses = []
    
        # learning rate decay
        if (epoch+1) == 11:
            G_optimizer.param_groups[0]['lr'] /= 10
            D_optimizer.param_groups[0]['lr'] /= 10
            print("learning rate change!")
    
        if (epoch+1) == 16:
            G_optimizer.param_groups[0]['lr'] /= 10
            D_optimizer.param_groups[0]['lr'] /= 10
            print("learning rate change!")
    
        num_iter = 0
    
        epoch_start_time = time.time()
        for x_, _ in train_data_loader:
            
            #For stability, update discriminator several times before updating generator
            mini_batch = x_.size()[0]
    
            y_real_ = torch.ones(mini_batch)
            y_fake_ = torch.zeros(mini_batch)
            
            if use_cuda :
                x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
            else :
                x_, y_real_, y_fake_ = Variable(x_), Variable(y_real_), Variable(y_fake_)
            
            # generator.eval()
            # discriminator.train()
            D_loss_sum = 0
            for n in range(1,critic_max+1) :
                
                # train discriminator D : maximize E[log(D(x))]+E[log(1-D(G(z)))], minimize -[]
                discriminator.zero_grad()
        

                D_result = discriminator(x_).squeeze() #(batch,100,1,1) => (batch,100)
                D_real_loss = Loss_fun(D_result, y_real_) #-log(D(x)) BEC_loss = -(ylogx+(1-y)log(1-x))
                D_real_loss.backward()

                z_ = torch.randn((mini_batch, hidden_size)).view(-1, hidden_size, 1, 1)
                if use_cuda : 
                    z_ = Variable(z_.cuda())
                else :
                    z_ = Variable(z_)
                G_result = generator(z_)
        
                D_result = discriminator(G_result).squeeze()
                D_fake_loss = Loss_fun(D_result, y_fake_) #-log(1-D(G(z)))
                D_fake_loss.backward()
                
                D_optimizer.step()

                # gradiant penalty is work, for bilinear model, [-0.05,0.05] is bettre than [-0.01,0.01]
                for p in discriminator.parameters():
                    p.data.clamp_(-0.05, 0.05)

                # D_train_loss = D_real_loss + D_fake_loss
                # D_train_loss.backward()
                # D_optimizer.step()
                # D_train_loss_sum += D_train_loss.data[0]

                D_loss_sum += (D_real_loss.data[0]+D_fake_loss.data[0])/2

                if (D_real_loss.data[0]+D_fake_loss.data[0])/2<0.55: break  #-log0.5=0.693
    
            D_losses.append(D_loss_sum/n)
            str_critic=str(n)
    
            # train generator G : maximize E[log(D(G(z)))], minimize -[]
            # discriminator.eval()
            # generator.train()
            G_loss_sum=0
            for n in range(1,critic_max+1) :
                generator.zero_grad()
        
                z_ = torch.randn((mini_batch, hidden_size)).view(-1, hidden_size, 1, 1)
                if use_cuda : 
                    z_ = Variable(z_.cuda())
                else :
                    z_ = Variable(z_)
        
                G_result = generator(z_)
                D_result = discriminator(G_result).squeeze()
                G_train_loss = Loss_fun(D_result, y_real_) #-log(1-D(G(z)))
                G_train_loss.backward()
                G_optimizer.step()

                G_loss_sum += G_train_loss.data[0]

                if G_train_loss.data[0]<0.55: break  #-log0.5=0.693
    
            G_losses.append(G_loss_sum/n)
            str_critic=str_critic+':'+str(n)
    
            num_iter += 1
    
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
    
    
        print('[%d/%d], loss_D %.3f, loss_G %.3f - critic %s, ptime %.2fs' % ((epoch + 1), num_epochs, torch.mean(torch.FloatTensor(D_losses)),
                                                                  torch.mean(torch.FloatTensor(G_losses)), str_critic, per_epoch_ptime))
#        p = 'CelebA_DCGAN_results/Random_results/CelebA_DCGAN_' + str(epoch + 1) + '.png'
#        fixed_p = 'CelebA_DCGAN_results/Fixed_results/CelebA_DCGAN_' + str(epoch + 1) + '.png'
#        show_result((epoch+1), save=True, path=p, isFix=False)
#        show_result((epoch+1), save=True, path=fixed_p, isFix=True)
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
        
        if score :
            # score=inception_score(test_z,G,D,128,cuda=use_cuda,splits=10)
            score=inception_score(test_z, generator, batch_size=128, cuda=use_cuda, resize=True, splits=10)
            print("Inception score: ",score)
            train_hist['Inc_score'].append(score)

        # if (epoch+1) % 3==0:
        end_time = time.time()
        total_ptime = end_time - start_time
        train_hist['total_ptime'].append(total_ptime)
        saveCheckpoint(generator,discriminator,train_hist,savepath+'_ep'+str(epoch+1),use_cuda)
        
    
    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)
    
    print("Avg per epoch ptime: %.2fs, total %d epochs ptime: %.2fs" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), num_epochs, total_ptime))
    print("Training finish!")
    return  train_hist

def train3(generator, discriminator, G_optimizer, D_optimizer,train_data_loader, Loss_fun, num_epochs, hidden_size=100, critic_max=15, score=False,savepath='GAN') :

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['Inc_score'] = []
    train_hist['total_ptime'] = []
    
    print('Training start!')
    start_time = time.time()
    
    if score : 
        test_z = torch.randn(10000,generator.hidden_size,1,1)

    # ini_threshold=0.8 #0.8->0.3, -log0.5=0.693
        
    for epoch in range(num_epochs):
        D_losses = []
        G_losses = []
    
        # learning rate decay
        if (epoch+1) == 11:
            G_optimizer.param_groups[0]['lr'] /= 10
            D_optimizer.param_groups[0]['lr'] /= 10
            print("learning rate change!")
    
        if (epoch+1) == 16:
            G_optimizer.param_groups[0]['lr'] /= 10
            D_optimizer.param_groups[0]['lr'] /= 10
            print("learning rate change!")
            
        # threshold=ini_threshold-epoch*(ini_threshold-0.3)/num_epochs
    
        num_iter = 0
    
        epoch_start_time = time.time()
        for x_, _ in train_data_loader:
            
            #For stability, update discriminator several times before updating generator
            mini_batch = x_.size()[0]
    
            y_real_ = torch.ones(mini_batch)
            y_fake_ = torch.zeros(mini_batch)
            
            if use_cuda :
                x_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_real_.cuda()), Variable(y_fake_.cuda())
            else :
                x_, y_real_, y_fake_ = Variable(x_), Variable(y_real_), Variable(y_fake_)
            
            # generator.eval()
            # discriminator.train()
            D_loss_sum = 0
            for n in range(1,critic_max+1) :
                
                # train discriminator D : maximize E[log(D(x))]+E[log(1-D(G(z)))], minimize -[]
                discriminator.zero_grad()
        

                D_result = discriminator(x_).squeeze() #(batch,100,1,1) => (batch,100)
                D_real_loss = Loss_fun(D_result, y_real_) #-log(D(x)) BEC_loss = -(ylogx+(1-y)log(1-x))
                D_real_loss.backward()

                z_ = torch.randn((mini_batch, hidden_size)).view(-1, hidden_size, 1, 1)
                if use_cuda : 
                    z_ = Variable(z_.cuda())
                else :
                    z_ = Variable(z_)
                G_result = generator(z_)
        
                D_result = discriminator(G_result).squeeze()
                D_fake_loss = Loss_fun(D_result, y_fake_) #-log(1-D(G(z)))
                D_fake_loss.backward()
                
                D_optimizer.step()

                # D_train_loss = D_real_loss + D_fake_loss
                # D_train_loss.backward()
                # D_optimizer.step()
                # D_train_loss_sum += D_train_loss.data[0]

                D_loss_sum += (D_real_loss.data[0]+D_fake_loss.data[0])/2

                # if (D_real_loss.data[0]+D_fake_loss.data[0])/2<threshold: break  
                if len(G_losses)>0 and G_losses[-1]>(D_real_loss.data[0]+D_fake_loss.data[0])/2: break
    
            D_losses.append(D_loss_sum/n)
            str_critic=str(n)
    
            # train generator G : maximize E[log(D(G(z)))], minimize -[]
            # discriminator.eval()
            # generator.train()
            G_loss_sum=0
            for n in range(1,critic_max+1) :
                generator.zero_grad()
        
                z_ = torch.randn((mini_batch, hidden_size)).view(-1, hidden_size, 1, 1)
                if use_cuda : 
                    z_ = Variable(z_.cuda())
                else :
                    z_ = Variable(z_)
        
                G_result = generator(z_)
                D_result = discriminator(G_result).squeeze()
                G_train_loss = Loss_fun(D_result, y_real_) #-log(1-D(G(z)))
                G_train_loss.backward()
                G_optimizer.step()

                G_loss_sum += G_train_loss.data[0]

                # if G_train_loss.data[0]<threshold: break  
                if D_losses[-1]>G_train_loss.data[0]: break
    
            G_losses.append(G_loss_sum/n)
            str_critic=str_critic+':'+str(n)
    
            num_iter += 1
    
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
    
    
        print('[%d/%d], loss_D %.3f, loss_G %.3f - critic %s, ptime %.2fs' % ((epoch + 1), num_epochs, torch.mean(torch.FloatTensor(D_losses)),
                                                                  torch.mean(torch.FloatTensor(G_losses)), str_critic, per_epoch_ptime))
#        p = 'CelebA_DCGAN_results/Random_results/CelebA_DCGAN_' + str(epoch + 1) + '.png'
#        fixed_p = 'CelebA_DCGAN_results/Fixed_results/CelebA_DCGAN_' + str(epoch + 1) + '.png'
#        show_result((epoch+1), save=True, path=p, isFix=False)
#        show_result((epoch+1), save=True, path=fixed_p, isFix=True)
        train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
        train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
        
        if score :
            # score=inception_score(test_z,G,D,128,cuda=use_cuda,splits=10)
            score=inception_score(test_z, generator, batch_size=128, cuda=use_cuda, resize=True, splits=10)
            print("Inception score: ",score)
            train_hist['Inc_score'].append(score)
        
        if (epoch+1) % 3==0:
            end_time = time.time()
            total_ptime = end_time - start_time
            train_hist['total_ptime'].append(total_ptime)
            saveCheckpoint(generator,discriminator,train_hist,savepath+'_ep'+str(epoch+1),use_cuda)
        
    
    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)
    
    print("Avg per epoch ptime: %.2fs, total %d epochs ptime: %.2fs" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), num_epochs, total_ptime))
    print("Training finish!")
    return  train_hist

def train_W(generator, discriminator, G_optimizer, D_optimizer,train_data_loader, Loss_fun='NLLLoss', num_epochs=10, hidden_size=100, critic_max=15, score=False,savepath='GAN') :

    train_hist = {}
    train_hist['D_losses'] = []
    train_hist['G_losses'] = []
    train_hist['per_epoch_ptimes'] = []
    train_hist['Inc_score'] = []
    train_hist['total_ptime'] = []
    
    print('Training start!')
    start_time = time.time()
    
    if score : 
        test_z = torch.randn(10000,generator.hidden_size,1,1)

    # critic_max=1
    # ini_threshold=0.8 #0.8->0.3, -log0.5=0.693
        
    for epoch in range(num_epochs):
            
        # threshold=ini_threshold-epoch*(ini_threshold-0.3)/num_epochs
    
        num_iter = 0
    
        D_losses = 0
        G_losses = 0
        epoch_start_time = time.time()
        for x_, _ in train_data_loader:
            
            #For stability, update discriminator several times before updating generator
            mini_batch = x_.size()[0]
    
            one = torch.FloatTensor([1])
            mone = one * -1
            
            if use_cuda :
                x_, one, mone = Variable(x_.cuda()), Variable(one.cuda()), Variable(mone.cuda())
            else :
                x_, one, mone = Variable(x_), Variable(one), Variable(mone)
            
            # generator.eval()
            # discriminator.train()
            D_loss_sum = 0
            for n in range(1,critic_max+1) :
                
                discriminator.zero_grad()
        
                #mean(f(x))
                D_result_r = discriminator(x_) #(batch,100,1,1) => (batch,100)
                
                #grandiant mean(-f(x))
                D_result_r.backward(mone)

                z_ = torch.randn((mini_batch, hidden_size)).view(-1, hidden_size, 1, 1)
                if use_cuda : 
                    z_ = Variable(z_.cuda())
                else :
                    z_ = Variable(z_)
                G_result = generator(z_)

                #mean(f(z))
                D_result_f = discriminator(G_result)
                
                #grandiant mean(f(z))
                D_result_f.backward(one)

                D_optimizer.step()
                
                for p in discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

                '''
                Wasserstein distance
                max( mean(f(x)) - mean(f(z)) )
                '''
                # D_loss_sum +=np.abs(D_result_r.data[0]-D_result_f.data[0])
                # D_loss=np.abs(D_result_r.data[0]-D_result_f.data[0])
                D_loss=D_result_r.data[0]-D_result_f.data[0]

                # if (D_real_loss.data[0]+D_fake_loss.data[0])/2<threshold: break  
    
            # D_losses+=D_loss_sum/n
            D_losses+=D_loss
            str_critic=str(n)
    
            G_loss_sum=0
            for n in range(1,1+1) :
                generator.zero_grad()
        
                z_ = torch.randn((mini_batch, hidden_size)).view(-1, hidden_size, 1, 1)
                if use_cuda : 
                    z_ = Variable(z_.cuda())
                else :
                    z_ = Variable(z_)
        
                G_result = generator(z_)
                D_result = discriminator(G_result)
                
                D_result.backward(mone)
                G_optimizer.step()
                G_loss_sum += D_result.data[0]
                # if G_train_loss.data[0]<threshold: break  
    
            G_losses+=G_loss_sum/n
            str_critic=str_critic+':'+str(n)
    
            num_iter += 1
    
        epoch_end_time = time.time()
        per_epoch_ptime = epoch_end_time - epoch_start_time
    
    
        print('[%d/%d], loss_D %.3f, loss_G %.3f - critic %s, ptime %.2fs' % \
                ((epoch + 1), num_epochs, D_losses/num_iter,
                G_losses/num_iter, str_critic, per_epoch_ptime))
        train_hist['D_losses'].append(D_losses/num_iter)
        train_hist['G_losses'].append(G_losses/num_iter)
        train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
        
        # if score :
        #     # score=inception_score(test_z,G,D,128,cuda=use_cuda,splits=10)
        #     score=inception_score(test_z, generator, discriminator, batch_size=128, cuda=use_cuda, resize=True, splits=10)
        #     print("Inception score: ",score)
        #     train_hist['Inc_score'].append(score)
        
        if (epoch+1) % 3==0:
            end_time = time.time()
            total_ptime = end_time - start_time
            train_hist['total_ptime'].append(total_ptime)
            saveCheckpoint(generator,discriminator,train_hist,savepath+'_ep'+str(epoch+1),use_cuda)
        
    
    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)
    
    print("Avg per epoch ptime: %.2fs, total %d epochs ptime: %.2fs" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), num_epochs, total_ptime))
    print("Training finish!")
    return  train_hist

def saveCheckpoint(generator,discriminator,train_hist, path='GAN', use_cuda=True) :
    print('Saving..')
    state = {
        'generator':  generator.cpu().state_dict() if use_cuda else generator.state_dict(),
        'discriminator': discriminator.cpu().state_dict() if use_cuda else discriminator.state_dict(),
        'train_hist' : train_hist
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/'+path)
    if use_cuda:
        generator.cuda()
        discriminator.cuda()

def loadCheckpoint(path='GAN', hidden_size = 100, use_cuda=True,dir='./checkpoint/'):
    dtype = torch.FloatTensor
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(dir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(dir+path)
    generator_params = checkpoint['generator']
    discriminator_params = checkpoint['discriminator']
    G = generator(128,hidden_size)
    G.load_state_dict(generator_params)
    D = discriminator(128)
    D.load_state_dict(discriminator_params)
    if use_cuda :
        G.cuda()
        D.cuda()
    train_hist = checkpoint['train_hist']

    return G,D,train_hist

def loadCheckpoint_Upsampling(path='GAN', hidden_size = 100, use_cuda=True,mode='nearest',dir='./checkpoint/'):
    dtype = torch.FloatTensor
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(dir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(dir+path)
    generator_params = checkpoint['generator']
    discriminator_params = checkpoint['discriminator']
    if mode=='nearest':
        G = generator_Upsampling(128, hidden_size,'nearest')
    else:
        G=generator_Upsampling(128, hidden_size,'bilinear')
    G.load_state_dict(generator_params)
    D = discriminator(128)
    D.load_state_dict(discriminator_params)
    if use_cuda :
        G.cuda()
        D.cuda()
    train_hist = checkpoint['train_hist']

    return G,D,train_hist
    
def loadCheckpoint_Upsampling_old(path='GAN', hidden_size = 100, use_cuda=True,mode='nearest',dir='./checkpoint/'):
    dtype = torch.FloatTensor
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(dir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(dir+path)
    generator_params = checkpoint['generator']
    discriminator_params = checkpoint['discriminator']
    if mode=='nearest':
        G = generator_Upsampling_old(128, hidden_size,'nearest')
    else:
        G=generator_Upsampling_old(128, hidden_size,'bilinear')
    G.load_state_dict(generator_params)
    D = discriminator(128)
    D.load_state_dict(discriminator_params)
    if use_cuda :
        G.cuda()
        D.cuda()
    train_hist = checkpoint['train_hist']

    return G,D,train_hist

def loadCheckpoint_W(path='GAN', hidden_size = 100, use_cuda=True,dir='./checkpoint/'):
    dtype = torch.FloatTensor
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(dir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(dir+path)
    generator_params = checkpoint['generator']
    discriminator_params = checkpoint['discriminator']
    G = generator(128,hidden_size)
    G.load_state_dict(generator_params)
    D = discriminator_W(128)
    D.load_state_dict(discriminator_params)
    if use_cuda :
        G.cuda()
        D.cuda()
    train_hist = checkpoint['train_hist']

    return G,D,train_hist

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


# G(z)
class generator(nn.Module):
    '''
    Deconvolution generator (transposed convolution) with paddings and strides
    '''
    # initializers
    def __init__(self, d=128, hidden_size=100):
        super(generator, self).__init__()
        # nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0
        self.deconv1 = nn.ConvTranspose2d(hidden_size, d*8, 4, 1, 0) #1->4
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1) #4->8
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1) #8->16
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1) #16->32
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, 3, 4, 2, 1) #32->64
        
        self.hidden_size = hidden_size

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input): #input (batch,hidden_size,1,1)
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.tanh(self.deconv5(x)) #output (batch,3,64,64)

        return x
        
class generator_Upsampling(nn.Module):
    # initializers
    def __init__(self, d=128, hidden_size=100, mode='nearest'):
        super(generator_Upsampling, self).__init__()
        self.upsampling1 = nn.Upsample(scale_factor=4,mode=mode) #1->4 input(batch,100,1,1)=>(batch,100,4,4)
        self.conv1 = nn.Conv2d(hidden_size, d*8, 4, 2, 3) # => (batch,d*4,4,4)=>(batch,d*8,4,4)   (4-4+3*2)/2+1=4
        self.conv1_bn = nn.BatchNorm2d(d*8)
        self.upsampling2 = nn.Upsample(scale_factor=4,mode=mode) #=>(batch,d*8,16,16)
        self.conv2 = nn.Conv2d(d*8, d*4, 4, 2, 1) #=>(batch,d*4,8,8)
        self.conv2_bn = nn.BatchNorm2d(d*4)
        self.upsampling3 = nn.Upsample(scale_factor=4,mode=mode) #=>(batch,d*4,32,32)
        self.conv3 = nn.Conv2d(d*4, d*2, 4, 2, 1) #=>(batch,d*2,16,16)
        self.conv3_bn = nn.BatchNorm2d(d*2)
        self.upsampling4 = nn.Upsample(scale_factor=4,mode=mode) #=>(batch,d*2,64,64)
        self.conv4 = nn.Conv2d(d*2, d, 4, 2, 1) #=>(batch,d,32,32)
        self.conv4_bn = nn.BatchNorm2d(d)
        self.upsampling5 = nn.Upsample(scale_factor=4,mode=mode) #=>(batch,d,128,128)
        self.conv5 = nn.Conv2d(d, 3, 4, 2, 1)  #=>(batch,3,64,64)

        self.hidden_size = hidden_size

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        x = F.relu(self.conv1_bn(self.conv1(self.upsampling1(input))))
        x = F.relu(self.conv2_bn(self.conv2(self.upsampling2(x))))
        x = F.relu(self.conv3_bn(self.conv3(self.upsampling3(x))))
        x = F.relu(self.conv4_bn(self.conv4(self.upsampling4(x))))
        x = F.tanh(self.conv5(self.upsampling5(x)))

        return x

class generator_Upsampling_old(nn.Module):
    '''
    Nearest-Neighbor/Bilinear Upsampling followed by regular convolution
    '''
    # initializers
    def __init__(self, d=128, hidden_size=100, mode='nearest'):
        super(generator_Upsampling_old, self).__init__()
        self.upsampling1 = nn.Upsample(scale_factor=4,mode=mode) #1->4 input(batch,100,1,1)=>(batch,100,4,4)
        self.conv1 = nn.Conv2d(hidden_size, d*8, 3, 1, 1) # => (batch,d*8,4,4) (110)(4-1+0)/1+1 (4-k+2p)/s+1 (4-3+2)/1+1
        self.conv1_bn = nn.BatchNorm2d(d*8)
        self.upsampling2 = nn.Upsample(scale_factor=2,mode=mode) #=>(batch,d*8,8,8)
        self.conv2 = nn.Conv2d(d*8, d*4, 3, 1, 1) #=>(batch,d*4,8,8)
        self.conv2_bn = nn.BatchNorm2d(d*4)
        self.upsampling3 = nn.Upsample(scale_factor=2,mode=mode) #=>(batch,d*4,16,16)
        self.conv3 = nn.Conv2d(d*4, d*2, 3, 1, 1) #=>(batch,d*2,16,16)
        self.conv3_bn = nn.BatchNorm2d(d*2)
        self.upsampling4 = nn.Upsample(scale_factor=2,mode=mode) #=>(batch,d*2,32,32)
        self.conv4 = nn.Conv2d(d*2, d, 3, 1, 1) #=>(batch,d,32,32)
        self.conv4_bn = nn.BatchNorm2d(d)
        self.upsampling5 = nn.Upsample(scale_factor=2,mode=mode) #=>(batch,d,64,64)
        self.conv5 = nn.Conv2d(d, 3, 3, 1, 1)  #=>(batch,3,64,64)
        
        self.hidden_size = hidden_size

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input):
        # x = F.relu(self.deconv1(input))
        x = F.relu(self.conv1_bn(self.conv1(self.upsampling1(input))))
        x = F.relu(self.conv2_bn(self.conv2(self.upsampling2(x))))
        x = F.relu(self.conv3_bn(self.conv3(self.upsampling3(x))))
        x = F.relu(self.conv4_bn(self.conv4(self.upsampling4(x))))
        x = F.tanh(self.conv5(self.upsampling5(x)))

        return x


class discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(discriminator, self).__init__()
        # in_channels, out_channels, kernel_size, stride, padding
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1) # (64-4+2)/2+1 = 32
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1) # (32-4+2)/2+1= 16
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1) #(16-4+2)/2+1=8
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1) #(8-4+2)/2+1=4
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0) #(4-4)/1+1=1 =>(batch,1,1,1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input): #input (batch,3,64,64)
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x)) #output (batch,1,1,1)

        return x


# discriminator of Wasserstein
class discriminator_W(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(discriminator_W, self).__init__()
        # in_channels, out_channels, kernel_size, stride, padding
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1) # (64-4+2)/2+1 = 32
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1) # (32-4+2)/2+1= 16
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1) #(16-4+2)/2+1=8
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1) #(8-4+2)/2+1=4
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0) #(4-4)/1+1=1 =>(batch,1,1,1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input): #input (batch,3,64,64)
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = self.conv5(x) #output (batch,1,1,1)
        # mean of the batch, we get a single scale finally.
        x = x.mean(0)

        return x.view(1)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
        

def show_result(G,D,num_epoch, hidden_size = 100, show = False, save = False, path = 'result.png',use_cuda=True):
    z_ = torch.randn((5*5, hidden_size)).view(-1, hidden_size, 1, 1)
    if use_cuda : 
        z_ = Variable(z_.cuda())
    else : 
        z_ = Variable(z_)
        # z_ = Variable(z_.cuda(), volatile=True)

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
    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def compareRandomPoint(x_0,x_1,G,chooseRandn, hidden_size = 100, show = False, save = False, path = 'result.png',use_cuda=True,label='model'):

    for i in range(chooseRandn):
        z_0 = torch.randn(hidden_size)
    #因为seed设置后，要选择需要的图片，就得做几次随机选取
    for i in range(chooseRandn*2):
        z_1 = torch.randn(hidden_size)

    z_prime=torch.zeros(11,hidden_size)
    for n in range(0,11):
        z_prime[n]=n*0.1*z_0+(1-n*0.1)*z_1
    z_prime=z_prime.view(-1, hidden_size, 1, 1)

    if use_cuda : 
        z_prime = Variable(z_prime.cuda())
    else : 
        z_prime = Variable(z_prime)

    x_prime=torch.zeros(11,3,64,64)
    for n in range(0,11):
        x_prime[n]=n*0.1*x_0+(1-n*0.1)*x_1
    # x_prime=x_prime.view().view(-1, hidden_size, 1, 1)


    G.eval()
    test_images = G(z_prime)
    G.train()

    fig, ax = plt.subplots(2, 11, figsize=(10, 2))
    for i, j in itertools.product(range(2), range(11)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(11):
        ax[0, k].cla()
        #原图像是(x+0.5u)/0.5sigma,这里转变回去
        ax[0, k].imshow((x_prime[k].numpy().transpose(1, 2, 0) + 1) / 2)

    for k in range(11):
        ax[1, k].cla()
        ax[1, k].imshow((test_images[k].cpu().data.numpy().transpose(1, 2, 0) + 1) / 2)

    fig.text(0.5, 0.04, label, ha='center')
    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

#%%

if __name__ == '__main__':
    
    train_epoch = 40
    hidden_dim = 100
    img_root = "img_align_celeba/resized_celebA/"

    IMAGE_RESIZE = 64
    train_sampler = range(2000)

    train_epoch = 40
    batch_size = 128
    lr = 0.001
    hidden_dim = 100
    use_cuda = torch.cuda.is_available()

    data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset = datasets.ImageFolder(root=img_root, transform=data_transform)

    #%%
    G,D,train_hist = loadCheckpoint('GANDeconvolution_t4000_h100_ep20',hidden_dim,use_cuda=use_cuda)

    show_result(G,D,train_epoch, hidden_dim, show=True,save=True, path='figures/result_Deconvolution.pdf', use_cuda=use_cuda)

    test_z = torch.randn(10000,100,1,1)
    inception_score(test_z, G, batch_size=128, cuda=use_cuda, resize=True, splits=10)
    

    #from inception_score import inception_score
    #for ep in range(100) :
    #    imgs = torch.randn(100,hidden_dim,1,1)    
    #imgs = G(Variable(imgs.cuda()))



    #%%
    '''
    Compare 3 method for increase/double the feature map size:
    '''

    x = dataset[0][0]
    print ("image size : ", x.size())
    plt.imshow((x.numpy().transpose(1, 2, 0)*0.5 +0.5))

    #%%
    x = dataset[0][0]
    x = Variable(x.view(1,3,64,64))
    deconv = nn.ConvTranspose2d(3, 3, 4, 2, 1)
    deconv.weight.data.normal_(mean=0.0, std=0.02)
    deconv.bias.data.zero_()
    deconv_x = deconv(x).squeeze(0)
    print ('Deconvolution :' , deconv_x.size())
    plt.imshow((deconv_x.data.numpy().transpose(1, 2, 0) + 1) / 2)

    #%%
    
    x = dataset[0][0]

    print ("image size : ", x.size())
    plt.subplot(2,2,1)
    plt.imshow((x.numpy().transpose(1, 2, 0) + 1) / 2)

    x = Variable(x.view(1,3,64,64))
    conv = nn.Conv2d(3,10,4,2,1)
    conv_x = conv(x) # (batch,10,32,32)

    # Deconvolution (transposed convolution) with paddings and strides.
    deconv = nn.ConvTranspose2d(10,3,4,2,1)
    #deconv.weight.data.normal_(mean=0.0, std=0.05)
    #deconv.bias.data.zero_()

    deconv_x = deconv(conv_x).squeeze(0)
    print ("deconv_x size : ", deconv_x.size())
    plt.subplot(2,2,2)
    plt.imshow((deconv_x.data.numpy().transpose(1, 2, 0) + 1) / 2)

    # Nearest-Neighbor Upsampling followed by regular convolution.
    plt.subplot(2,2,3)
    upsampling_nearest = nn.Upsample(scale_factor=2,mode='nearest')
    conv2 = nn.Conv2d(10, 3, 1, 1, 0)
    upsampling_nearest_x = conv2(upsampling_nearest(conv_x)).squeeze(0)
    print ("nearest_x size : ", upsampling_nearest_x.size())
    plt.imshow((upsampling_nearest_x.data.numpy().transpose(1, 2, 0) + 1) / 2)

    # Bilinear Upsampling followed by regular convolution
    plt.subplot(2,2,4)
    upsampling_bilinear = nn.Upsample(scale_factor=2,mode='bilinear')
    upsampling_bilinear_x = conv2(upsampling_bilinear(conv_x)).squeeze(0)
    print ("bilinear_x size : ", upsampling_bilinear_x.size())
    plt.imshow((upsampling_bilinear_x.data.numpy().transpose(1, 2, 0) + 1) / 2)
    # plt.savefig('/Users/fanxiao/Google Drive/UdeM/IFT6135 Representation Learning/homework4/figures/faces.pdf')
    plt.savefig('faces.pdf')