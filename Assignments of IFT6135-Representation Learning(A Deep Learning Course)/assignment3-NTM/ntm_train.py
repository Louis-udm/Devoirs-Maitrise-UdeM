#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
import os
os.chdir("/Users/louis/Google Drive/M.Sc-DIRO-UdeM/IFT6135-Apprentissage de représentations/assignment3/")
print(os.getcwd())

from ntm.aio import EncapsulatedNTM

import random
import time
import numpy as np

import os.path
import shutil
import time

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
import torch.autograd as autograd
import torch.distributions as distributions

from matplotlib import gridspec
import matplotlib.pyplot as plt



#%%


RANDOM_SEED = 2333
# REPORT_INTERVAL = 200
# CHECKPOINT_INTERVAL = 100 #1000
# CHECKPOINT_PATH='./checkpoint/'

controller_size = 100 
controller_layers = 1 
num_heads = 1 
memory_n = 128 #128
memory_m = 20 #20

rmsprop_lr = 3e-4 # 3e-4 #paper=3e-5
rmsprop_momentum = 0.9 
rmsprop_alpha = 0.95 
# BATCH_SIZE_TRAIN=25
# BATCH_SIZE_VALID=50
# LR_0=0.1 #4e-4
# MOMENTUM=0.9
# WEIGHT_DECAY=5e-4


INTERVAL=100 #100
TOTAL_BATCHES = 3000 #20000
#in each batch, there are batch_size sequences together as same length of sequence.
BYTE_WIDTH = 8 
BATCH_SIZE = 20 #50
SEQUENCE_MIN_LEN = 1 #1
SEQUENCE_MAX_LEN = 20 #20


loss_function = nn.BCELoss()

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


sample_binary=distributions.Bernoulli(torch.Tensor([0.5]))

cuda_available = False


def clip_grads(net):
    """Gradient clipping to the range [10, 10]."""
    parameters = list(filter(lambda p: p.grad is not None, net.parameters()))
    for p in parameters:
        p.grad.data.clamp_(-10, 10)


def get_ms():
    """Returns the current time in miliseconds."""
    return time.time() * 1000

def gen1seq():
    length=np.random.randint(2,SEQUENCE_MAX_LEN+1)
    # length=SEQ_SIZE+1
    seq=sample_binary.sample_n(9*length).view(length, 1, -1)
    seq[:,:,-1]=0.0
    seq[-1]=0.0
    seq[-1,-1,-1]=1.0
    return seq

def gen1seq_act(length):
    seq=torch.zeros(9*length).view(length, 1, -1)+0.5
    seq[:,:,-1]=0.0
    seq[-1]=0.0
    seq[-1,-1,-1]=1.0
    return seq

#total_batches: total of the batch 
#batch_size: in each batch, there are batch_size sequences together as same length of sequence.
def dataloader(total_batches,
               batch_size,
               seq_width,
               min_len,
               max_len):
    """Generator of random sequences for the copy task.

    Creates random batches of "bits" sequences.

    All the sequences within each batch have the same length.
    The length is [`min_len`, `max_len`]

    :param num_batches: Total number of batches to generate.
    :param seq_width: The width of each item in the sequence.
    :param batch_size: Batch size.
    :param min_len: Sequence minimum length.
    :param max_len: Sequence maximum length.

    return: batch_num,Xs(seq_len+1,Byte_size+1),Ys(seq_len,Byte_size),act_seqs(seq_len,Byte_size+1)
    act_seqs is for the action of copy!
    
    NOTE: The input width is `seq_width + 1`, the additional input
    contain the delimiter.
    """
    for batch_num in range(total_batches):

        # All batches have the same sequence length
        seq_len = random.randint(min_len, max_len)
        seq = np.random.binomial(1, 0.5, (seq_len, batch_size, seq_width))
        seq = Variable(torch.from_numpy(seq))

        # The input includes an additional channel used for the delimiter
        inp = Variable(torch.zeros(seq_len + 1, batch_size, seq_width + 1))
        inp[:seq_len, :, :seq_width] = seq
        inp[seq_len, :, seq_width] = 1.0 # delimiter in our control channel
        outp = seq.clone()

        seq2 = Variable(torch.zeros(seq_len, batch_size, seq_width)+0.5)
        act_inp = Variable(torch.zeros(seq_len, batch_size, seq_width + 1))
        act_inp[:seq_len, :, :seq_width] = seq2

        yield batch_num+1, inp.float(), outp.float(), act_inp.float()

def train_model(model,criterion,optimizer, seqs_loader, interval=500):
    
    # in_seqs=[ gen1seq() for i in range(SEQS_TOTAL)]
    
    # print(inputs[0])
    if cuda_available:
        model = model.cuda()

    # optimizer = optim.RMSprop(model.parameters(), lr=rmsprop_lr, momentum = MOMENTUM)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 200], gamma=0.1)

    #training
    print('Training Begining, %d batches (batch_size=%d)...' % \
                (TOTAL_BATCHES, BATCH_SIZE))
    start_ms = get_ms()
    
    list_losses =[]
    list_costs =[]
    list_bits=[]
    list_seq_num=[]
    losses=0
    costs=0
    lengthes=0
    for batch_num, X, Y, act in seqs_loader:
        # start = time.time()
        if cuda_available:
            X, Y, act = X.cuda(), Y.cuda(), act.cuda()
        
        model.init_sequence(BATCH_SIZE)
        optimizer.zero_grad()

        inp_seq_len = X.size(0)
        # sequence_len, batch_size, byte_size
        outp_seq_len, _, _ = Y.size()

        # Feed the sequence + delimiter
        for i in range(inp_seq_len):
            model(X[i])

        # Read the output (no input given)
        y_out = Variable(torch.zeros(Y.size()))
        for i in range(outp_seq_len):
            y_out[i], _ = model()

        loss = criterion(y_out, Y)
        loss.backward()
        clip_grads(model)
        optimizer.step()

        list_losses.append(loss.data[0])

        out_binarized = y_out.clone().data.numpy()
        out_binarized=np.where(out_binarized>0.5,1,0)
        # The cost is the number of error bits per sequence
        cost = np.sum(np.abs(out_binarized - Y.data.numpy()))

        losses+=loss
        costs+=cost
        lengthes+=BATCH_SIZE
        # end = time.time()

        if (batch_num) % INTERVAL==0 :
            list_costs.append(costs/INTERVAL/BATCH_SIZE) #per sequence
            list_losses.append(losses.data[0]/INTERVAL/BATCH_SIZE)
            list_seq_num.append(lengthes) # per thousand
            mean_time = ((get_ms() - start_ms) / INTERVAL) / BATCH_SIZE
            print ("Batch %d th, loss %f, cost %f, Time %.3f ms/sequence." % (batch_num, list_losses[-1], list_costs[-1], mean_time) )
            
            costs = 0
            losses = 0
            start_ms = get_ms()

    return list_losses,list_costs,list_seq_num

def evaluate(model,criterion,optimizer, test_data_loader) : 

    if cuda_available:
        model = model.cuda()

    costs = 0
    losses = 0
    lengthes = 0
    # optimizer = optim.RMSprop(model.parameters(), lr=rmsprop_lr, momentum = MOMENTUM)

    for batch_num, X, Y, act in test_data_loader:
        # start = time.time()
        if cuda_available:
            X, Y, act = X.cuda(), Y.cuda(), act.cuda()
        
        model.init_sequence(BATCH_SIZE)
        optimizer.zero_grad()

        inp_seq_len = X.size(0)
        outp_seq_len, _, _ = Y.size()

        # Feed the sequence + delimiter
        for i in range(inp_seq_len):
            model(X[i])

        # Read the output (no input given)
        y_out = Variable(torch.zeros(Y.size()))
        for i in range(outp_seq_len):
            y_out[i], _ = model()

        loss = criterion(y_out, Y)
        loss.backward()
        clip_grads(model)
        optimizer.step()

        lengthes+=BATCH_SIZE
    
        losses += loss
        
        out_binarized = y_out.clone().data.numpy()
        out_binarized=np.where(out_binarized>0.5,1,0)
        # The cost is the number of error bits per sequence
        cost = np.sum(np.abs(out_binarized - Y.data.numpy()))

        costs += cost
        
    print ("T = %d, Average loss %f, average cost %f" % (Y.size(0), losses.data[0]/lengthes, costs/lengthes))
    return losses.data/lengthes, costs/lengthes

def saveCheckpoint(model,list_batch_num,list_loss, list_cost, path='ntm') :
    print('Saving..')
    state = {
        'model': model,
        'list_batch_num': list_batch_num,
        'list_loss' : list_loss,
        'list_cost' : list_cost
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/'+path)

def loadCheckpoint(path='ntm'):
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+path)
    model = checkpoint['model']
    list_batch_num = checkpoint['list_batch_num']
    list_loss = checkpoint['list_loss']
    list_cost = checkpoint['list_cost']
    return model, list_batch_num, list_loss, list_cost


train_loader=dataloader(TOTAL_BATCHES, BATCH_SIZE,
                    BYTE_WIDTH,
                    SEQUENCE_MIN_LEN, SEQUENCE_MAX_LEN)

criterion=nn.BCELoss()


#%%

'''
Train EncapsulatedNTM
Use LSTM Controller
'''

model = EncapsulatedNTM(BYTE_WIDTH + 1, BYTE_WIDTH,
                              controller_size, controller_layers,
                              num_heads,
                              memory_n, memory_m,controller_type ='lstm')

optimizer=optim.RMSprop(model.parameters(),
                             momentum=rmsprop_momentum,
                             alpha=rmsprop_alpha,
                             lr=rmsprop_lr)
print('Total params of Model EncapsulatedNTM with LSTM controller :',model.calculate_num_params())
list_loss,list_cost,list_seq_num=train_model(model,loss_function,optimizer,train_loader)

saveCheckpoint(model,list_seq_num,list_loss, list_cost, path='lstm-ntm-uniform') 

#%%
model, list_seq_num, list_loss, list_cost = loadCheckpoint(path='lstm-ntm-uniform')

plt.figure()
# plt.plot(range(0,TOTAL_BATCHES),list_cost,label='NTM')
plt.plot(list_seq_num,list_cost)
# plt.plot(range(1,1000),valid_acc_list,label='Validation')
plt.xlabel('Sequence number')
plt.ylabel('Cost per sequence')
plt.legend()
plt.savefig('lstm-ntm-uniform.pdf')
plt.show()

# #%%
# list_avg_loss = []
# list_avg_cost = []
# list_T_num = []
# for T in range(10,110,10) : 
#     test_data_loader = dataloader(TOTAL_BATCHES, BATCH_SIZE,
#                     BYTE_WIDTH,min_len=T,max_len=T)
#     avg_loss, avg_cost = evaluate(model,loss_function,optimizer,test_data_loader)
#     list_avg_loss.append(avg_loss)
#     list_avg_cost.append(avg_cost)
#     list_T_num.append(T)

# saveCheckpoint(model,list_T_num,list_avg_loss, list_avg_cost, path='ntm1-Ts') 

# #%%
# model, list_T_num, list_avg_loss, list_avg_cost = loadCheckpoint(path='ntm1-Ts')
    
# plt.plot(list_T_num,list_avg_cost)
# plt.xlabel('T')
# plt.ylabel('average cost')
# plt.savefig('ntm-cost-T.pdf')

# #%%
# test1=Variable(gen1seq())
# print(test1)
# model.init_sequence(1)
# model.forward(test1)
# # model.init_hidden(1)
# print(model.forward())

#%%
'''
Train EncapsulatedNTM
Use MLP Controller
'''
model = EncapsulatedNTM(BYTE_WIDTH + 1, BYTE_WIDTH,
                              controller_size, controller_layers,
                              num_heads,
                              memory_n, memory_m,controller_type ='mlp')

optimizer=optim.RMSprop(model.parameters(),
                             momentum=rmsprop_momentum,
                             alpha=rmsprop_alpha,
                             lr=rmsprop_lr)
print('Total params of Model EncapsulatedNTM with MLP controller:',model.calculate_num_params())
list_loss,list_cost,list_seq_num=train_model(model,loss_function,optimizer,train_loader)

saveCheckpoint(model,list_seq_num,list_loss, list_cost, path='mlp-ntm-uniform') 

#%%
model, list_seq_num, list_loss, list_cost = loadCheckpoint(path='mlp-ntm-uniform')

plt.figure()
# plt.plot(range(0,TOTAL_BATCHES),list_cost,label='NTM')
plt.plot(list_seq_num,list_cost)
# plt.plot(range(1,1000),valid_acc_list,label='Validation')
plt.xlabel('Sequence number')
plt.ylabel('Cost per sequence')
plt.legend()
plt.savefig('mlp-NTM-uniform.pdf')
plt.show()

# #%%
# list_avg_loss = []
# list_avg_cost = []
# list_T_num = []
# for T in range(10,110,10) : 
#     test_data_loader = dataloader(TOTAL_BATCHES, BATCH_SIZE,
#                     BYTE_WIDTH,min_len=T,max_len=T)
#     avg_loss, avg_cost = evaluate(model,loss_function,optimizer,test_data_loader)
#     list_avg_loss.append(avg_loss)
#     list_avg_cost.append(avg_cost)
#     list_T_num.append(T)

# saveCheckpoint(model,list_T_num,list_avg_loss, list_avg_cost, path='mlp-ntm1-Ts') 

# #%%
# model, list_T_num, list_avg_loss, list_avg_cost = loadCheckpoint(path='mlp-ntm1-Ts')
    
# plt.plot(list_T_num,list_avg_cost)
# plt.xlabel('T')
# plt.ylabel('average cost')
# plt.savefig('mlp-ntm-cost-T.pdf')

#%%
'''
Visualize
'''

#%%
model, list_seq_num, list_loss, list_cost = loadCheckpoint(path='mlp-ntm-uniform') #lstm-ntm.bak, mlp-ntm.bak

def evaluate_single_batch(net, criterion, X, Y):
    """Evaluate a single batch (without training)."""
    inp_seq_len = X.size(0)
    outp_seq_len, batch_size, _ = Y.size()

    # New sequence
    net.init_sequence(batch_size)

    # Feed the sequence + delimiter
    states = []
    for i in range(inp_seq_len):
        o, state = net(X[i])
        states += [state]

    # Read the output (no input given)
    print('---start to copy---')
    y_out = Variable(torch.zeros(Y.size()))
    for i in range(outp_seq_len):
        # x2 = Variable(torch.Tensor(batch_size, 9).uniform_(0,1))
        x2=Variable(torch.zeros(batch_size, 9)+0.9)
        # x2=Variable(torch.Tensor([0,1,0,0,1,0,1,0,0]).repeat(batch_size, 1))
        y_out[i], state = net(x2)
        # y_out[i], state = net()
        states += [state]

    loss = criterion(y_out, Y)

    y_out_binarized = y_out.clone().data
    y_out_binarized.apply_(lambda x: 0 if x < 0.5 else 1)

    # The cost is the number of error bits per sequence
    cost = torch.sum(torch.abs(y_out_binarized - Y.data))

    result = {
        'loss': loss.data[0],
        'cost': cost / batch_size,
        'y_out': y_out,
        'y_out_binarized': y_out_binarized,
        'states': states
    }

    return result

def visualize_read_write(X,result,N,i_th) :
    T, batch_size, num_bits = X.size()
    T = T - 1
    num_bits = num_bits - 1
    
    plt.figure(figsize=(8, 6)) 
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 3]) 
    
    ax = plt.subplot(gs[0,0])
    y_in = torch.cat((X[:,i_th,:].data,torch.zeros(T,num_bits+1)),dim=0)
    ax.imshow(torch.t(y_in), cmap='gray',aspect='auto')
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    ax.set_title('inputs')
    
    ax = plt.subplot(gs[0,1])
    y_out = torch.cat((torch.zeros(T+1,num_bits),result['y_out_binarized'][:,i_th,:]),dim=0)
    y_out = torch.cat((y_out,torch.zeros(2*T+1,1)),dim=1)
    ax.imshow(torch.t(y_out), cmap='gray',aspect='auto')
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    ax.set_title('outputs')
    
    states = result['states']
    read_state = torch.zeros(len(states),N)  # read weight
    write_state = torch.zeros(len(states),N) # write weight
    for i in range(0,len(states)) :
        reads, controller_state, heads_states = states[i]
        read_state[i,:] = heads_states[0][i_th].data
        write_state[i,:] = heads_states[1][i_th].data
        
        
    ax = plt.subplot(gs[1,0])
    ax.imshow(torch.t(write_state), cmap='gray',aspect='auto') #[:,90:]
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    #ax.text(1,40,'Time', fontsize=11)
    # ax.text(6,41,'Write Weightings',fontsize=12)
    ax.set_title('Write Weightings')
    #ax.arrow(6,60,60, fc="k", ec="k", head_width=0.5, head_length=1, color='w')
    #ax.annotate('Time', xy=(0.4, -0.1), xycoords='axes fraction', xytext=(0, -0.1),
    #            arrowprops=dict(arrowstyle="->", color='black'))
    #ax.annotate('Location', xy=(-0.2, 0.4), xycoords='axes fraction', xytext=(-0.26, 0), 
    #            arrowprops=dict(arrowstyle="->", color='black'))
    
    
    ax = plt.subplot(gs[1,1])
    ax.imshow(torch.t(read_state), cmap='gray',aspect='auto') #[:,90:]
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    #ax.text(1,40,'Time', fontsize=11)
    ax.set_title('Read Weightings')
    # ax.text(6,41,'Read Weightings',fontsize=12)
    
    plt.tight_layout()
    plt.savefig('visualization.pdf')
    plt.show() 

train_loader=dataloader(1, 1,
                    BYTE_WIDTH,
                    SEQUENCE_MAX_LEN, SEQUENCE_MAX_LEN)

for batch_num, X, Y, act in train_loader:
    re=evaluate_single_batch(model,criterion,X,Y)
    visualize_read_write(X,re,memory_n,0)
    
    break