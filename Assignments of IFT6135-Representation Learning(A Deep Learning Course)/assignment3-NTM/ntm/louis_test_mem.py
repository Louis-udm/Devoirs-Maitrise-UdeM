
#%%
import os

os.chdir("/Users/louis/Google Drive/M.Sc-DIRO-UdeM/IFT6135-Apprentissage de représentations/assignment3/ref1-pytorch-ntm/")
print(os.getcwd())

import pytest
import torch
from torch.autograd import Variable
from ntm.memory import NTMMemory

def _t(*l):
    return Variable(torch.Tensor(l)).unsqueeze(0)

#%%
#Louis test
mem = NTMMemory(4,4)
mem.reset(batch_size=1)
test1=TestMemoryReadWrite()

w=[
   _t(1, 0, 0, 0),
_t(0, 1, 0, 0),
_t(0, 0, 1, 0),
_t(0, 0, 0, 1),
_t(1, 0, 0, 0),
_t(0, 1, 0, 0),
_t(0, 0, 1, 0),
_t(0, 0, 0, 1),
_t(0.5, 0.5, 0,0) 
]
e=[
    _t(1, 1, 1, 1),
_t(1, 1, 1, 1),
_t(1, 1, 1, 1),
_t(1, 1, 1, 1),
_t(0, 1, 1, 1),
_t(0, 0, 0, 0),
_t(0, 0, 0, 0),
_t(0, 0, 0, 0.5),
_t(1, 1, 1, 1)
]
a=[
    _t(1, 0, 0, 0),
_t(0, 1, 0, 0),
_t(0, 0, 1, 0),
_t(0, 0, 0, 1),
_t(0, 1, 1, 1),
_t(0, 0, 1, 0),
_t(0, 0, 0, 0),
_t(0, 0, 0, 0.2),
_t(0, 0, 0, 0)
]
expected=[
    _t(1, 0, 0, 0),
_t(0, 1, 0, 0),
_t(0, 0, 1, 0),
_t(0, 0, 0, 1),
_t(1, 1, 1, 1),
_t(0, 1, 1, 0),
_t(0, 0, 1, 0),
_t(0, 0, 0, 0.7),
_t(0.25, 0.5, 0.5, 0.25)
]

#%%
print('**memory orignial:**\n',mem.memory)
print('w:\n',w[-1])
print('e:\n',e[-1])
print('a:\n',a[-1])

#意思：删除w行e列，加入w行a列,值为a
mem.write(w[-1], e[-1], a[0])
print('**after write**\n',mem.memory)

#%%
print('**memory orignial:**\n',mem.memory)
print('**read:**,w:\n',w[-1])
result = mem.read(w[-1])
print('**result**\n',result)

#%%
k=_t(0.99, 0.001, 0.01, 0) #定义希望找到的类似的内容
beta=_t(100)
g=_t(0.9) #1选择由k选出来的,0直接使用w_prev
shift=_t(0, 1, 0)  #010不位移
gamma=_t(100)
w_prev=_t(0.1, 0.6, 0.3, 0.2)
# expected= _t(1, 0, 0, 0)
print('**memory orignial:**\n',mem.memory)
#最后获得的地址w
m1=mem.address(k, beta, g, shift, gamma, w_prev)
print('**get address:**\n',m1)