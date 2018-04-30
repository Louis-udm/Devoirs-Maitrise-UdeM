"""NTM Read and Write Heads."""
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

# 用于分出k, β, g, s, γ或k, β, g, s, γ, e, a这几个参数
def _split_cols(mat, lengths):
    """Split a 2D matrix to variable length columns."""
    assert mat.size()[1] == sum(lengths), "Lengths must be summed to num columns"
    l = np.cumsum([0] + lengths)
    results = []
    for s, e in zip(l[:-1], l[1:]):
        results += [mat[:, s:e]]
    return results


'''
对一个已经训练好的MLP-NTM测试的结果
写入时:
read header的s基本不变，不移位(010)，g=1,说明读头内容寻址后，一直保持在这个位置
write header的s移位，(001),g=0.1，说明写头更多的参考上一个写地址，并移到下一个地址写入
读取时：
read header的s移位(001)，g=0,说明读头更多的参考上一个读地址，并移到下一个地址读取
write header的w此时已经弥散，g一般=1,s一般不移位。

说明读写时，NTM是选择地址寻址而非内容寻址。

'''
class NTMHeadBase(nn.Module):
    """An NTM Read/Write Head."""

    def __init__(self, memory, controller_size):
        """Initilize the read/write head.

        :param memory: The :class:`NTMMemory` to be addressed by the head.
        :param controller_size: The size of the internal representation.
        """
        super(NTMHeadBase, self).__init__()

        self.memory = memory
        self.N, self.M = memory.size()
        self.controller_size = controller_size

    def create_new_state(self, batch_size):
        raise NotImplementedError

    def register_parameters(self):
        raise NotImplementedError

    def is_read_head(self):
        return NotImplementedError

    def _address_memory(self, k, β, g, s, γ, w_prev):
        # Handle Activations
        k = k.clone()
        β = F.softplus(β)
        g = F.sigmoid(g)
        #Removed the unnecessary softplus in _address_memory
        # s = F.softmax(F.softplus(s), dim=1)
        s = F.softmax(s, dim=1)
        γ = 1 + F.softplus(γ)

        # print('---Head--k, β, g, s, γ--',k, β, g, s, γ)
        # print('---Head-- g, s--',g, s)

        w = self.memory.address(k, β, g, s, γ, w_prev)

        return w

#用controller的out的size也即隐藏层的size和memory建立
#aio.py->EncapsulatedNTM 生成header，ntm.py->NTM调用
class NTMReadHead(NTMHeadBase):
    def __init__(self, memory, controller_size):
        super(NTMReadHead, self).__init__(memory, controller_size)

        # Corresponding to k, β, g, s, γ sizes from the paper,设定各参数位数
        self.read_lengths = [self.M, 1, 1, 3, 1]
        #fc: fully connected layer
        self.fc_read = nn.Linear(controller_size, sum(self.read_lengths))
        self.reset_parameters()

    def create_new_state(self, batch_size):
        # The state holds the previous time step address weightings
        # w_prev用N个0初始化，也就是没有attention        
        return Variable(torch.zeros(batch_size, self.N))

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform(self.fc_read.weight, gain=1.4)
        nn.init.normal(self.fc_read.bias, std=0.01)

    def is_read_head(self):
        return True
        
    # 输入controller的100个输出(lstm的out部分由[真实输入x+pre_head]经controller运算后得到)
    # 对controller的输出做线性运算，获得k, β, g, s, γ
    # 然后使用这些参数以及w_prev，调用memory的address()，获得“读”要使用的w，
    # 最后使用获得的w，调用memory.read返回读到的内容和w
    # w_prev是ntm中的prev_head_stat,由ntm控制w_prev的保存,ntm吧当前head返回的w保存为w_prev
    # w_prev用N个0初始化，也就是没有attention
    def forward(self, embeddings, w_prev):
        """NTMReadHead forward function.

        :param embeddings: input representation of the controller.
        :param w_prev: previous step state
        """
        # print('----read Head-- ')
        
        o = self.fc_read(embeddings)
        k, β, g, s, γ = _split_cols(o, self.read_lengths)

        # Read from memory
        w = self._address_memory(k, β, g, s, γ, w_prev)
        r = self.memory.read(w)

        return r, w


class NTMWriteHead(NTMHeadBase):
    def __init__(self, memory, controller_size):
        super(NTMWriteHead, self).__init__(memory, controller_size)

        # Corresponding to k, β, g, s, γ, e, a sizes from the paper
        self.write_lengths = [self.M, 1, 1, 3, 1, self.M, self.M]
        self.fc_write = nn.Linear(controller_size, sum(self.write_lengths))
        self.reset_parameters()

    def create_new_state(self, batch_size):
        # w_prev用N个0初始化，也就是没有attention        
        return Variable(torch.zeros(batch_size, self.N))

    def reset_parameters(self):
        # Initialize the linear layers
        nn.init.xavier_uniform(self.fc_write.weight, gain=1.4)
        nn.init.normal(self.fc_write.bias, std=0.01)

    def is_read_head(self):
        return False

    # 输入controller的100个输出(lstm的out部分由[真实输入x+pre_head]经controller运算后得到)
    # 对controller的输出做线性运算，获得k, β, g, s, γ, e, a
    # 然后使用这些参数以及w_prev，调用memory的address()，获得“写”要使用的w,e,a
    # 最后使用获得的w，调用memory.write写入a并返回w
    # w_prev是ntm中的prev_head_stat,由ntm控制w_prev的保存,ntm吧当前head返回的w保存为w_prev
    # w_prev用N个0初始化，也就是没有attention
    def forward(self, embeddings, w_prev):
        """NTMWriteHead forward function.

        :param embeddings: input representation of the controller.
        :param w_prev: previous step state
        """
        # print('---write Head-- ')

        o = self.fc_write(embeddings)
        k, β, g, s, γ, e, a = _split_cols(o, self.write_lengths)

        # e should be in [0, 1]
        e = F.sigmoid(e)

        # Write to memory
        w = self._address_memory(k, β, g, s, γ, w_prev)

        # print('-- e, a, w--',e.size(), a.size(), w.size())

        self.memory.write(w, e, a)

        return w
