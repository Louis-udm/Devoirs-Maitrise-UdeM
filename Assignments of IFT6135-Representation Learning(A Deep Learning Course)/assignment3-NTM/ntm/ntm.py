#!/usr/bin/env python
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class NTM(nn.Module):
    """A Neural Turing Machine."""
    def __init__(self, num_inputs, num_outputs, controller, memory, heads):
        """Initialize the NTM.

        注意：外部输入的向量的维度同内存块尺寸不一定相等
        :param num_inputs: External number of inputs.外部输入维度，比如9位
        :param num_outputs: External number of outputs.外部输出维度，比如8位
        :param controller: :class:`LSTMController`
        :param memory: :class:`NTMMemory` 大小NxM, 共有batch_size个内存块
        :param heads: list of :class:`NTMReadHead` or :class:`NTMWriteHead`

        Note: This design allows the flexibility of using any number of read and
              write heads independently, also, the order by which the heads are
              called in controlled by the user (order in list)
        """
        super(NTM, self).__init__()

        # Save arguments
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller = controller
        self.memory = memory
        self.heads = heads

        self.N, self.M = memory.size()
        _, self.controller_size = controller.size()

        # Initialize the initial previous read values to random biases
        # 随机值初始化读头的previous read参数, 这个就是head读出的内容
        # init_r就是第一个prev_reads, prev_reads大小=M*读写头个数
        # 维度为read heads个数 x M (在下面create_new_state中扩展到batch_size)
        self.num_read_heads = 0
        self.init_r = []
        for head in heads:
            if head.is_read_head():
                init_r_bias = Variable(torch.randn(1, self.M) * 0.01) #*0.01 ?????
                self.register_buffer("read{}_bias".format(self.num_read_heads), init_r_bias.data)
                self.init_r += [init_r_bias]
                self.num_read_heads += 1

        assert self.num_read_heads > 0, "heads list must contain at least a single read head"

        # Initialize a fully connected layer to produce the actual output:
        #   [controller_output; previous_reads ] -> output
        self.fc = nn.Linear(self.controller_size + self.num_read_heads * self.M, num_outputs)
        self.reset_parameters()

    def create_new_state(self, batch_size):
        #aio.py中初始化，
        #init_r就是第一个prev_reads,这里扩展到batch_size，维度最后为heads x batch_size x M
        init_r = [r.clone().repeat(batch_size, 1) for r in self.init_r]
        controller_state = self.controller.create_new_state(batch_size)
        #heads_state就是读写头的地址寻址w，用0初始化
        heads_state = [head.create_new_state(batch_size) for head in self.heads]
        # init_r就是prev_reads,head读出的内容
        return init_r, controller_state, heads_state

    def reset_parameters(self):
        # Initialize the linear layer
        nn.init.xavier_uniform(self.fc.weight, gain=1)
        nn.init.normal(self.fc.bias, std=0.01)

    # prev_state=prev_reads, controller_state, heads_states
    # prev_state由create_new_state(bach_size) 创建
    # x是真正的外部输入的x(宽度8+1)
    def forward(self, x, prev_state):
        """NTM forward function.

        :param x: input vector (batch_size x num_inputs)
        :param prev_state: The previous state of the NTM
        """
        # Unpack the previous state
        # prev_reads大小=M*读写头个数,就是head读出的内容
        # prev_heads_states就是读写头用的地址寻址w
        prev_reads, prev_controller_state, prev_heads_states = prev_state

        # Use the controller to get an embeddings
        # 形成 x+prev_reads尺寸: batch_size x (BYTE_SIZE+M)
        inp = torch.cat([x] + prev_reads, dim=1)
        #将真正的外部输入数据x和prev_reads(一开始是随机值)组合，输入 lstm controller
        controller_outp, controller_state = self.controller(inp, prev_controller_state)

        # Read/Write from the list of heads
        reads = []
        heads_states = []
        # 逐个head遍历，self.heads=[read,write,read,write,...]
        # read和write的prev寻址w，分开对应提供给read head和write head
        for head, prev_head_state in zip(self.heads, prev_heads_states):
            if head.is_read_head():
                #prev_head_state是head用的前一个地址寻址w, 初始值是0xN,也就是没有attention
                r, head_state = head(controller_outp, prev_head_state)
                reads += [r]
            else:
                head_state = head(controller_outp, prev_head_state)
            heads_states += [head_state]

        # Generate Output
        # 控制器的输出+直接读出的内存内容，经过线性运算和sigmoid，产生真正的输出
        inp2 = torch.cat([controller_outp] + reads, dim=1)
        o = F.sigmoid(self.fc(inp2))

        # Pack the current state
        # 直接读出的内存内容，控制器状态，读写头的地址位移权重w， 组成NTM的状态
        state = (reads, controller_state, heads_states)

        return o, state
