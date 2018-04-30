"""All in one NTM. Encapsulation of all components."""
import torch
from torch import nn
from torch.autograd import Variable

from .ntm import NTM
from .controller import LSTMController, MLPController
from .head import NTMReadHead, NTMWriteHead
from .memory import NTMMemory


class EncapsulatedNTM(nn.Module):

    def __init__(self, num_inputs, num_outputs,
                 controller_size, controller_layers, num_heads, N, M, controller_type ='lstm'):
        """Initialize an EncapsulatedNTM.
        注意：外部输入的向量的维度同内存块尺寸不一定相等
        这个类的用处就是组装内存/读写头/控制器/NTM图灵机操作流程
        memeory的大小N*M, 共有batch_size个内存块
        :param num_inputs: External number of inputs.外部输入维度，比如9位
        :param num_outputs: External number of outputs.外部输出维度，比如8位
        :param controller_size: The size of the internal representation.
        :param controller_layers: Controller number of layers.
        :param num_heads: Number of heads.
        :param N: Number of rows in the memory bank.
        :param M: Number of cols/features in the memory bank.
        exemple:
        net = EncapsulatedNTM(sequence_width + 1, sequence_width,
                              controller_size, controller_layers,
                              num_heads,
                              memory_n, memory_m)
        """
        super(EncapsulatedNTM, self).__init__()

        # Save args
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.controller_size = controller_size
        self.controller_layers = controller_layers
        self.num_heads = num_heads
        self.N = N
        self.M = M

        # Create the NTM components
        #N代表内存块的地址数或行数，M代表内存块每个地址的向量大小
        memory = NTMMemory(N, M)
        #for the assignment3: Use only 1 read head and 1 write head
        #external_input_size=9 + M*num_heads, controller_size, controller_layers
        if controller_type == 'lstm' : 
            controller = LSTMController(num_inputs + M*num_heads, controller_size, controller_layers)
        else :
            controller = MLPController(num_inputs + M*num_heads, controller_size, controller_layers)
            
        heads = nn.ModuleList([])
        #heads=[read,write,read,write,...]
        for i in range(num_heads):
            heads += [
                #建立读写头，用controller的out的size也即隐藏层的size和memory建立
                # controller_size用于获得读写头需要的k, β, g, s, γ, e, a
                NTMReadHead(memory, controller_size),
                NTMWriteHead(memory, controller_size)
            ]

        self.ntm = NTM(num_inputs, num_outputs, controller, memory, heads)
        self.memory = memory

    def init_sequence(self, batch_size):
        """Initializing the state."""
        self.batch_size = batch_size
        self.memory.reset(batch_size)
        self.previous_state = self.ntm.create_new_state(batch_size)

    #x=真正的外部输入的x(宽度8+1)
    def forward(self, x=None):
        if x is None:
            # 当x没有输入时，表示做copy动作，获取内存
            # 注意，这里用什么来训练，会直接影响使用ntm时，用什么来读取
            # 比如读取用0来训练，当使用ntm时，也得用0-0.5来读取，如果用0.5-1则任务失败
            # x = Variable(torch.zeros(self.batch_size, self.num_inputs))
            # 如果用0-1随机值来训练，会导致controller使用delimiter并记住第一个vector，并在copy时控制读头在第一次使用内容寻址
            x = Variable(torch.Tensor(self.batch_size, self.num_inputs).uniform_(0,1))
            
        #x=真正的外部输入的x(宽度8+1)
        #o=真正的外部输出(宽度8)
        o, self.previous_state = self.ntm(x, self.previous_state)
        return o, self.previous_state

    def calculate_num_params(self):
        """Returns the total number of parameters."""
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params
