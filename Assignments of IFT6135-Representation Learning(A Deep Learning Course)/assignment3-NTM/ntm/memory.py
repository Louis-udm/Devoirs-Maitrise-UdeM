"""An NTM's memory implementation."""
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import numpy as np
'''
对一个已经训练好的MLP-NTM测试的结果
写入时:
read header的s基本不变，不移位(010)，g=1,说明读头内容寻址后，一直保持在这个位置
write header的s移位，(001),g=0.1，说明写头更多的参考上一个写地址，并移到下一个地址写入
读取时：
read header的s移位(001)，g=0,说明读头更多的参考上一个读地址，并移到下一个地址读取
write header的w此时已经弥散，g一般=1,s一般不移位。

说明读写时，NTM是选择地址寻址而非内容寻址。

k应该是从读内存的r_{t-1}学到 ?
a应该是从输入x学到 ?

----------------------------------

# w是一个N维度向量，代表对某行地址的聚焦
#假设N=6:
# wg=0.1000  0.1000  0.8000  0.1000  0.1000  0.1000  0.1000
# t相当于w的padding=2, t=0.1000  0.1000  wg  0.1000  0.1000
# s=    0     0     1  如果1在中间不移位
# c的维度=t-s+1=pading*2+1=5,  
# c[1:-1]=0.1000  0.8000  0.1000  0.1000  0.1000  0.1000  0.1000 看到0.8前移1位
# 返回维度为n，相当于n+2+2-3+1-2
# 也可以padding=1(取一个尾到头，取一个头到尾)，完成卷积后不去头不去尾
'''

def _convolve(w, s):
    """Circular convolution implementation."""
    assert s.size(0) == 3
    # 强制朝右
    # s2=s.clone()
    # s2[0]=s2[0]+s2[2]
    # s2[2]=0.0
    # s=s2
    # w向量的后两个元素+w+前两个元素
    t = torch.cat([w[-2:], w, w[:2]])
    c = F.conv1d(t.view(1, 1, -1), s.view(1, 1, -1)).view(-1)
    return c[1:-1]


class NTMMemory(nn.Module):
    """Memory bank for NTM."""
    def __init__(self, N, M):
        """Initialize the NTM Memory matrix.

        The memory's dimensions are (batch_size x N x M).
        Each batch has it's own memory matrix.

        :param N: Number of rows in the memory.
        :param M: Number of columns/features in the memory.
        N代表地址数或行数，M代表每个地址的向量大小，共有batch_size个内存块        
        """
        super(NTMMemory, self).__init__()

        self.N = N
        self.M = M

        # The memory bias allows the heads to learn how to initially address
        # memory locations by content
        # 就是产生self.mem_bias这个参数,nxm
        self.register_buffer('mem_bias', Variable(torch.Tensor(N, M)))

        # Initialize memory bias
        stdev = 1 / (np.sqrt(N + M))  #?????
        #Fills the input Tensor--self.mem_bias with values drawn from the uniform distribution(-stdev, stdev)
        #初始化self.mem_bias这个参数,nxm
        nn.init.uniform(self.mem_bias, -stdev, stdev)

    def reset(self, batch_size):
        """Initialize memory from bias, for start-of-sequence."""
        self.batch_size = batch_size
        #初始化memory内存块：从self.mem_bias产生batch_size个nxm的memory
        #memory=batch_size × N × M
        self.memory = self.mem_bias.clone().repeat(batch_size, 1, 1)

    def size(self):
        return self.N, self.M

    def read(self, w):
        """Read from memory (according to section 3.1)."""
        # n是地址，相当于sum(w(i->n) × Mem(n×m) )表示如果第i个权重高，则attention就在第i个地址
        # 比如 [001000]xMem=第3行内容
        #  w.unsqueeze-> (j×1×heads×n)   #??????????
        #  memory-> (batch_size × N × M)
        #  out will be an (j × batch_size × heads × M).squeeze(1)
        return torch.matmul(w.unsqueeze(1), self.memory).squeeze(1)

    #给定t时刻的写头权重w_t，以及一个擦出向量e_t，其中M个元素均在0~1范围内
    #w相当于聚焦某行，e则提供那些列要删除，则(某行某列)=1表示删除
    #加向量a_t相当于写入内容
    #意思：删除w行e列，加入w行a列
    def write(self, w, e, a):
        """write to memory (according to section 3.2)."""
        self.prev_mem = self.memory
        self.memory = Variable(torch.Tensor(self.batch_size, self.N, self.M))
        
        #Made the outer products in NTMMemory.write parallel batch-wise
        erase = torch.matmul(w.unsqueeze(-1), e.unsqueeze(1))
        add = torch.matmul(w.unsqueeze(-1), a.unsqueeze(1))
        self.memory = self.prev_mem * (1 - erase) + add
        
        # for b in range(self.batch_size):
        #     #ger产生两向量的笛卡尔积
        #     #w相当于聚焦某行，e则提供那些列要删除，则(某行某列)=1表示删除
        #     erase = torch.ger(w[b], e[b])
        #     # 要写入的所有的(某行某列)的值
        #     #通过w和a交织成同内存块同大小的块，但是这其中只有w行a列有内容会被更新
        #     add = torch.ger(w[b], a[b])
        #     #执行擦写操作
        #     self.memory[b] = self.prev_mem[b] * (1 - erase) + add

    #产生寻址权重
    def address(self, k, β, g, s, γ, w_prev):
        """NTM Addressing (according to section 3.3).产生寻址权重
        内容寻址的权重被key作用后会基于上一时刻的权重和gate值g_t进行插值调整。 
        随后位移向量s_t会决定是否或者进行多少的旋转操作。最后，依赖于γ_t, 权重会被sharpen锐化以用于内存访问。
        Returns a softmax weighting over the rows of the memory matrix.
        k应该是从读内存的r_{t-1}学到 ?
        a应该是从输入x学到 ?
        :param k: The key vector.
        :param β: The key strength (focus).key的强度
        :param g: Scalar interpolation gate (with previous weighting).
        :param s: Shift weighting.
        :param γ: Sharpen weighting scalar.
        :param w_prev: The weighting produced in the previous time step.
        结合权重插值、内容寻址和地址寻址的寻址系统可以在三种补充（complementary）模式下工作。
        第一，权重列表可以由内容系统来自主选择而不被地址系统所修改。
        第二，有内容系统产生的权重可以再选择和位移。这使得焦点能够跳跃到通过内容寻址产生的地址附近而不是只能在其上。
        在计算方面，这使得读写头可以访问一个连续的数据块，并访问这个块中特定数据。
        第三，来自上一个时刻的权重可以在没有任何内容系统输入的情况下被旋转，以便权重可以以相同的时间间隔连续地访问一个地址序列。
        """
        # Content focus
        # wc=batch_size x N, N代表批处理下的每个向量，代表每一行的同key相似度的softmax比重
        wc = self._similarity(k, β)

        # Location focus
        # wg为指定地址寻址，结合了当前内容寻址权重wc和前一时刻的w, g为两者的比重,0则w_prev,1则wc
        # wg=batch_size x N
        wg = self._interpolate(w_prev, wc, g)
        ŵ = self._shift(wg, s)
        w = self._sharpen(ŵ, γ)

        return w

    # The ﬁrst mechanism, “content-based addressing,” focuses attention on locations 
    # based on the similarity between their current values and values emitted by the controller. 
    # This is related to the content-addressing of Hopﬁeld networks (Hopﬁeld, 1982).
    # 3.3.1 按内容聚焦
    # 对于内容寻址，每个读写头都首先产生一个M长度的key向量k_t，并通过一个相似度度量函数K[.,.] (这里是余弦相似)
    # 分别与每个行向量M_t(i)逐一比较。基于内容的系统会基于相似度和key的强度产生一个归一化的权重列表w_{t}^{c}，
    # β_t可以放大或减弱聚焦的精度。
    # 返回w=batch_size x N
    def _similarity(self, k, β):
        k = k.view(self.batch_size, 1, -1)
        #余弦相似torch.F.cosine_similarity
        #w[batch_size]是一个向量,代表每一行的同key相似度的softmax比重
        w = F.softmax(β * F.cosine_similarity(self.memory + 1e-16, k + 1e-16, dim=-1), dim=1)
        return w

    def _interpolate(self, w_prev, wc, g):
        return g * wc + (1 - g) * w_prev

    # wg=batch_size x N， 每个向量代表当前batch_size下对某行地址的聚焦
    # 对wg对焦点进行位移, convolve计算时会先对s进行旋转
    def _shift(self, wg, s):
        result = Variable(torch.zeros(wg.size()))
        for b in range(self.batch_size):
            result[b] = _convolve(wg[b], s[b])
        return result

    # 锐化算法：w_i=(w_i)^γ/[(w_1)^γ+(w_2)^γ+...]
    def _sharpen(self, ŵ, γ):
        w = ŵ ** γ
        w = torch.div(w, torch.sum(w, dim=1).view(-1, 1) + 1e-16)
        return w
