"""LSTM Controller."""
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np


class LSTMController(nn.Module):
    """
    An NTM controller based on LSTM.
    exemple: controller = LSTMController(external_input_size=9 + M=8*num_heads, controller_size, controller_layers)
    # 将真正的外部输入数据x和prev_reads(就是之前读到的内存内容=M*读写头个数)组合，输入lstm controller
    # 输入x=torch.cat([x] + prev_reads, dim=1) ，prev_reads=M*读写头个数
    # controller的输出有两个用处：
    # 1.提供给header，由header生成k, β, g, s, γ, e, a， 
    # 2.在NTM层在最后结合header读出的内容运算出output
    """
    def __init__(self, num_inputs, num_outputs, num_layers):
        super(LSTMController, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers

        # LSTM模型的hidden_size是out和hidden一起用，因为同样的单元数输出的这两个值
        self.lstm = nn.LSTM(input_size=num_inputs, #输入x=torch.cat([x] + prev_reads, dim=1) ，prev_reads=M*读写头个数
                            hidden_size=num_outputs, #100
                            num_layers=num_layers)

        # The hidden state is a learned parameter
        self.lstm_h_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05) #?????
        self.lstm_c_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0.05)

        self.reset_parameters()

    def create_new_state(self, batch_size):
        # Dimension: (num_layers * num_directions, batch, hidden_size)
        lstm_h = self.lstm_h_bias.clone().repeat(1, batch_size, 1)
        lstm_c = self.lstm_c_bias.clone().repeat(1, batch_size, 1)
        return lstm_h, lstm_c

    def reset_parameters(self):
        for p in self.lstm.parameters():
            if p.dim() == 1:
                nn.init.constant(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.num_inputs +  self.num_outputs))  #????
                nn.init.uniform(p, -stdev, stdev)

    def size(self):
        return self.num_inputs, self.num_outputs

    # lstm自己一般不保留h和c，由调用者自己保留h和c,
    # h和c的初始化用create_new_state(),reset_parameters()
    # 将真正的外部输入数据x和prev_reads(就是之前读到的内存内容=M*读写头个数)组合，输入lstm controller
    # 输入x=torch.cat([x] + prev_reads, dim=1) ，prev_reads=M*读写头个数
    # controller的输出有两个用处：
    # 1.提供给header，由header生成k, β, g, s, γ, e, a， 
    # 2.在NTM层在最后结合header读出的内容运算出output
    def forward(self, x, prev_state):
        x = x.unsqueeze(0)
        outp, state = self.lstm(x, prev_state)
        return outp.squeeze(0), state

class MLPController(nn.Module):
    """An NTM controller based on MLP."""
    def __init__(self, num_inputs, num_outputs, num_layers):
        super(MLPController, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers

        self.mlp = nn.Linear(num_inputs, num_outputs)

        self.reset_parameters()

    def create_new_state(self, batch_size):
        # Dimension: (num_layers * num_directions, batch, hidden_size)
        state = torch.zeros(1, batch_size, 1)
        return state

    def reset_parameters(self):
        for p in self.mlp.parameters():
            if p.dim() == 1:
                nn.init.constant(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.num_inputs +  self.num_outputs))
                nn.init.uniform(p, -stdev, stdev)

    def size(self):
        return self.num_inputs, self.num_outputs

    def forward(self, x, state):
        x = x.unsqueeze(0)
        outp = self.mlp(x)
        # outp = F.tanh(outp)
        # outp = F.softplus(outp)
        # outp = F.sigmoid(outp)
        return outp.squeeze(0), state