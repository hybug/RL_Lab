from typing import Dict
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class XavierLinear(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None, bias=True, w_init='linear'):
        super(XavierLinear, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        self.activation = activation
        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init))

    def forward(self, x):
        x = self.linear_layer(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, kernel_size=(3, 3), padding=(1, 1), stride: int = 1, residual_path: bool = False):
        super().__init__()
        self.residual_path = residual_path

        self.c1 = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding=padding)
        self.b1 = nn.BatchNorm2d(out_dim)
        self.a1 = nn.ReLU()

        self.c2 = nn.Conv2d(out_dim, out_dim, kernel_size, 1, padding=padding)
        self.b2 = nn.BatchNorm2d(out_dim)

        if residual_path:
            self.down_c1 = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding=padding, bias=False)
            self.down_b1 = nn.BatchNorm2d(out_dim)

        self.a2 = nn.ReLU()

    def forward(self, x):
        residual = x
        delta_1 = self.a1(self.b1(self.c1(x)))
        if self.residual_path:
            residual = self.down_b1(self.down_c1(residual))
        delta_2 = self.a2(self.b2(self.c2(delta_1)) + residual)

        return delta_2


class Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_channel = 26
        self.input_conv = nn.Conv2d(26, 64, (3, 3), 1, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(64)
        self.input_activate = nn.ReLU()

        self.res_blocks = []
        in_channel = 64
        out_channel = 64
        for block_id in range(len([1, 1, 4])):
            for layer_id in range([1, 1, 4][block_id]):
                if layer_id == 0 and block_id != 0:
                    self.res_blocks.append(ResnetBlock(in_channel, out_channel, stride=2, residual_path=True))
                elif block_id >= 2:
                    self.res_blocks.append(ResnetBlock(in_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)))
                else:
                    self.res_blocks.append(ResnetBlock(in_channel, out_channel))
                in_channel = out_channel
            out_channel *= 2
        out_channel = out_channel // 2
        self.res_blocks = nn.ModuleList(self.res_blocks)

        self.output_layer = XavierLinear(out_channel, classes)

        self.value_dense = XavierLinear(out_channel, 64)
        self.value_activate = nn.ReLU()
        self.value_output_layer = XavierLinear(64, 1)

    def forward(self, feature_dict: Dict):
        x = feature_dict['feature']
        batch_size = x.size(0)
        input_channel = x.size(1)
        if self.input_channel < input_channel:
            x = x[:, :self.input_channel, :, :]
        output = self.input_activate(self.input_bn(self.input_conv(x)))

        for res_block in self.res_blocks:
            output = res_block(output)

        pooling_output = F.adaptive_avg_pool2d(output, (1, 1))
        pooling_output = pooling_output.view(batch_size, -1)

        action_output = self.output_layer(pooling_output)
        if "output_mask" in feature_dict and action_output.size(1) == feature_dict["output_mask"].size(1):
            action_output = action_output.masked_fill(feature_dict["output_mask"], -np.inf)

        action_output = nn.Softmax(dim=1)(action_output)
        value_output = self.value_output_layer(self.value_activate(self.value_dense(pooling_output)))
        value_output = value_output.squeeze(1)

        return action_output, value_output
