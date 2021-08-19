from torch.nn import GroupNorm, Module, Sequential, Conv2d, BatchNorm2d, ConvTranspose2d, ReLU, MaxPool2d, Sigmoid, Parameter
from torch import tensor, cat
import torch.nn as nn
import torch.nn.functional as F
import torch

class Block(Module):
    """
    Phase1 block
    :skip: all N incoming skips
    :alpha: a selection matrix with shape N*(N-2)
    """
    def __init__(self, in_channel, out_channel, conv=None, batch_norm_momentum=0.01, last_decoder=False):
        super(Block, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.operator1 = Sequential(
            Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)) if conv is None else conv[0],
            BatchNorm2d(out_channel, momentum=batch_norm_momentum),
            ReLU(inplace=True)
        )

        self.operator2 = Sequential(
            Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)) if conv is None else conv[1],
            BatchNorm2d(out_channel, momentum=batch_norm_momentum),
            ReLU(inplace=True)
        )

        self.last_decoder = last_decoder

        # selection matrix
        alpha = torch.nn.Parameter(torch.ones((5, 3)), requires_grad=True)
        # alpha = torch.nn.Parameter(torch.ones((4, 3)), requires_grad=True)

        self.register_parameter('alpha', alpha)
        torch.nn.init.xavier_uniform_(self.alpha)

    def forward(self, x, skip):
        bn, c, w, h = x.size()
        
        if skip is not None:
            aligned_skip = [F.interpolate(v, size=x.size()[-2:], mode='bilinear').unsqueeze(-1) for v in skip+[x]]
            aligned_skip = torch.cat(aligned_skip, dim=-1) # bn c w h 5

            score = F.gumbel_softmax(self.alpha, dim=0)
            skip = torch.matmul(aligned_skip, score) # bn c w h 3
            
            # weight matrix
            maxidx = torch.argmax(self.alpha, dim=0, keepdim=False)
            un, idx, counts = torch.unique(maxidx, return_inverse=True, return_counts=True, dim=0)
            weights = torch.ones(3,).cuda() #cuda
            # weights = torch.ones(3,) #cpu

            for i in range(idx.size(0)):
                weights[i] /= counts[idx[i]]
  
            skip = torch.sum(skip * weights.view(1, 1, 1, 1, -1), dim=-1, keepdim=False) # bn c w h 
            skip = skip + x
            
            x = skip/(un.size(0) +1.)

        x = self.operator1(x)
        x = self.operator2(x)

        return x

#########################################
############ Phase1 of BiX-NAS #########
#########################################
class Phase1(Module):
    """
    Phase1 of BiX-NAS
    """
    def __init__(self,
                 in_channel :int = 3,
                 num_classes: int = 1,
                 iterations: int = 2,
                 multiplier: float = 1.0):
        super(Phase1, self).__init__()
        self.iterations = iterations
        self.multiplier = multiplier
        self.num_layers = 4
        self.batch_norm_momentum = 0.99
        self.initial_filter = int(16 * 2 * self.multiplier)

        # define filters 
        self.filters_list = [int(16 * (3 ** 1) * self.multiplier) for i in range(self.num_layers + 1)]

        # preprocessing block
        self.pre_transform_conv_block = Sequential(
            Conv2d(in_channel, self.initial_filter, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),  # 生成f[1]*512*512
            BatchNorm2d(self.initial_filter, momentum=self.batch_norm_momentum),
            ReLU(inplace=True),
            Conv2d(self.initial_filter, self.initial_filter, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            BatchNorm2d(self.initial_filter, momentum=self.batch_norm_momentum),
            ReLU(inplace=True),
            Conv2d(self.initial_filter, self.filters_list[0], kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            BatchNorm2d(self.filters_list[0], momentum=self.batch_norm_momentum),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        )
      
        self.reuse_convs = []  
        self.encoders = nn.ModuleList()  
        self.reuse_deconvs = [] 
        self.decoders = nn.ModuleList()  

        for iteration in range(self.iterations):
            for layer in range(self.num_layers):
                # encoder blocks
                in_channel = self.filters_list[layer]
                out_channel = self.filters_list[layer + 1]
                if iteration == 0:
                    conv1 = Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False)
                    conv2 = Conv2d(out_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False)
                block = Block(in_channel, out_channel, (conv1, conv2), self.batch_norm_momentum)
                self.add_module("iteration{0}_layer{1}_encoder_blocks".format(iteration, layer), block)
                self.encoders.append(block)

                # decoder blocks
                in_channel = self.filters_list[self.num_layers - layer]
                out_channel = self.filters_list[self.num_layers - 1 - layer]
                if iteration == 0:
                    conv1 = Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False)
                    conv2 = Conv2d(out_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False)
                block = Block(in_channel, out_channel, (conv1, conv2), self.batch_norm_momentum, last_decoder=iteration==self.iterations-1)
                self.add_module("iteration{0}_layer{1}_decoder_blocks".format(iteration, layer), block)
                self.decoders.append(block)

        # bridge block
        self.middles = Sequential(
            Conv2d(self.filters_list[-1], self.filters_list[-1], kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            BatchNorm2d(self.filters_list[-1], momentum=self.batch_norm_momentum),
            ReLU(),
            Conv2d(self.filters_list[-1], self.filters_list[-1], kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            BatchNorm2d(self.filters_list[-1], momentum=self.batch_norm_momentum),
            ReLU()
        )

        # postprocessing block
        self.post_transform_conv_block = Sequential(
            Conv2d(self.filters_list[0], self.initial_filter, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            BatchNorm2d(self.initial_filter, momentum=self.batch_norm_momentum),
            ReLU(inplace=True),
            Conv2d(self.initial_filter, self.initial_filter, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            BatchNorm2d(self.initial_filter, momentum=self.batch_norm_momentum),
            ReLU(inplace=True),
            Conv2d(self.initial_filter, num_classes, kernel_size=(1, 1), stride=(1, 1), bias=False),
            Sigmoid()
        )

    def forward(self, x):
        enc = [None for i in range(self.num_layers)]
        dec = [None for i in range(self.num_layers)]
        enc = [x] + enc

        x = self.pre_transform_conv_block(x)
        e_i = 0
        d_i = 0
        for iteration in range(self.iterations):
            # encoding path
            for layer in range(self.num_layers):
                if layer == 0:
                    x_in = x

                x_in = self.encoders[e_i](x_in, dec if iteration != 0 else None)
                enc[layer+1] = x_in
                x_in = F.max_pool2d(x_in, 2)
                e_i = e_i + 1

            # bridging
            x_in = self.middles(x_in)
            x_in = F.interpolate(x_in, size=enc[-1].size()[-2:], mode='bilinear', align_corners=True)

            # decodong path
            for layer in range(self.num_layers):
                x_in = self.decoders[d_i](x_in, enc[1:])
                dec[layer] = x_in
                x_in = F.interpolate(x_in, size=enc[-1 - layer - 1].size()[-2:], mode='bilinear', align_corners=True)
                d_i = d_i + 1

        x_in = self.post_transform_conv_block(x_in)
        return x_in
