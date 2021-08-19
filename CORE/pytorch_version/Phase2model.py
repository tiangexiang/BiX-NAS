from torch.nn import GroupNorm, Module, Sequential, Conv2d, BatchNorm2d, ConvTranspose2d, ReLU, MaxPool2d, Sigmoid, Parameter
from torch import tensor, cat
import torch.nn as nn
import torch.nn.functional as F
import torch
from .Phase2tools import BFS, decode
from .att_modules import *

NORM_TYPE = 'BN'

class Block(Module):
    """
    Phase2 block
    """
    def __init__(self, in_channel, out_channel, conv=None, batch_norm_momentum=0.01, skip_pre=None, skipped=False):
        super(Block, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv = conv
        self.operator1 = Sequential(
            Conv2d(int(in_channel), out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)) if conv is None else conv[0],
            GroupNorm(out_channel//16, out_channel) if NORM_TYPE == 'GN' else BatchNorm2d(out_channel, momentum=batch_norm_momentum),
            ReLU(inplace=True),
        )

        self.operator2 = Sequential(
            Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)) if conv is None else conv[1],
            GroupNorm(out_channel//16, out_channel) if NORM_TYPE == 'GN' else BatchNorm2d(out_channel, momentum=batch_norm_momentum),
            ReLU(inplace=True),
        )

        if skip_pre is not None:
            skip_pre = list(set(skip_pre))
        else:
            skip_pre = [1,2,3,4]
        self.skip_pre = skip_pre
        self.skipped = skipped
        self.interpolate = None

    def forward(self, x, skip):
        if self.conv[0] is None and self.conv[1] is None: 
            # if block are abandoned across all iterations, no convs initialized
            return x
        if self.skipped: 
            # if the blocks are skipped
            return x
        bn, c, w, h = x.size()

        if self.interpolate is None:
            self.interpolate = torch.nn.Upsample(size=(w,h), mode='bilinear', align_corners=True)
        
        aligned_skip = [x] + skip
        use = [self.interpolate(aligned_skip[i]).unsqueeze(-1) for i in self.skip_pre]
        use = torch.cat(use, dim=-1)
        x = torch.sum(use, dim=-1, keepdim=False) / len(self.skip_pre)

        x = self.operator1(x)
        x = self.operator2(x)
        
        return x

class Phase2(Module):
    """
    Phase2 / retraining model of BiX-NAS
    """
    def __init__(self,
                 in_channel: int = 3,
                 num_classes: int = 1,
                 iterations: int = 2,
                 multiplier: float = 1.0,
                 gene=None,
                 bionetpp=False,
                 with_att=False):
        super(Phase2, self).__init__()
        self.iterations = iterations
        self.multiplier = multiplier
        self.num_layers = 4
        self.batch_norm_momentum = 0.99
        self.initial_filter = int(16 * 2 * self.multiplier)
        self.with_att = with_att
        
        self.filters_list = [int(16 * (3 ** 1) * self.multiplier) for i in range(self.num_layers + 1)]

        self.pre_transform_conv_block = Sequential(
            Conv2d(in_channel, self.initial_filter, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),  # 生成f[1]*512*512
            GroupNorm(self.initial_filter//16, self.initial_filter) if NORM_TYPE == 'GN' else BatchNorm2d(self.initial_filter, momentum=self.batch_norm_momentum),
            ReLU(inplace=True),
            Conv2d(self.initial_filter, self.initial_filter, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            GroupNorm(self.initial_filter//16, self.initial_filter) if NORM_TYPE == 'GN' else BatchNorm2d(self.initial_filter, momentum=self.batch_norm_momentum),
            ReLU(inplace=True),
            Conv2d(self.initial_filter, self.filters_list[0], kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            GroupNorm(self.filters_list[0]//16, self.filters_list[0]) if NORM_TYPE == 'GN' else BatchNorm2d(self.filters_list[0], momentum=self.batch_norm_momentum),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        )

        # if self.with_att:
        #     self.att = ChannelAttention(self.filters_list[0])

        self.reuse_convs = []  # convs for encoder
        self.encoders = nn.ModuleList() 
        self.reuse_deconvs = []  # convs for decoder
        self.decoders = nn.ModuleList()  
        
        count = 0

        skip_codes = BFS(iterations, gene) # check if a block is skipped or not
        enc_skip_or_not = [True for _ in range(self.num_layers)]
        dec_skip_or_not = [True for _ in range(self.num_layers)]
        for layer in range(self.num_layers):
            for iteration in range(self.iterations*2):
                if iteration % 2 == 0: # encoder
                    if not skip_codes[iteration][layer]: # if not skipped
                        enc_skip_or_not[layer] = False
                        continue
                else: # decoder
                    if not skip_codes[iteration][layer]: # if not skipped
                        dec_skip_or_not[layer] = False
                        continue

        for iteration in range(self.iterations):
            # initialize encoders
            for layer in range(self.num_layers):
                in_channel = self.filters_list[layer] 
                out_channel = self.filters_list[layer + 1]
                if iteration == 0:
                    if not enc_skip_or_not[layer]:
                        conv1 = Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False)
                        conv2 = Conv2d(out_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False)
                    else:
                        conv1 = None
                        conv2 = None
                        print('enc', layer, 'abandonned, conv is none!')
                block = Block(in_channel, out_channel, (conv1, conv2), self.batch_norm_momentum, gene[count][layer])
                self.add_module("iteration{0}_layer{1}_encoder_blocks".format(iteration, layer), block)
                self.encoders.append(block)
            count += 1
            #  initialize decoders
            for layer in range(self.num_layers):
                in_channel = self.filters_list[self.num_layers - layer]
                out_channel = self.filters_list[self.num_layers - 1 - layer]
                if iteration == 0:
                    if not dec_skip_or_not[layer]:
                        conv1 = Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False)
                        conv2 = Conv2d(out_channel, out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False)
                    else:
                        conv1 = None
                        conv2 = None
                        print('dec', layer, 'abandoned, conv is none!')
                block = Block(in_channel, out_channel, (conv1, conv2), self.batch_norm_momentum, gene[count][layer])
                self.add_module("iteration{0}_layer{1}_decoder_blocks".format(iteration, layer), block)
                self.decoders.append(block)
            count += 1

        #  bridge block
        self.middles = Sequential(
            Conv2d(self.filters_list[-1], self.filters_list[-1], kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            GroupNorm(self.filters_list[-1]//16, self.filters_list[-1]) if NORM_TYPE == 'GN' else BatchNorm2d(self.filters_list[-1], momentum=self.batch_norm_momentum),
            ReLU(),
            Conv2d(self.filters_list[-1], self.filters_list[-1], kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            GroupNorm(self.filters_list[-1]//16, self.filters_list[-1]) if NORM_TYPE == 'GN' else BatchNorm2d(self.filters_list[-1], momentum=self.batch_norm_momentum),
            ReLU()
        )

        # post processing
        self.post_transform_conv_block = Sequential(
            Conv2d(self.filters_list[0], self.initial_filter, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            #ReLU(inplace=True),
            #GroupNorm(self.initial_filter//16, self.initial_filter),
            BatchNorm2d(self.initial_filter, momentum=self.batch_norm_momentum),
            ReLU(inplace=True),
            Conv2d(self.initial_filter, self.initial_filter, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False),
            #ReLU(inplace=True),
            #GroupNorm(self.initial_filter//16, self.initial_filter),
            BatchNorm2d(self.initial_filter, momentum=self.batch_norm_momentum),
            ReLU(inplace=True)
        )

        # with attention
        if self.with_att:
            self.att2 = ChannelAttention(self.initial_filter)

        # post processing
        self.post2 = Sequential(
            Conv2d(self.initial_filter, num_classes, kernel_size=(1, 1), stride=(1, 1), bias=False),
            Sigmoid(),
        )

        # reset genes
        if not bionetpp and gene is not None:
            self.reset_gene(gene, skip_codes)


    def reset_gene(self, genes, skip_genes=None):
        """
        reset genes
        """
        for idx, gene in enumerate(genes):
            for level in range(self.num_layers):
                if idx % 2 == 0:
                    self.encoders[idx//2 * self.num_layers + level].skip_pre = gene[level]
                    self.encoders[idx//2 * self.num_layers + level].skipped = (skip_genes is not None) and skip_genes[idx][level]
                    if self.encoders[idx//2 * self.num_layers + level].skipped:
                        print('find one encoder skipped! at iteration '+str(idx)+' level '+str(level))
                else:
                    self.decoders[idx//2 * self.num_layers + level].skip_pre = gene[level]
                    self.decoders[idx//2 * self.num_layers + level].skipped = (skip_genes is not None) and skip_genes[idx][level]
                    if self.decoders[idx//2 * self.num_layers + level].skipped:
                        print('find one decoder skipped! at iteration '+str(idx)+' level '+str(level))


    def head_forward(self, x, iterations):
        """
        forward behavior for head networks
        """
        enc = [None for i in range(self.num_layers)]
        dec = [None for i in range(self.num_layers)]
        enc = [x] + enc

        x = self.pre_transform_conv_block(x)

        e_i = 0
        d_i = 0
        for iteration in range(iterations):
 
            for layer in range(self.num_layers):
                
                if iteration%2 == 0 and layer == 0:
                    x_in = x

                # encoder
                if iteration%2 == 0:
                    x_in = self.encoders[e_i](x_in, dec if iteration != 0 else [])
                    enc[layer+1] = x_in
                    x_in = F.max_pool2d(x_in, 2)
                    e_i = e_i + 1
                # decoder
                else:
                    x_in = self.decoders[d_i](x_in, enc[1:])
                    dec[layer] = x_in
                    x_in = F.interpolate(x_in, size=enc[-1 - layer - 1].size()[-2:], mode='bilinear', align_corners=True)
                    d_i = d_i + 1 

            if iteration%2 == 0:
                x_in = self.middles(x_in)
                x_in = F.interpolate(x_in, size=enc[-1].size()[-2:], mode='bilinear', align_corners=True)

        return_value= {}
        return_value['x'] = x
        return_value['x_in'] = x_in
        return_value['enc'] = enc
        return_value['dec'] = dec
        return_value['e_i'] = e_i
        return_value['d_i'] = d_i

        return return_value

    def tail_forward(self, x, iterations, return_value):
        """
        forward behavior for tail networks
        """
        x = return_value['x']
        x_in = return_value['x_in']
        enc = return_value['enc']
        dec = return_value['dec']
        e_i = return_value['e_i']
        d_i = return_value['d_i']

        for iteration in range(iterations, self.iterations * 2):
            for layer in range(self.num_layers):
                
                if iteration%2 == 0 and layer == 0:
                    x_in = x

                # encoder
                if iteration%2 == 0:
                    x_in = self.encoders[e_i](x_in, dec if iteration != 0 else [])
                    enc[layer+1] = x_in
                    x_in = F.max_pool2d(x_in, 2)
                    e_i = e_i + 1
                # decoder
                else:
                    x_in = self.decoders[d_i](x_in, enc[1:])

                    dec[layer] = x_in
                    x_in = F.interpolate(x_in, size=enc[-1 - layer - 1].size()[-2:], mode='bilinear', align_corners=True)
                    d_i = d_i + 1 

            if iteration%2 == 0:
                x_in = self.middles(x_in)
                x_in = F.interpolate(x_in, size=enc[-1].size()[-2:], mode='bilinear', align_corners=True)

        x_in = self.post_transform_conv_block(x_in)
        x_in = self.post2(x_in)
        return x_in

    def forward(self, x, iterations=0, return_value=None, profile=True):
        if profile:
            return self.onepass_forward(x)

        if return_value is None:
            return self.head_forward(x, iterations)
        else:
            return self.tail_forward(x, iterations, return_value)

    def onepass_forward(self, x):
        """
        forward behavior for BiX-Net
        """
        enc = [None for i in range(self.num_layers)]
        dec = [None for i in range(self.num_layers)]
        enc = [x] + enc

        x = self.pre_transform_conv_block(x)
        # if self.with_att:
        #     x = self.att(x)

        e_i = 0
        d_i = 0
        for iteration in range(self.iterations * 2):
            
            for layer in range(self.num_layers):
                
                if iteration%2 == 0 and layer == 0:
                    x_in = x

                if iteration%2 == 0:
                    x_in = self.encoders[e_i](x_in, dec if iteration != 0 else [])
                    enc[layer+1] = x_in
                    x_in = F.max_pool2d(x_in, 2)
                    e_i = e_i + 1
                else:
                    x_in = self.decoders[d_i](x_in, enc[1:])

                    dec[layer] = x_in
                    x_in = F.interpolate(x_in, size=enc[-1 - layer - 1].size()[-2:], mode='bilinear', align_corners=True)
                    d_i = d_i + 1 

            if iteration%2 == 0:
                x_in = self.middles(x_in)
                x_in = F.interpolate(x_in, size=enc[-1].size()[-2:], mode='bilinear', align_corners=True)

        x_in = self.post_transform_conv_block(x_in)
        if self.with_att:
            x_in = self.att2(x_in)
        x_in = self.post2(x_in)
        return x_in

if __name__ == '__main__':
    codes = decode('./BiX-NAS/bixnet_gene.txt', 3, 4)
    print(codes)
