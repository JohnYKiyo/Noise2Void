#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet1D(nn.Module):

    def __init__(self, 
        in_channels=1, 
        out_channels=1,
        channel_expansion=64,
        depth=5, 
        n_filters_at_firstlayer=6, 
        padding=False, 
        batch_norm=False, 
        up_mode='upconv',):

        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation(Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            depth (int): depth of the network
            n_filters_at_firstlayer (int): number of filters in the first layer is 2**n_filters_at_firstlayer
                default is 6 => 2**6 = 64
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                                activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                            'upconv' will use transposed convolutions for
                            learned upsampling.
                            'upsample' will use bilinear upsampling.
        """
        super(UNet1D, self).__init__()

        assert up_mode in ('upconv', 'upsample')
        
        self.padding = padding
        self.depth = depth
        
        self.conv1d = nn.Conv1d(in_channels,channel_expansion,kernel_size=3,stride=1,padding=1,bias=False)
        
        prev_channels = in_channels
        #make down sample layers
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (n_filters_at_firstlayer + i), padding, batch_norm)
            )
            prev_channels = 2 ** (n_filters_at_firstlayer + i)
		
        #make up sample layers
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(prev_channels, 2 ** (n_filters_at_firstlayer + i), up_mode, padding, batch_norm)
            )
            prev_channels = 2 ** (n_filters_at_firstlayer + i)
		
        self.last = nn.Conv2d(prev_channels, out_channels, kernel_size=1)

        self.deconv1d = nn.ConvTranspose1d(channel_expansion,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        
    def forward(self, x):
        x = self.conv1d(x)
        x = x.unsqueeze(1)
    
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
        
        x = self.last(x)
        x = x.squeeze(1)
        return self.deconv1d(x)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2,align_corners=True),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )
            
        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out