# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from nnunet.ODConv import ODConv2d, ODConv3d


normalizations = {
    "instancenorm3d": nn.InstanceNorm3d,
    "instancenorm2d": nn.InstanceNorm2d,
    "batchnorm3d": nn.BatchNorm3d,
    "batchnorm2d": nn.BatchNorm2d,
}

convolutions = {
    "ODConv2d": ODConv2d,
    "ODConv3d": ODConv3d,
    "Conv2d": nn.Conv2d,
    "Conv3d": nn.Conv3d,
    "ConvTranspose2d": nn.ConvTranspose2d,
    "ConvTranspose3d": nn.ConvTranspose3d,
}


def get_norm(name, out_channels, groups=32):
    if "groupnorm" in name:
        return nn.GroupNorm(groups, out_channels, affine=True)
    return normalizations[name](out_channels, affine=True)


def get_conv(in_channels, out_channels, kernel_size, stride, dim=3, conv_type="", bias=False):
    conv = convolutions[f"{conv_type}Conv{dim}d"]
    padding = get_padding(kernel_size, stride)
    try: return conv(in_channels, out_channels, kernel_size, stride, padding)
    except: return conv(in_channels, out_channels, kernel_size[0], stride, padding)


def get_transp_conv(in_channels, out_channels, kernel_size, stride, dim):
    conv = convolutions[f"ConvTranspose{dim}d"]
    padding = get_padding(kernel_size, stride)
    output_padding = get_output_padding(kernel_size, stride, padding)
    return conv(in_channels, out_channels, kernel_size, stride, padding, output_padding)


def get_padding(kernel_size, stride):
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = (kernel_size_np - stride_np + 1) / 2
    padding = tuple(int(p) for p in padding_np)
    return padding if len(padding) > 1 else padding[0]


def get_output_padding(kernel_size, stride, padding):
    kernel_size_np = np.atleast_1d(kernel_size)
    stride_np = np.atleast_1d(stride)
    padding_np = np.atleast_1d(padding)
    out_padding_np = 2 * padding_np + stride_np - kernel_size_np
    out_padding = tuple(int(p) for p in out_padding_np)
    return out_padding if len(out_padding) > 1 else out_padding[0]


class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(InputBlock, self).__init__()
        self.conv1 = get_conv(in_channels, out_channels, 3, 1)
        self.conv2 = get_conv(out_channels, out_channels, 3, 1)
        self.norm = get_norm(kwargs["norm"], out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, conv_type="", **kwargs):
        super(ConvLayer, self).__init__()
        self.conv = get_conv(in_channels, out_channels, kernel_size, stride, 3, conv_type)
        self.norm = get_norm(kwargs["norm"], in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        x = self.relu(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, conv_type="", **kwargs):
        super(ConvBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride, conv_type, **kwargs)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size, 1, conv_type, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, conv_type="", **kwargs):
        super(UpsampleBlock, self).__init__()
        self.conv_block = ConvBlock(out_channels + in_channels, out_channels, kernel_size, 1, **kwargs)

    def forward(self, x, x_skip):
        x = nn.functional.interpolate(x, scale_factor=2, mode="trilinear", align_corners=True)
        x = torch.cat((x, x_skip), dim=1)
        x = self.conv_block(x)
        return x


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dim):
        super(OutputBlock, self).__init__()
        self.conv = get_conv(in_channels, out_channels, kernel_size=1, stride=1, dim=dim, bias=True)

    def forward(self, input_data):
        return self.conv(input_data)


class UNet3D(nn.Module):
    def __init__(
        self,
        kernels,
        strides,
    ):
        super(UNet3D, self).__init__()
        self.dim = 3
        self.n_class = 3
        self.deep_supervision = True
        self.norm = "instancenorm3d"
        self.filters = [64, 128, 256, 512, 768, 1024, 2048][: len(strides)]
 
        down_block = ConvBlock
        self.input_block = InputBlock(5, self.filters[0], norm=self.norm)
        self.downsamples = self.get_module_list(
            conv_block=down_block,
            in_channels=self.filters[:-1],
            out_channels=self.filters[1:],
            kernels=kernels[1:-1],
            strides=strides[1:-1],
            # conv_type="OD"
        )
        self.bottleneck = self.get_conv_block(
            conv_block=down_block,
            in_channels=self.filters[-2],
            out_channels=self.filters[-1],
            kernel_size=kernels[-1],
            stride=strides[-1],
            conv_type="OD"
        )
        self.input_block_interpolate = InputBlock(5, self.filters[0], norm=self.norm)
        self.downsamples_interpolate = self.get_module_list(
            conv_block=down_block,
            in_channels=self.filters[:-1],
            out_channels=self.filters[1:],
            kernels=kernels[1:-1],
            strides=strides[1:-1],
            # conv_type="OD"
        )
        self.bottleneck_interpolate = self.get_conv_block(
            conv_block=down_block,
            in_channels=self.filters[-2],
            out_channels=self.filters[-1],
            kernel_size=kernels[-1],
            stride=strides[-1],
            # conv_type="OD"
        )
        self.upsamples = self.get_module_list(
            conv_block=UpsampleBlock,
            in_channels=self.filters[1:][::-1],
            out_channels=self.filters[:-1][::-1],
            kernels=kernels[1:][::-1],
            strides=strides[1:][::-1],
        )
        self.output_block = self.get_output_block(decoder_level=0)
        self.deep_supervision_heads = self.get_deep_supervision_heads()
        self.apply(self.initialize_weights)

    def cross_attention(self, out, out_inp):
        pad = torch.zeros(out_inp.shape, device=0)
        pad[:, :, :out.shape[-3], :out.shape[-2], :out.shape[-1]] = out
        out = pad

        MHA_output = torch.zeros(out.shape).cuda()
        multihead_attn = nn.MultiheadAttention(out.shape[2]*out.shape[3]*out.shape[4], 8, 0.1, device=0)

        for i in range(out.shape[0]):
            attn_output, _ = multihead_attn(out[i].reshape(out.shape[1], out.shape[2]*out.shape[3]*out.shape[4]),
                                            out[i].reshape(out.shape[1], out.shape[2]*out.shape[3]*out.shape[4]),
                                            out_inp[i].reshape(out_inp.shape[1], out_inp.shape[2]*out_inp.shape[3]*out_inp.shape[4]))
            MHA_output[i] = attn_output.reshape(out.shape[1], out.shape[2], out.shape[3], out.shape[4])

        return MHA_output[:, :, :out.shape[-3], :out.shape[-2], :out.shape[-1]], MHA_output

    def forward(self, input_data):      #(128, 128, 128)
        input_big   = input_data
        input_small = F.interpolate(input_data, size=(64, 64, 64), mode='trilinear', align_corners=False)
        
        out = self.input_block(input_small)
        out_inp = self.input_block_interpolate(input_big)

        encoder_outputs = [out]
        encoder_outputs_inp = [out_inp]
        for i, (downsample, downsample_inp) in enumerate(zip(self.downsamples, self.downsamples_interpolate)):
            out, out_inp = downsample(out), downsample_inp(out_inp)
            encoder_outputs.append(out)
            encoder_outputs_inp.append(out_inp)
        out = self.bottleneck(out)
        out_inp = self.bottleneck_interpolate(out_inp)
        
        
        out, out_inp = self.cross_attention(out, out_inp)
        # out = MHA_output[:, :, :input_small.shape[-3], :input_small.shape[-2], :input_small.shape[-1]]
        # out = MHA_output

        decoder_outputs = []
        for upsample, skip in zip(self.upsamples, reversed(encoder_outputs_inp)):
            out = upsample(out, skip)
            decoder_outputs.append(out)
        out = self.output_block(out)

        if self.training and self.deep_supervision:
            out = [out]
            for i, decoder_out in enumerate(decoder_outputs[-3:-1][::-1]):
                out.append(self.deep_supervision_heads[i](decoder_out))
        return out

    def get_conv_block(self, conv_block, in_channels, out_channels, kernel_size, stride, drop_block=False, conv_type=""):
        # print("####################################", conv_type)
        return conv_block(
            dim=self.dim,
            stride=stride,
            norm=self.norm,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            conv_type=conv_type
        )

    def get_output_block(self, decoder_level):
        return OutputBlock(in_channels=self.filters[decoder_level], out_channels=self.n_class, dim=self.dim)

    def get_deep_supervision_heads(self):
        return nn.ModuleList([self.get_output_block(1), self.get_output_block(2)])

    def get_module_list(self, in_channels, out_channels, kernels, strides, conv_block, conv_type=""):
        layers = []
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@", conv_type)
        for in_channel, out_channel, kernel, stride in zip(in_channels, out_channels, kernels, strides):
            conv_layer = self.get_conv_block(conv_block, in_channel, out_channel, kernel, stride, False, conv_type)
            layers.append(conv_layer)
        return nn.ModuleList(layers)

    def initialize_weights(self, module):
        name = module.__class__.__name__.lower()
        if name in ["conv2d", "conv3d"]:
            nn.init.kaiming_normal_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, 0)

    def get_output_block(self, decoder_level):
        return OutputBlock(in_channels=self.filters[decoder_level], out_channels=self.n_class, dim=self.dim)

    def get_deep_supervision_heads(self):
        return nn.ModuleList([self.get_output_block(1), self.get_output_block(2)])

    def initialize_weights(self, module):
        name = module.__class__.__name__.lower()
        if name in ["conv2d", "conv3d"]:
            nn.init.kaiming_normal_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.constant_(module.bias, 0)
