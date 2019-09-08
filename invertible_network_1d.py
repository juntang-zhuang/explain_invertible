from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import matplotlib.pyplot as plt
import torch.nn as nn
import math
import torch.nn.functional as F
import torch
import numpy as np
#from .zero_padding import DownsamplePad
#from .revop import ReversibleBlock
#from .model_utils import psi, injective_pad, MyPad
__all__ = ['invertible_resnet']


class ReversibleBlock(nn.Module):
    def __init__(self,Fm,Gm):
        super(ReversibleBlock,self).__init__()
        self.Fm = Fm
        self.Gm = Gm
    def forward(self, x):
        N,C = x.size()
        x1 = x[:,:C//2,...].contiguous()
        x2 = x[:,C//2:,...].contiguous()

        tmp1 = self.Fm(x2)
        y1 = x1 + tmp1

        tmp2 = self.Gm(y1)
        y2 = x2 + tmp2

        out = torch.cat((y1,y2),dim=1)
        return out

    def inverse(self,y):
        N, C = y.size()
        y1 = y[:, :C // 2, ...].contiguous()
        y2 = y[:, C // 2:, ...].contiguous()

        tmp1 = self.Gm(y1)
        x2 = y2 - tmp1

        tmp2 = self.Fm(x2)
        x1 = y1 - tmp2

        out = torch.cat((x1,x2),dim=1)
        return out

def conv3x3(in_planes, out_planes, stride=1, kernel_size = 3):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=(kernel_size-1)//2, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def batch_norm(input):
    """match Tensorflow batch norm settings"""
    return nn.BatchNorm2d(input, momentum=0.99, eps=0.001)



class RevBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, noactivation=False, kernel_size = 3):
        super(RevBasicBlock, self).__init__()
        if downsample is None and stride == 1:
            Gm = BasicBlockSub(inplanes // 2, planes // 2, stride, noactivation, kernel_size = kernel_size)
            Fm = BasicBlockSub(inplanes // 2, planes // 2, stride, noactivation, kernel_size = kernel_size)
            self.revblock = ReversibleBlock(Gm, Fm)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.revblock(x)
        return out

    def inverse(self,y):
        out = self.revblock.inverse(y)
        return out



class BasicBlockSub(nn.Module):
    def __init__(self, inplanes, planes, stride=1, noactivation=False, kernel_size = 3):
        super(BasicBlockSub, self).__init__()
        self.noactivation = noactivation
        self.fc1 = nn.Linear(in_features=inplanes,out_features=planes*4,bias=True)
        self.fc2 = nn.Linear(in_features=planes*4, out_features=planes, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        x = torch.relu(x)
        return x

class InvertibleResNet(nn.Module):

    def __init__(self, depth, planes = 16,layers = [3,3,3],num_classes=1000,inplanes=3,
                 batch_norm_fix = False, implementation = 0, block = RevBasicBlock, code_dim = None):
        super(InvertibleResNet, self).__init__()
        self.batch_norm_fix = batch_norm_fix
        self.implementation = implementation
        self.num_classes = num_classes
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        self.input_planes = inplanes

        self.conv_feature = ConvFeature(depth, planes = planes,layers = layers,num_classes=num_classes,
                                        inplanes=inplanes,
                 batch_norm_fix = batch_norm_fix, implementation = implementation, block = block)

        self.inplanes = inplanes

        self.fc = nn.Linear(self.inplanes, num_classes)

        # self.configure()
        self.init_weights()

    def init_weights(self):
        """Initialization using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.reset_parameters()

    def forward(self, x,  out_code = False):
        #N,C,H,W = x.size()
        #inputs = x
        x = self.conv_feature(x)

        encode = x

        pred = self.fc(encode.view(encode.size(0),-1))

        return pred

    def inverse(self,y):
        y = self.conv_feature.inverse(y)
        return y

    def test_inverse(self,x):
        # forward pass
        x0 = x
        x = self.conv_feature(x)
        x_reverse = self.conv_feature.inverse(x)

        residual = x0 - x_reverse
        residual = residual.data.cpu().numpy()
        out  = np.sum(residual**2)
        return out

class ConvFeature(nn.Module):

    def __init__(self, depth, planes = 16,layers = [3,3,3],num_classes=1000,inplanes=3,
                 batch_norm_fix = False, implementation = 0, block = RevBasicBlock):
        super(ConvFeature, self).__init__()
        self.batch_norm_fix = batch_norm_fix
        self.implementation = implementation
        # Model type specifies number of layers for CIFAR-10 model
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        n = (depth - 2) // 6

        self.input_planes = inplanes
        self.inplanes = inplanes

        self.block1 = block(self.inplanes,self.inplanes)
        self.block2 = block(self.inplanes,self.inplanes)
        self.block3 = block(self.inplanes,self.inplanes)

        self.block4 = block(self.inplanes, self.inplanes)
        self.block5 = block(self.inplanes, self.inplanes)
        self.block6 = block(self.inplanes, self.inplanes)

        #self.configure()
        self.init_weights()

    def init_weights(self):
        """Initialization using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.reset_parameters()

    def forward(self, x,  out_code = False):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        return x

    def inverse(self,y):
        y = self.block6.inverse(y)
        y = self.block5.inverse(y)
        y = self.block4.inverse(y)

        y = self.block3.inverse(y)
        y = self.block2.inverse(y)
        y = self.block1.inverse(y)

        return y

class ReverseConv2(nn.Module):
    def __init__(self, mod):
        super(ReverseConv2, self).__init__()
        self.mod = mod
    def forward(self, x):
        return self.mod.inverse(x)



def invertible_resnet(depth, num_classes,**kwargs):
    """
    Constructs a ResNet model.
    """
    if depth == 10:
        return InvertibleResNet(depth=depth,num_classes=num_classes, layers =[1, 1, 1],
                     **kwargs)
    elif depth == 20:
        return InvertibleResNet(depth=depth,num_classes=num_classes, layers =[3, 3, 3],
                     **kwargs)
    elif depth == 26:
        return InvertibleResNet(depth=depth, num_classes=num_classes, layers=[3, 3, 3],
                       **kwargs)
    elif depth == 38:
        return InvertibleResNet(depth=depth, num_classes=num_classes, layers=[6, 6, 6],
                      **kwargs)
    elif depth == 56:
        return InvertibleResNet(depth=depth, num_classes=num_classes, layers=[9, 9, 9],
                       **kwargs)
    elif depth == 110:
        return InvertibleResNet(depth=depth, num_classes=num_classes, layers=[18, 18, 18],
                      **kwargs)
    else:
        return InvertibleResNet(depth=depth, num_classes=num_classes, layers=[3, 3, 3],
                       **kwargs)
