import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd.variable import Variable

# from .densenet import *
# from .resnet import *
from .vgg import *

# from densenet import *
# from resnet import *
# from vgg import *

import numpy as np
import sys

thismodule = sys.modules[__name__]
import pdb

img_size = 256

dim_dict = {'vgg': [64, 128, 256, 512, 512]}


def proc_vgg(model):
    """
    对于预训练的VGG网络进行调整
    
    :param model:
    :type model:
    :return:
    :rtype:
    """
    
    # The hook will be called every time after forward() has computed an output.
    # It should have the following signature:
    # `hook(module, input, output) -> None`
    # The hook should not modify the input or output.
    def hook(module, input, output):
        # 在模块每次执行完一次forward()后, 将output收集起来
        model.feats[output.device.index] += [output]
    
    # 排除最后一层
    for m in model.features[:-1]:
        # 这里是在收集每个block的最后的ReLU的输出, 正好为尚未进入MaxPool2d
        m[-2].register_forward_hook(hook)
    
    def remove_sequential(all_layers, network):
        """
        从network中递归的方式收集各层的模块, 汇总到all_layers里, 移除外部Sequential结构
        
        :param all_layers:
        :type all_layers:
        :param network:
        :type network:
        :return:
        :rtype:
        """
        for layer in network.children():
            # if sequential layer, apply recursively to layers in sequential layer
            if isinstance(layer, nn.Sequential):
                remove_sequential(all_layers, layer)
            if list(layer.children()) == []:  # if leaf node, add it to list
                all_layers.append(layer)

    # 这里并没有完全的删掉每个模块的最大池化, 而是替换其步长和核大小为1来将其变为恒等变换层
    
    # 修改MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    model.features[2][-1].stride = 1
    model.features[2][-1].kernel_size = 1
    
    all_layers = []
    # 提取model.features[3], 收集所有的子模块, 将卷积参数更新为扩张卷积
    remove_sequential(all_layers, model.features[3])
    for m in all_layers:
        if isinstance(m, nn.Conv2d):
            m.dilation = (2, 2)
            m.padding = (2, 2)
    model.features[3][-1].stride = 1
    model.features[3][-1].kernel_size = 1
    
    all_layers = []
    # 这里使用扩张率更大的扩张卷积
    remove_sequential(all_layers, model.features[4])
    for m in model.features[4]:
        if isinstance(m, nn.Conv2d):
            m.dilation = (4, 4)
            m.padding = (4, 4)
    model.features[4][-1].stride = 1
    model.features[4][-1].kernel_size = 1

    return model


procs = {'vgg16': proc_vgg, }


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """
    Make a 2D bilinear kernel suitable for upsampling
    
    :param in_channels:
    :type in_channels:
    :param out_channels:
    :type out_channels:
    :param kernel_size:
    :type kernel_size:
    :return:
    :rtype:
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()


class Net(nn.Module):
    def __init__(self, pretrained=True, base='vgg16'):
        super(Net, self).__init__()
        if 'vgg' in base:
            # 倒叙, 第二个冒号之后是步长
            dims = dim_dict['vgg'][::-1]
        else:
            dims = dim_dict[base][::-1]
        
        self.base = base
        # [ C, C, C, C, C ]
        odims = [64] * 5
        # D
        hdim = 512
        
        self.classifier = nn.Linear(512, 1)
        
        # dim_dict = {'vgg': [64, 128, 256, 512, 512]}
        # dims是dim_dict的反向排序
        self.proc_feats_list = nn.ModuleList([
            # convtranspose2d=>out_size=(in_size-1)xstride-2xpadding+kernel_size
            nn.Sequential(
                # x4 512卷积块对应的输出
                nn.ConvTranspose2d(dims[0], dims[0], 8, 4, 2),
                nn.Conv2d(dims[0], odims[0], kernel_size=3, padding=1)),
            nn.Sequential(
                # x4 512对应的输出
                nn.ConvTranspose2d(dims[1], dims[1], 8, 4, 2),
                nn.Conv2d(dims[1], odims[1], kernel_size=3, padding=1)),
            nn.Sequential(
                # x4 256对应的输出
                nn.ConvTranspose2d(dims[2], dims[2], 8, 4, 2),
                nn.Conv2d(dims[2], odims[2], kernel_size=3, padding=1)),
            nn.Sequential(
                # x2 128对应的输出
                nn.ConvTranspose2d(dims[3], dims[3], 4, 2, 1),
                nn.Conv2d(dims[3], odims[3], kernel_size=3, padding=1)),
            
            # 使用亚像素卷积实现上采样 #############################################
            # 不清楚这里为什么放弃了使用亚像素卷积的手段
            # 这里的nn.PixelShuffle(up_scale)便是可以用来实现亚像素卷积的一个类
            # nn.Sequential(
            #     nn.Conv2d(dims[0], odims[0], kernel_size=3, padding=1),
            #     nn.PixelShuffle(4)),
            # nn.Sequential(
            #     nn.Conv2d(dims[1], odims[1], kernel_size=3, padding=1),
            #     nn.PixelShuffle(4)),
            # nn.Sequential(
            #     nn.Conv2d(dims[2], odims[2], kernel_size=3, padding=1),
            #     nn.PixelShuffle(4)),
            # nn.Sequential(
            #     nn.Conv2d(dims[3], odims[3], kernel_size=3, padding=1),
            #     nn.PixelShuffle(2)),
            
            # 64 对应的输出
            nn.Conv2d(dims[4], dims[4], kernel_size=3, padding=1),
        ])
        
        # 5C->D
        self.proc_feats = nn.Conv2d(sum(odims), hdim, kernel_size=3, padding=1)
        # 5C->D
        self.proc_mul = nn.Conv2d(sum(odims), hdim, kernel_size=3, padding=1)
        
        # 初始化各层
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)
        
        # 这里就是调用models.vgg16, 相当于, self.feature=models.vgg16
        self.feature = getattr(thismodule, base)(pretrained=pretrained)
        
        self.feature.feats = {}
        
        # procs_vgg(self.feature)
        self.feature = procs[base](self.feature)
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad = False
    
    def forward(self, x, prior):
        """
        在已有架构的基础上, 对不同块的输出特征图上采样后进行拼接, 进行空间映射
        
        `pre_net + upsample + embedding(region, pixel)`
        
        :param x: 网络的输入图像
        :type x:
        :param prior: 先验显著性图
        :type prior:
        :return: 测度空间中, 像素点到两类锚点的距离差
        :rtype:
        """
        # 计算VGG网络的输出
        self.feature.feats[x.device.index] = []
        
        out_feats_net = self.feature(x)
        
        # 这里的feats究竟是什么样的存在? 针对不同的设备(多卡), 处理不同的输出
        feats = self.feature.feats[out_feats_net.device.index]
        
        feats += [out_feats_net]
        feats = feats[::-1]
        # 获取各个块的输出特征图
        for i, block in enumerate(self.proc_feats_list):
            feats[i] = block(feats[i])
        
        # 这里应该是将对应的五个输出拼接出来的结果
        feats = torch.cat(feats, 1)
        
        # region embedding #####################################################
        # 这里对先验进行更新, 卷积层的输入是5x64-channel的这样一个特征图组合与先验图的乘积,
        # 输出之后在宽高的维度都进行了求和, 输出变成了(B, 512, 1, 1)的特征图
        # 这里选择的是前景
        c1 = self.proc_mul(feats * prior).sum(3, keepdim=True).sum(
            2, keepdim=True) / (prior.sum())
        # 这里在选择背景, 输出形状是一致的
        c2 = self.proc_mul(feats * (1 - prior)).sum(3, keepdim=True).sum(
            2, keepdim=True) / ((1 - prior).sum())
        
        # pisel embedding ######################################################
        # 这个直接作用了输出的5C-channel上
        feats = self.proc_feats(feats)
        
        # 计算测度空间中, 像素对应的特征向量与前景背景对应的锚点的距离, 距离小的就划归到那一类
        dist1 = (feats - c1) ** 2
        dist1 = torch.sqrt(dist1.sum(dim=1, keepdim=True))
        dist2 = (feats - c2) ** 2
        dist2 = torch.sqrt(dist2.sum(dim=1, keepdim=True))
        
        # 输出与背景的距离减去与前景的距离
        return dist2 - dist1


if __name__ == "__main__":
    net = Net()
    net.cuda()
    x = torch.Tensor(2, 3, 256, 256).cuda()
    z = torch.Tensor(2, 1, 256, 256).cuda()
    sb = net(x, z)
    pdb.set_trace()
