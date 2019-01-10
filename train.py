# coding=utf-8
import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm

from datasets import Folder, PriorFolder
from datasets.saliency import collate_more
from evaluate import fm_and_mae
from models import FCN, Net

home = os.path.expanduser("~")

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default='%s/Datasets/DUTS/Train' % home)
# 这里的先验文件夹是另外的一个文件夹, 存放被训练好的原始显著性检测网络的生成的预测结果
# 里面只存放着.png格式的文件
parser.add_argument('--prior_dir', default='output')
parser.add_argument('--val_dir', default='%s/Datasets/ECSSD/' % home)
parser.add_argument('--base', default='vgg16')  # training dataset
parser.add_argument('--img_size', type=int, default=256)  # batch size
parser.add_argument('--b', type=int, default=8)  # batch size
parser.add_argument('--max', type=int, default=100000)  # epoches
parser.add_argument('--use_tensorboard', type=bool, default=False)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def make_image_grid(img, mean, std):
    """
    以网格的形式显示图片数据
    
    :param img:
    :type img:
    :param mean:
    :type mean:
    :param std:
    :type std:
    :return:
    :rtype:
    """
    img = make_grid(img)
    for i in range(3):
        img[i] *= std[i]
        img[i] += mean[i]
    return img


def validate(loader, net, output_dir, gt_dir):
    """
    验证集的测试
    
    :param loader:
    :type loader:
    :param net:
    :type net:
    :param output_dir:
    :type output_dir:
    :param gt_dir:
    :type gt_dir:
    :return:
    :rtype:
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    net.eval()
    
    # 获取验证集数据, 这里用进度条包装了一下
    loader = tqdm(loader, desc='validating')
    for ib, (data, lbl, prior, img_name, w, h) in enumerate(loader):
        with torch.no_grad():
            outputs = net(data.cuda(), prior[:, None].cuda())
            outputs = F.sigmoid(outputs)
        
        outputs = outputs.squeeze(1).cpu().numpy()
        # 从0~1放缩到0~255
        outputs *= 255
        for ii, msk in enumerate(outputs):
            msk = Image.fromarray(msk.astype(np.uint8))
            msk = msk.resize((w[ii], h[ii]))
            msk.save('{}/{}.png'.format(output_dir, img_name[ii]), 'PNG')
    
    # 对两个文件夹中的图片进行评估
    fm, mae, _, _ = fm_and_mae(output_dir, gt_dir)
    
    net.train()
    
    return fm, mae


def main(opt, name):
    check_dir = 'output/' + name
    if not os.path.exists(check_dir):
        os.mkdir(check_dir)
    
    net = Net(base=opt.base)
    
    """
    # data for FCN 先训练FCN, 利用FCN生成先验图 ###############################
    val_loader = torch.utils.data.DataLoader(
        PriorFolder(opt.val_dir, opt.prior_dir, size=256,
                    mean=mean, std=std),
        batch_size=opt.b * 3, shuffle=False, num_workers=4, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(
        Folder(opt.train_dir, scales=[64] * 3 + [128, 256],
               crop=0.9, flip=True, rotate=None,
               mean=mean, std=std),
        collate_fn=collate_more,
        batch_size=opt.b * 6, shuffle=True, num_workers=4, pin_memory=True)
    
    fcn = FCN(net)
    fcn = fcn.cuda()
    # sdict =torch.load('/home/crow/LPSfiles/Train2_vgg16/fcn-iter13800.pth')
    # fcn.load_state_dict(sdict)
    fcn.train()
    
    optimizer = torch.optim.Adam([{'params': fcn.parameters(), 'lr': 1e-4}])
    logs = {'best_it': 0, 'best': 0}
    
    sal_data_iter = iter(train_loader)
    i_sal_data = 0
    for it in tqdm(range(opt.max)):
        if i_sal_data >= len(train_loader):
            sal_data_iter = iter(train_loader)
            i_sal_data = 0
        
        data, lbls, _ = sal_data_iter.next()
        i_sal_data += 1
        
        data = data.cuda()
        lbls = [lbl.unsqueeze(1).cuda() for lbl in lbls]
        msks = fcn(data)
        
        loss = sum([F.binary_cross_entropy_with_logits(msk, lbl)
                    for msk, lbl in zip(msks, lbls)])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if opt.use_tensorboard and it % 10 == 0:
            writer.add_scalar('loss', loss.item(), it)
            
            image = make_image_grid(data[:6], mean, std)
            writer.add_image('Image', torchvision.utils.make_grid(image), it)
            
            big_msk = F.sigmoid(msks[-1]).expand(-1, 3, -1, -1)
            writer.add_image('msk',
                             torchvision.utils.make_grid(big_msk.data[:6]), it)
                             
            big_msk = lbls[-1].expand(-1, 3, -1, -1)
            writer.add_image('gt',
                             torchvision.utils.make_grid(big_msk.data[:6]), it)

        if it != 0 and it % 100 == 0:
            fm, mae = validate(val_loader, fcn,
                               os.path.join(check_dir, 'results'),
                               os.path.join(opt.val_dir, 'masks'))
            
            print(u'损失: %.4f' % (loss.item()))
            print(u'最大FM: iteration %d的%.4f, 这次FM: %.4f' % (
                logs['best_it'], logs['best'], fm))
            
            logs[it] = {'FM': fm}
            if fm > logs['best']:
                logs['best'] = fm
                logs['best_it'] = it
                torch.save(fcn.state_dict(), '%s/fcn-best.pth' % (check_dir))
                
            with open(os.path.join(check_dir, 'logs.json'), 'w') as outfile:
                json.dump(logs, outfile)
                
            torch.save(fcn.state_dict(), '%s/fcn-iter%d.pth' % (check_dir, it))
    """
    
    # 再训练新提出的结构 #########################################################
    
    # 准备数据
    val_loader = torch.utils.data.DataLoader(
        PriorFolder(opt.val_dir, opt.prior_dir, size=256,
                    mean=mean, std=std),
        batch_size=opt.b, shuffle=False, num_workers=4, pin_memory=True)
    
    train_loader = torch.utils.data.DataLoader(
        Folder(opt.train_dir, scales=256,
               crop=0.9, flip=True, rotate=None,
               mean=mean, std=std),
        collate_fn=collate_more,
        batch_size=opt.b, shuffle=True, num_workers=4, pin_memory=True)
    
    net = net.cuda()
    net.train()
    
    optimizer = torch.optim.Adam([{'params': net.parameters(), 'lr': 1e-4}])
    
    logs = {'best_it': 0, 'best': 0}
    p = 5
    
    # 转换为迭代器
    sal_data_iter = iter(train_loader)
    
    i_sal_data = 0
    for it in tqdm(range(opt.max)):
        if i_sal_data >= len(train_loader):
            sal_data_iter = iter(train_loader)
            i_sal_data = 0
        
        # 迭代计数
        i_sal_data += 1
        
        # 获取一个batch的数据和各个尺寸的真值, 训练集只返回一个尺寸的真值
        data, lbl, _ = sal_data_iter.next()
        data = data.cuda()
        lbl = lbl[0].unsqueeze(1)
        
        # 随机翻转标签用的掩膜, 这里使用n=1的伯努利分布(二项分布, 非0即1, p为1的概率), 与原
        # 本的真值的二指图像加和, 对2取余数, 这里改变的就是原始为1, 而且对应的扰动也为1的位置
        # 以及真值为0, 而且扰动为1的位置, 总体来说就是改变了扰动为1的位置. 对其进行了翻转
        noisy_label = (lbl.numpy() + np.random.binomial(
            1, float(p) / 100.0, (256, 256))) % 2
        
        noisy_label = torch.Tensor(noisy_label).cuda()
        # 输出的是像素与前景背景向量之间的距离(dist_back - dist_fore)
        msk = net(data, noisy_label)
        # 这里的输出可能有负值吧?
        print([msk < 0])
        
        lbl = lbl.cuda()
        # 计算二值交叉熵损失, 因为获得的距离有正有负, 进过该函数的内置的sigmoid之后, 变为大
        # 于0.5和小于0.5的值, 这样就可以可以认为是msk各像素是显著性区域的概率, 负对数损失总
        # 体变小, 也就是数据更为接近真正预测
        loss = F.binary_cross_entropy_with_logits(msk, lbl)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 每十个batch就记录一下数据
        if opt.use_tensorboard and it % 10 == 0:
            writer.add_scalar('loss', loss.item(), it)
            
            image = make_image_grid(data[:6], mean, std)
            writer.add_image('Image', torchvision.utils.make_grid(image), it)
            
            big_msk = F.sigmoid(msk).expand(-1, 3, -1, -1)
            writer.add_image('msk',
                             torchvision.utils.make_grid(big_msk.data[:6]), it)
            
            big_msk = lbl.expand(-1, 3, -1, -1)
            writer.add_image('gt',
                             torchvision.utils.make_grid(big_msk.data[:6]), it)
        
        # 每100个batch就记验证集上从测试一下
        if it != 0 and it % 100 == 0:
            fm, mae = validate(val_loader, net,
                               os.path.join(check_dir, 'results'),
                               os.path.join(opt.val_dir, 'masks'))
            print(u'损失: %.4f' % (loss.item()))
            print(u'最大FM: iteration %d的%.4f, 这次FM: %.4f' % (
                logs['best_it'], logs['best'], fm))
            
            logs[it] = {'FM': fm}
            if fm > logs['best']:
                logs['best'] = fm
                logs['best_it'] = it
                torch.save(net.state_dict(), '%s/net-best.pth' % (check_dir))
            
            with open(os.path.join(check_dir, 'logs.json'), 'w') as outfile:
                json.dump(logs, outfile)
            
            torch.save(net.state_dict(), '%s/net-iter%d.pth' % (check_dir, it))


if __name__ == "__main__":
    opt = parser.parse_args()
    print(opt)
    
    name = 'Train_{}'.format(opt.base)
    # tensorboard writer
    if opt.use_tensorboard:
        # os.system('rm -rf ./output/runs_%s/*' % name)
        writer = SummaryWriter(
            './runs_%s/' % name + datetime.now().strftime('%B%d  %H:%M:%S'))
        if not os.path.exists('./runs_%s' % name):
            os.mkdir('./runs_%s' % name)
    
    main(opt, name)
