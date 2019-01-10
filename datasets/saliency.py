import os
import numpy as np
import PIL.Image as Image
import torch
from torch.utils import data
import pdb
import random


def rotated_rect_with_max_area(w, h, angle):
    """
  Given a rectangle of size wxh that has been rotated by 'angle' (in
  radians), computes the width and height of the largest possible
  axis-aligned rectangle (maximal area) within the rotated rectangle.
  """
    if w <= 0 or h <= 0:
        return 0, 0
    
    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)
    
    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(np.sin(angle)), abs(np.cos(angle))
    if side_short <= 2. * sin_a * cos_a * side_long \
        or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) \
            if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, \
                 (h * cos_a - w * sin_a) / cos_2a
    
    return wr, hr


class BaseFolder(data.Dataset):
    def __init__(self, root, crop=None, rotate=None, flip=False,
                 mean=None, std=None):
        super(BaseFolder, self).__init__()
        
        self.mean, self.std = mean, std
        self.flip = flip
        self.rotate = rotate
        self.crop = crop
        
        img_dir = os.path.join(root, 'images')
        gt_dir = os.path.join(root, 'masks')
        # 确定样本名字
        names = ['.'.join(name.split('.')[:-1]) for name in os.listdir(gt_dir)]
        
        self.img_filenames = [os.path.join(img_dir, name + '.jpg')
                              for name in names]
        self.gt_filenames = [os.path.join(gt_dir, name + '.png')
                             for name in names]
        self.names = names
    
    def random_crop(self, *images):
        """
        对输入的Image对象进行随机裁剪操作
        
        :param images:
        :type images:
        :return:
        :rtype:
        """
        images = list(images)
        sz = [img.size for img in images]
        # list、set、dict：是不可哈希的
        # int、float、str、tuple：是可以哈希的
        # set的可哈希，不可哈希，是对可迭代类型（iterables）所存储元素（elements）的要求，
        # [1, 2, 3]是可迭代类型，其存储元素的类型为int，是可哈希的，如果set([[1, 2],
        # [3, 4]])，[[1, 2], [3, 4]]list of lists（list 构成的 list）自然是可迭代的，
        # 但其元素为 [1, 2] 和 [3, 4]是不可哈希的
        # 这里的sz的元素是元组, 是可以哈希的
        sz = set(sz)
        # 确保图像大小一致
        assert (len(sz) == 1)
        
        # 当集合是由列表和元组组成时,set.pop()是从排好序后的集合的左边删除元素的
        # 对于是字典和字符转换的集合是随机删除元素的
        w, h = sz.pop()
        # 剪裁的边长
        th, tw = int(self.crop * h), int(self.crop * w)
        # 不需要剪裁
        if w == tw and h == th:
            return 0, 0, h, w
        
        # 确定剪裁内容的左上角坐标
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        
        results = [img.crop((j, i, j + tw, i + th)) for img in images]
        return tuple(results)
    
    def random_flip(self, *images):
        """
        这里直接对整体批次进行了翻转
        
        :param images:
        :type images:
        :return:
        :rtype:
        """
        if self.flip and random.randint(0, 1):
            images = list(images)
            print(images)
            results = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in images]
            return tuple(results)
        else:
            return images
    
    def random_rotate(self, *images):
        """
        实现随机旋转
        
        :param images:
        :type images:
        :return:
        :rtype:
        """
        images = list(images)
        sz = [img.size for img in images]
        sz = set(sz)
        assert (len(sz) == 1)
        
        w, h = sz.pop()
        degree = random.randint(-1 * self.rotate, self.rotate)
        images_r = [img.rotate(degree, expand=1) for img in images]
        w_b, h_b = images_r[0].size
        w_r, h_r = rotated_rect_with_max_area(w, h, np.radians(degree))
        
        ws = (w_b - w_r) / 2
        ws = max(ws, 0)
        hs = (h_b - h_r) / 2
        hs = max(hs, 0)
        
        we = ws + w_r
        he = hs + h_r
        we = min(we, w_b)
        he = min(he, h_b)
        
        results = [img.crop((ws, hs, we, he)) for img in images_r]
        
        return tuple(results)
    
    def __len__(self):
        return len(self.names)


class PriorFolder(BaseFolder):
    def __init__(self, root, prior_dir, size=256,
                 crop=None, rotate=None, flip=False,
                 mean=None, std=None):
        # 初始化父类方法
        super(PriorFolder, self).__init__(root,
                                          crop=crop, rotate=rotate, flip=flip,
                                          mean=mean, std=std)
        self.size = size
        self.pr_filenames = [os.path.join(prior_dir, name + '.png')
                             for name in self.names]
    
    def __getitem__(self, index):
        # 载入数据
        name = self.names[index]
        img_file = self.img_filenames[index]
        img = Image.open(img_file)
        gt_file = self.gt_filenames[index]
        gt = Image.open(gt_file)
        pr_file = self.pr_filenames[index]
        pr = Image.open(pr_file)
        
        # 调整先验和图像的尺寸
        WW, HH = gt.size
        img = img.resize((WW, HH))
        pr = pr.resize((WW, HH))
        
        # 一次送入一组数据(一副图片对应的数据), 剪裁旋转翻转数据
        if self.crop is not None:
            img, gt, pr = self.random_crop(img, gt, pr)
        if self.rotate is not None:
            img, gt, pr = self.random_rotate(img, gt, pr)
        if self.flip:
            img, gt, pr = self.random_flip(img, gt, pr)
        
        img, gt, pr = [_img.resize((self.size, self.size))
                       for _img in [img, gt, pr]]
        
        gt = np.array(gt, dtype=np.uint8)
        # 对真值二值化
        gt[gt != 0] = 1
        img = np.array(img, dtype=np.float64) / 255.0
        pr = np.array(pr, dtype=np.float64) / 255.0
        
        if len(gt.shape) > 2:
            gt = gt[:, :, 0]
        if len(img.shape) < 3:
            img = np.stack((img, img, img), 2)
        if img.shape[2] > 3:
            img = img[:, :, :3]
        if self.mean is not None:
            img -= self.mean
        if self.std is not None:
            img /= self.std
        
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        gt = torch.from_numpy(gt).float()
        pr = torch.from_numpy(pr).float()
        
        return img, gt, pr, name, WW, HH


class Folder(BaseFolder):
    def __init__(self, root, scales=[256],
                 crop=None, rotate=None, flip=False,
                 mean=None, std=None):
        super(Folder, self).__init__(root,
                                     crop=crop, rotate=rotate, flip=flip,
                                     mean=mean, std=std)
        self.scales = scales
    
    def __getitem__(self, index):
        # load image
        name = self.names[index]
        img_file = self.img_filenames[index]
        img = Image.open(img_file)
        gt_file = self.gt_filenames[index]
        gt = Image.open(gt_file)
        
        # 对图片进行调整
        WW, HH = gt.size
        img = img.resize((WW, HH))
        if self.crop is not None:
            img, gt = self.random_crop(img, gt)
        if self.rotate is not None:
            img, gt = self.random_rotate(img, gt)
        if self.flip:
            img, gt = self.random_flip(img, gt)
        
        # 图像放缩到最大的尺寸
        max_size = max(self.scales)
        img = img.resize((max_size, max_size))
        # 真值挨个放缩到特定尺寸
        gts = [gt.resize((s, s)) for s in self.scales]
        
        img = np.array(img, dtype=np.float64) / 255.0
        gts = [np.array(gt, dtype=np.uint8) for gt in gts]
        
        # 挨个二值化
        for gt in gts: gt[gt != 0] = 1
        
        # 确保图像维度为三维
        if len(img.shape) < 3:
            img = np.stack((img, img, img), 2)
        if img.shape[2] > 3:
            img = img[:, :, :3]
        # 确保真值只有一个通道
        for i, gt in enumerate(gts):
            if len(gt.shape) > 2:
                gts[i] = gt[:, :, 0]
        
        if self.mean is not None:
            img -= self.mean
        if self.std is not None:
            img /= self.std
        
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        gts = [torch.from_numpy(gt).float() for gt in gts]
        
        return img, gts, name


def collate_more(data):
    images, gts, name = zip(*data)
    gts = list(map(list, zip(*gts)))
    
    images = torch.stack(images, 0)
    gts = [torch.stack(gt, 0) for gt in gts]
    
    return images, gts, name


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    random.seed(datetime.now())
    sb = Folder('/home/zeng/data/datasets/saliency_Dataset/ECSSD',
                crop=None, rotate=10, flip=True)
    img, gt, _ = sb.__getitem__(random.randint(0, 1000))
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.show()
    plt.imshow(gt[0])
    plt.show()
    
    # sb = PriorFolder('/home/zeng/data/datasets/saliency_Dataset/ECSSD',
    #                  '/home/zeng/data/datasets/saliency_Dataset/results/ECSSD-Sal/SRM',
    #                  crop=None, rotate=None, flip=True, size=256)
    # img, gt, pr, _, _, _ = sb.__getitem__(0)
    # img = img.numpy().transpose((1, 2, 0))
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(gt)
    # plt.show()
    # plt.imshow(pr)
    # plt.show()
