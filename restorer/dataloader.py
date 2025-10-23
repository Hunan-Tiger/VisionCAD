import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class ImageRestorationDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None, img_size=(384,384)):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.img_size = img_size

        self.filenames = sorted(os.listdir(self.input_dir))


    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # 获取文件路径
        img_name = self.filenames[idx]
        input_path = os.path.join(self.input_dir, img_name)

        # 从文件名中提取前缀
        splitted = os.path.splitext(img_name)[0]
        parts = splitted.split('_')
        # 取前 2 段为前缀并加上 .png
        base_name = '_'.join(parts[:2]) + '.png'

        # 组装目标路径
        target_path = os.path.join(self.target_dir, base_name)

        # 改为灰度图
        input_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)


        # 调整大小
        target_img = cv2.resize(target_img, self.img_size)
        input_img = cv2.resize(input_img, self.img_size)


        # Convert to tensor [H,W] -> [1,H,W]
        input_img = torch.FloatTensor(input_img).unsqueeze(0) / 255.0
        target_img = torch.FloatTensor(target_img).unsqueeze(0) / 255.0

        return input_img, target_img
    