import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

class ScreenGlareEffect(nn.Module):
    """模拟屏幕反光效果"""
    def __init__(self, p=0.5, max_intensity=0.3):
        super().__init__()
        self.p = p
        self.max_intensity = max_intensity
        
    def forward(self, img):
        if torch.rand(1) > self.p:
            return img
            
        # 创建反光斑点
        h, w = img.shape[-2:]
        center_x = torch.randint(w//4, w*3//4, (1,))
        center_y = torch.randint(h//4, h*3//4, (1,))
        
        # 生成高斯反光
        x = torch.arange(w)
        y = torch.arange(h)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        
        sigma = min(h, w) // 8
        glare = torch.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2 * sigma**2))
        glare = glare * torch.rand(1) * self.max_intensity
        
        # 应用到所有通道
        glare = glare.unsqueeze(0).repeat(img.shape[0], 1, 1)
        return torch.clamp(img + glare, 0, 1)

class ScreenMoireEffect(nn.Module):
    """模拟莫尔纹效果"""
    def __init__(self, p=0.3):
        super().__init__()
        self.p = p
        
    def forward(self, img):
        if torch.rand(1) > self.p:
            return img
            
        # 创建网格pattern
        h, w = img.shape[-2:]
        x = torch.arange(w)
        y = torch.arange(h)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        
        # 随机频率和角度
        freq = torch.rand(1) * 0.5 + 0.5  # 0.5-1.0
        angle = torch.rand(1) * np.pi
        
        # 生成莫尔纹pattern
        pattern = torch.sin(freq * (X * torch.cos(angle) + Y * torch.sin(angle)))
        pattern = pattern * 0.1  # 控制强度
        
        # 应用到所有通道
        pattern = pattern.unsqueeze(0).repeat(img.shape[0], 1, 1)
        return torch.clamp(img * (1 + pattern), 0, 1)

class ScreenCurveDistortion(nn.Module):
    """模拟CRT屏幕的曲面效果"""
    def __init__(self, p=0.3, distortion_scale=0.1):
        super().__init__()
        self.p = p
        self.distortion_scale = distortion_scale
        
    def forward(self, img):
        if torch.rand(1) > self.p:
            return img
            
        h, w = img.shape[-2:]
        
        # 创建网格
        x = torch.linspace(-1, 1, w)
        y = torch.linspace(-1, 1, h)
        X, Y = torch.meshgrid(x, y, indexing='xy')
        
        # 应用曲面变形
        r = torch.sqrt(X**2 + Y**2)
        displacement = r**2 * self.distortion_scale
        
        X_distorted = X * (1 + displacement)
        Y_distorted = Y * (1 + displacement)
        
        # 创建采样网格
        grid = torch.stack([X_distorted, Y_distorted], dim=2)
        grid = grid.unsqueeze(0)
        
        # 使用grid_sample进行变形
        return F.grid_sample(img.unsqueeze(0), grid, mode='bilinear', padding_mode='border').squeeze(0)

# 组合所有变换
def get_screen_photo_transforms(p=0.5):
    return nn.Sequential(
        # 基础变换
        TF.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
        TF.RandomRotation(degrees=10),
        
        # 自定义效果
        ScreenGlareEffect(p=p*0.7),
        ScreenMoireEffect(p=p*0.5),
        ScreenCurveDistortion(p=p*0.3),
        
        # 颜色和清晰度调整
        TF.ColorJitter(brightness=0.2, contrast=0.2),
        TF.RandomAdjustSharpness(sharpness_factor=0.5, p=p)
    )

# 使用示例
def apply_screen_photo_simulation(image_tensor):
    """
    应用所有屏幕拍照模拟效果
    
    Args:
        image_tensor (torch.Tensor): 输入图像张量 (C, H, W)，值范围[0,1]
    Returns:
        torch.Tensor: 变换后的图像张量
    """
    transforms = get_screen_photo_transforms()
    return transforms(image_tensor)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import math

class BoundingBoxTransform:
    """用于变换YOLO格式边界框的工具类"""
    @staticmethod
    def rotate_box(box, angle, image_size):
        """旋转边界框
        Args:
            box (np.array): [center_x, center_y, width, height] 归一化坐标
            angle (float): 旋转角度(度)
            image_size (tuple): (height, width)
        """
        # 转换为像素坐标
        h, w = image_size
        cx, cy = box[0] * w, box[1] * h
        bw, bh = box[2] * w, box[3] * h
        
        # 计算四个角点
        corners = np.array([
            [cx - bw/2, cy - bh/2],
            [cx + bw/2, cy - bh/2],
            [cx + bw/2, cy + bh/2],
            [cx - bw/2, cy + bh/2]
        ])
        
        # 旋转角点
        angle_rad = math.radians(angle)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        
        # 将角点移到原点进行旋转
        center = np.array([cx, cy])
        corners = corners - center
        corners = corners @ rotation_matrix.T
        corners = corners + center
        
        # 计算新的边界框
        min_x, min_y = np.min(corners, axis=0)
        max_x, max_y = np.max(corners, axis=0)
        
        # 转回归一化坐标
        new_cx = ((min_x + max_x) / 2) / w
        new_cy = ((min_y + max_y) / 2) / h
        new_w = (max_x - min_x) / w
        new_h = (max_y - min_y) / h
        
        return np.array([new_cx, new_cy, new_w, new_h])

    @staticmethod
    def apply_affine(box, matrix, image_size):
        """应用仿射变换到边界框
        Args:
            box (np.array): [center_x, center_y, width, height] 归一化坐标
            matrix (np.array): 2x3 仿射变换矩阵
            image_size (tuple): (height, width)
        """
        h, w = image_size
        cx, cy = box[0] * w, box[1] * h
        bw, bh = box[2] * w, box[3] * h
        
        # 计算四个角点
        corners = np.array([
            [cx - bw/2, cy - bh/2, 1],
            [cx + bw/2, cy - bh/2, 1],
            [cx + bw/2, cy + bh/2, 1],
            [cx - bw/2, cy + bh/2, 1]
        ])
        
        # 应用仿射变换
        new_corners = corners @ matrix.T
        
        # 计算新的边界框
        min_x, min_y = np.min(new_corners, axis=0)
        max_x, max_y = np.max(new_corners, axis=0)
        
        # 转回归一化坐标
        new_cx = ((min_x + max_x) / 2) / w
        new_cy = ((min_y + max_y) / 2) / h
        new_w = (max_x - min_x) / w
        new_h = (max_y - min_y) / h
        
        return np.array([new_cx, new_cy, new_w, new_h])
    
    

class ScreenPhotoAugmentation(nn.Module):
    """整合所有变换效果的类"""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.glare = ScreenGlareEffect(p=p*0.7)
        self.moire = ScreenMoireEffect(p=p*0.5)
        self.curve = ScreenCurveDistortion(p=p*0.3)
        self.box_transform = BoundingBoxTransform()
    
    def rotate_box(self, box, angle, image_size):
        """
        旋转边界框
        Args:
            box: [cx, cy, w, h] 归一化坐标
            angle: 旋转角度(顺时针为正)
            image_size: (height, width)
        """
        h, w = image_size
        cx, cy, bw, bh = box
        
        # 1. 转换到像素坐标
        cx, cy = cx * w, cy * h
        bw, bh = bw * w, bh * h
        
        # 2. 计算框的四个角点
        corners = np.array([
            [cx - bw/2, cy - bh/2],  # 左上
            [cx + bw/2, cy - bh/2],  # 右上
            [cx + bw/2, cy + bh/2],  # 右下
            [cx - bw/2, cy + bh/2]   # 左下
        ])
        
        # 3. 计算旋转矩阵（注意：TF.rotate是逆时针为正，所以这里取负）
        angle_rad = math.radians(-angle)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])
        
        # 4. 以图像中心为旋转中心
        image_center = np.array([w/2, h/2])
        corners = corners - image_center
        corners = corners @ rotation_matrix
        corners = corners + image_center
        
        # 5. 计算新的边界框
        min_xy = np.min(corners, axis=0)
        max_xy = np.max(corners, axis=0)
        
        # 6. 转换回归一化坐标
        new_cx = (min_xy[0] + max_xy[0]) / (2 * w)
        new_cy = (min_xy[1] + max_xy[1]) / (2 * h)
        new_w = (max_xy[0] - min_xy[0]) / w
        new_h = (max_xy[1] - min_xy[1]) / h
        
        # 7. 确保坐标在[0,1]范围内
        new_cx = np.clip(new_cx, 0, 1)
        new_cy = np.clip(new_cy, 0, 1)
        new_w = np.clip(new_w, 0, 1)
        new_h = np.clip(new_h, 0, 1)
        
        return np.array([new_cx, new_cy, new_w, new_h])
        
    def forward(self, img, boxes=None):
        """
        Args:
            img (torch.Tensor): (C, H, W) 图像张量
            boxes (np.array): (N, 5) YOLO格式标注框 [class_id, cx, cy, w, h]
        Returns:
            tuple: (变换后的图像, 变换后的标注框)
        """
        h, w = img.shape[-2:]
        
        # 应用其他效果
        img = self.glare(img)
        img = self.moire(img)
        #img = self.curve(img)
        
        # 颜色调整
        img = TF.adjust_brightness(img, brightness_factor=torch.rand(1).item() * 0.4 + 0.8)
        img = TF.adjust_contrast(img, contrast_factor=torch.rand(1).item() * 0.4 + 0.8)
        img = TF.adjust_sharpness(img, sharpness_factor=torch.rand(1).item() * 0.5 + 0.5)
        
        return img, boxes

# 使用示例
def apply_augmentation(image_path, label_path):
    """
    应用数据增强到YOLO格式的数据
    
    Args:
        image_path (str): 图像文件路径
        label_path (str): YOLO格式标注文件路径
    Returns:
        tuple: (增强后的图像张量, 变换后的标注框)
    """
    # 读取图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = TF.to_tensor(image)
    
    # 读取YOLO格式标注
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            boxes.append(values)
    boxes = np.array(boxes)
    
    # 应用增强
    augmentation = ScreenPhotoAugmentation()
    aug_image, aug_boxes = augmentation(image_tensor, boxes)
    
    return aug_image, aug_boxes

# 保存结果的函数
def save_augmented_data(image_tensor, boxes, output_image_path, output_label_path):
    """保存增强后的数据
    
    Args:
        image_tensor (torch.Tensor): 增强后的图像张量
        boxes (np.array): 变换后的标注框
        output_image_path (str): 输出图像路径
        output_label_path (str): 输出标注文件路径
    """
    # 保存图像
    image_pil = TF.to_pil_image(image_tensor)
    image_pil.save(output_image_path)
    
    # 保存标注
    with open(output_label_path, 'w') as f:
        for box in boxes:
            # 确保坐标在[0,1]范围内
            box[1:] = np.clip(box[1:], 0, 1)
            line = ' '.join(map(str, box))
            f.write(line + '\n')