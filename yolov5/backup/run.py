import os
from PIL import Image
import cv2
import torch
import torchvision.transforms.functional as TF
from distortion import ScreenPhotoAugmentation
import numpy as np
import itertools
from tqdm import tqdm
import random
import gc


def convert_images_to_png(folder_path):
    iter = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.png')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            png_filename = f'{iter:03d}' + '.png'
            png_path = os.path.join(folder_path, png_filename)
            img.save(png_path, 'PNG')
            os.remove(img_path)  # 删除原来的图片
            print(f"Converted {filename} to {png_filename} and deleted the original file")
            iter += 1

folder_path = 'radiograph/'
# convert_images_to_png(folder_path)

def save_image_with_bboxes():
    image_path = r'D:\Projects\yolo\generated_images\000_0000.png'
    image = cv2.imread(image_path)
    bboxes = []
    with open(r'D:\Projects\yolo\generated_images\generated_labels\000_0000.txt', 'r') as file:
        for line in file:
            bboxes.append(tuple(map(float, line.strip().split())))

    height, width = image.shape[:2]
    for bbox in bboxes:
        cls, center_x, center_y, w, h = bbox
        top_left = (int((center_x - w / 2) * width), int((center_y - h / 2) * height))
        bottom_right = (int((center_x + w / 2) * width), int((center_y + h / 2) * height))
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)
    
    # 保存带框图像
    save_path = os.path.join(os.path.dirname(image_path), f"annotated_{os.path.basename(image_path)}")
    cv2.imwrite(save_path, image)
    print(f"Annotated image saved to: {save_path}")



# 我想将PACS_template文件夹下的所有图片作为模版，每张图片有若干个矩形“坑”。其中“坑”的位置在labels文件夹下的txt文件中，每一行是cls, center_x, center_y, w, h，有几行就是有几个“坑”
# 然后将radiograph文件夹下的所有图片填充到这些坑中, 假如这张模版有一个“坑”，那么轮流将radiograph文件夹下的图片填充到这个坑中，如果这张模版有两个“坑”，就按排列组合的方式轮流取两张图片填充到这两个坑中。
# 注意，坑的大小和图片的大小不一定一样，所以需要缩放图片（拉伸，按中心裁剪，等等）
# 生成的图片保存在generated_images文件夹下, labels就是模版的labels然后保存在generated_labels文件夹下
# 命名以模板文件名加下划线加序号（4位），比如模版文件名是002.png，那么生成的图片就是002_0000.png，002_0001.png等等
def resize_and_crop(image, target_width, target_height):
    """调整图像大小并裁剪，处理图像小于目标尺寸的情况
    Args:
        image: 输入图像 (numpy array)
        target_width: 目标宽度
        target_height: 目标高度
    Returns:
        调整后的图像
    """
    height, width = image.shape[:2]
    
    # 计算需要的缩放比例
    scale_w = target_width / width
    scale_h = target_height / height
    scale = max(scale_w, scale_h)  # 确保图像至少达到目标尺寸
    
    # 调整图像大小
    if scale > 1:  # 如果需要放大
        # 使用 INTER_LINEAR 进行放大，效果更好
        resized_image = cv2.resize(image, (int(width * scale), int(height * scale)), 
                                 interpolation=cv2.INTER_LINEAR)
    else:
        # 缩小时使用 INTER_AREA 效果更好
        resized_image = cv2.resize(image, (int(width * scale), int(height * scale)), 
                                 interpolation=cv2.INTER_AREA)
    
    # 如果调整后的图像仍小于目标尺寸，添加边界填充
    resized_h, resized_w = resized_image.shape[:2]
    if resized_w < target_width or resized_h < target_height:
        pad_w = max(0, target_width - resized_w)
        pad_h = max(0, target_height - resized_h)
        
        # 计算填充量
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        # 使用边界复制模式进行填充
        resized_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right,
                                         cv2.BORDER_REPLICATE)
    
    # 最后进行裁剪以确保准确的目标尺寸
    resized_h, resized_w = resized_image.shape[:2]
    crop_x = (resized_w - target_width) // 2
    crop_y = (resized_h - target_height) // 2
    cropped_image = resized_image[crop_y:crop_y + target_height, 
                                 crop_x:crop_x + target_width]
    
    return cropped_image

def flip_box(box, flip_code):
    """翻转bbox坐标
    Args:
        box: [cls, cx, cy, w, h] - 归一化坐标
        flip_code: 0(上下翻转)或1(左右翻转)
    """
    cls, cx, cy, w, h = box
    if flip_code == 1:  # 左右翻转
        cx = 1 - cx
    elif flip_code == 0:  # 上下翻转
        cy = 1 - cy
    return [cls, cx, cy, w, h]
def generated_images():
    template_folder = r'D:\Projects\yolo\backup\PACS_template'
    radiograph_folder = r'D:\Projects\yolo\backup\radiograph'
    generated_images_folder = r'D:\Projects\yolo\datasets\generated_dataset\images'
    generated_labels_folder = r'D:\Projects\yolo\datasets\generated_dataset\labels'

    os.makedirs(generated_images_folder, exist_ok=True)
    os.makedirs(generated_labels_folder, exist_ok=True)

    # 初始化屏幕拍照效果增强器
    augmentation = ScreenPhotoAugmentation(p=0.8)

    template_images = [f for f in os.listdir(template_folder) if f.endswith('.png')]
    radiograph_images = [f for f in os.listdir(radiograph_folder) if f.endswith('.png')]

    for template_image_file in tqdm(template_images):

        template_image_path = os.path.join(template_folder, template_image_file)
        label_path = os.path.join(template_folder, 'labels', template_image_file.replace('.png', '.txt'))

        bboxes = []
        with open(label_path, 'r') as file:
            for line in file:
                bboxes.append(tuple(map(float, line.strip().split())))

        # bbox_combinations = list(itertools.product(radiograph_images, repeat=len(bboxes)))

        bbox_combinations = list(itertools.permutations(radiograph_images, len(bboxes)))
        if len(bboxes) == 1:
            # 对于单个bbox的情况
            i = 0
            for _, radiograph_image_file in enumerate(radiograph_images):
                template_image = cv2.imread(template_image_path)
                height, width = template_image.shape[:2]
                
                # 原始图像
                bbox = bboxes[0]
                cls, center_x, center_y, w, h = bbox
                top_left = (int((center_x - w / 2) * width), int((center_y - h / 2) * height))
                bottom_right = (int((center_x + w / 2) * width), int((center_y + h / 2) * height))
                
                radiograph_image_path = os.path.join(radiograph_folder, radiograph_image_file)
                radiograph_image = cv2.imread(radiograph_image_path)
                radiograph_resized = resize_and_crop(radiograph_image, bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])
                template_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = radiograph_resized
                
                # 保存原始图像
                output_image_path = os.path.join(generated_images_folder, f"{template_image_file.replace('.png', '')}_{i:05d}.png")
                output_label_path = os.path.join(generated_labels_folder, f"{template_image_file.replace('.png', '')}_{i:05d}.txt")
                cv2.imwrite(output_image_path, template_image)
                with open(output_label_path, 'w') as label_file:
                    label_file.write(f"{int(bbox[0])} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")
                
                # 左右翻转
                flipped_h = cv2.flip(template_image, 1)
                flipped_bbox_h = flip_box(bbox, 1)
                output_image_path = os.path.join(generated_images_folder, f"{template_image_file.replace('.png', '')}_{i+1:05d}.png")
                output_label_path = os.path.join(generated_labels_folder, f"{template_image_file.replace('.png', '')}_{i+1:05d}.txt")
                cv2.imwrite(output_image_path, flipped_h)
                with open(output_label_path, 'w') as label_file:
                    label_file.write(f"{int(flipped_bbox_h[0])} {flipped_bbox_h[1]} {flipped_bbox_h[2]} {flipped_bbox_h[3]} {flipped_bbox_h[4]}\n")
                
                # 上下翻转
                flipped_v = cv2.flip(template_image, 0)
                flipped_bbox_v = flip_box(bbox, 0)
                output_image_path = os.path.join(generated_images_folder, f"{template_image_file.replace('.png', '')}_{i+2:05d}.png")
                output_label_path = os.path.join(generated_labels_folder, f"{template_image_file.replace('.png', '')}_{i+2:05d}.txt")
                cv2.imwrite(output_image_path, flipped_v)
                with open(output_label_path, 'w') as label_file:
                    label_file.write(f"{int(flipped_bbox_v[0])} {flipped_bbox_v[1]} {flipped_bbox_v[2]} {flipped_bbox_v[3]} {flipped_bbox_v[4]}\n")
                i += 3
        elif len(bboxes) == 3:
            sample_size = max(1, len(bbox_combinations) // 600) # 只要200张
        elif len(bboxes) == 4:
            sample_size = max(1, len(bbox_combinations) // 32500) # 只要200张
        elif len(bboxes) == 5:
            sample_size = max(1, len(bbox_combinations) // 1560000) # 只要200张
        else:
            sample_size = len(bbox_combinations)
        try:
            sampled_combinations = random.sample(bbox_combinations, sample_size)
        except:
            sampled_combinations = bbox_combinations

        for i, combination in enumerate(sampled_combinations):
            template_image = cv2.imread(template_image_path)
            height, width = template_image.shape[:2]

            for bbox, radiograph_image_file in zip(bboxes, combination):
                cls, center_x, center_y, w, h = bbox
                top_left = (int((center_x - w / 2) * width), int((center_y - h / 2) * height))
                bottom_right = (int((center_x + w / 2) * width), int((center_y + h / 2) * height))

                radiograph_image_path = os.path.join(radiograph_folder, radiograph_image_file)
                radiograph_image = cv2.imread(radiograph_image_path)
                radiograph_resized = resize_and_crop(radiograph_image, bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])

                template_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = radiograph_resized

            # 转换为PyTorch张量并应用扭曲效果
            template_tensor = TF.to_tensor(cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB))
            bboxes_np = np.array(bboxes)
            
            # 应用扭曲效果和标注框转换
            augmented_tensor, augmented_bboxes = augmentation(template_tensor, bboxes_np)
            
            # 转换回OpenCV格式
            augmented_image = cv2.cvtColor(
                (augmented_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8),
                cv2.COLOR_RGB2BGR
            )

            output_image_path = os.path.join(generated_images_folder, f"{template_image_file.replace('.png', '')}_{i:05d}.png")
            output_label_path = os.path.join(generated_labels_folder, f"{template_image_file.replace('.png', '')}_{i:05d}.txt")

            cv2.imwrite(output_image_path, augmented_image)

            with open(output_label_path, 'w') as label_file:
                for bbox in augmented_bboxes:
                    label_file.write(f"{int(bbox[0])} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")

if __name__ == "__main__":
    generated_images()
    #save_image_with_bboxes()