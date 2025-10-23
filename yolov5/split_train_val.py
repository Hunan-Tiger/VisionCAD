# coding:utf-8
 
import os
import random
import argparse
 
# 按照data/train.txt中的文件名，将datasets/generated_dataset/images下新建一个train文件夹，将对应的图片复制到train文件夹下
# 同时在datasets/generated_dataset/labels下新建一个train文件夹，将对应的标签复制到train文件夹下
import os
import shutil

def split_dataset():
    # 源文件夹和目标文件夹路径
    train_list_path = 'data/val.txt'
    source_images = 'datasets/generated_dataset/images'
    source_labels = 'datasets/generated_dataset/labels' 
    target_images = 'datasets/generated_dataset/images/val'
    target_labels = 'datasets/generated_dataset/labels/val'

    # 创建目标文件夹
    os.makedirs(target_images, exist_ok=True)
    os.makedirs(target_labels, exist_ok=True)

    # 读取train.txt中的文件名
    with open(train_list_path, 'r') as f:
        filenames = f.read().splitlines()

    # 复制文件
    for filename in filenames:
        # 复制图片
        src_img = os.path.join(source_images, f"{filename}.png")
        dst_img = os.path.join(target_images, f"{filename}.png")
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
        
        # 复制标签
        src_label = os.path.join(source_labels, f"{filename}.txt")
        dst_label = os.path.join(target_labels, f"{filename}.txt")
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)

if __name__ == "__main__":
    split_dataset()