from module_load import module_load
import time
import cv2
from tools import Tools
import os
import pyautogui

import warnings
import pandas as pd
import re
warnings.filterwarnings("ignore", category=Warning)

modules_dict = module_load()
tools = Tools()

screenshot = "./screenshot"


# radint是按字典序排列，所以这边也需要按字典序排列
csv_path = r"D:\Projects\Ark+\datasets\VinDrCXR\VinDrCXR_train.csv"
df = pd.read_csv(csv_path)

nums = len(df)
print("Total images:", nums)
sorted_img_names = sorted(df.iloc[:, 0])

# for i, name in enumerate(sorted_img_names[:1000]):
#     print(f"{i+1:2d}: {name}")


# nums = 2000 # train只要前2000张图像
for i in range(1999, nums):
    tools.get_monitor_photo(screenshot, modules_dict['interactive_device'])
    img_path = "screenshot/screenshot.png"

    new_name = fr"D:\Projects\Ark+\datasets\VinDrCXR\kinect_capture_train\{sorted_img_names[i].split('.')[0]}.png"
    os.rename(img_path, new_name)

    # 下一张图像
    print('-'*70,f"Change to the {i+1} image",'-'*70, '\n')
    pyautogui.moveTo(3440+1440+200, 720, duration=0.5) #3440+1440+200
    pyautogui.click()
    pyautogui.press("down")
