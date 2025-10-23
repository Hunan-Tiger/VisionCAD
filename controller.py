import os
from PIL import Image, ImageEnhance
import pyautogui
import time
import cv2
import pykinect_azure as pykinect
from pykinect_azure.k4a import _k4a
import tkinter as tk
from tkinter import messagebox
from PIL import ImageTk
import edge_tts
import threading
import subprocess
import numpy as np


def get_screenshot(output_path):
    # 定义临时保存路径
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    temp_path = os.path.join(output_path, "screenshot_temp.png")
    save_path = os.path.join(output_path, "screenshot.jpg")
    
    # 创建截图并保存为PNG格式
    screenshot = pyautogui.screenshot()
    screenshot.save(temp_path)
    
    # 转换为JPEG格式
    image = Image.open(temp_path)
    image.convert("RGB").save(save_path, "JPEG")
    
    # 删除临时的PNG文件
    os.remove(temp_path)
    
    print(f"Screenshot saved at {save_path}")
    
    return save_path

def take_screenshot_and_save(x, y, x_range, y_range, filename):
    path = 'slice'
    if not os.path.exists(path):
        os.makedirs(path)
    screenshot_region = (int(x-x_range/2), int(y-y_range/2), x_range, y_range)  
    screenshot = pyautogui.screenshot(region=screenshot_region)
    time.sleep(0.01)  # 等待截图完成
    screenshot.save(os.path.join(path, filename))

def kinect_screenshot(output_path, device):
    save_path = os.path.join(output_path, "screenshot.png")
    if os.path.exists(save_path):
        os.remove(save_path)
    # 清空设备缓存
    for _ in range(7):
        capture = device.update()
        ret, _ = capture.get_color_image()
        if not ret:
            continue
    while True:
        capture = device.update()
        ret, color_image = capture.get_color_image()
        if not ret:
            continue
        else:
            cv2.imwrite(save_path, color_image)
            break

    # while True:
    #     ret, frame = device.read()
    #     if not ret:
    #         continue
    #     else:
    #         cv2.imwrite(save_path, frame)
    #         break
    # return 




def enhanced_image(img, factor):
    enhancer = ImageEnhance.Sharpness(img)
    img_enhanced = enhancer.enhance(factor)
    return img_enhanced


def confirm_result(medical_image, output_Type, output_ID, diagnosis_result):
    # original_width, original_height = int(medical_image.shape[1]), int(medical_image.shape[0])
    # resized_image = medical_image.resize((original_width // 2, original_height // 2), Image.LANCZOS)
    
    # img_window = tk.Tk()
    # img_window.title(f"{output_Type}_{output_ID}")
    # img_window.attributes('-topmost', True)  # Make window stay on top
    # img_label = tk.Label(img_window)
    # img_tk = ImageTk.PhotoImage(resized_image)
    # img_label.configure(image=img_tk)
    # img_label.image = img_tk
    # img_label.pack()

    stop_event = threading.Event()
    speecher_thread = threading.Thread(target=Speecher, args=(diagnosis_result, stop_event))
    speecher_thread.start()

    msg_box = tk.Tk()
    msg_box.withdraw()
    msg_box.attributes('-topmost', True)  # Make message box stay on top
    messagebox.showinfo("Diagnosis Result", diagnosis_result)
    msg_box.destroy()
    stop_event.set()
    # img_window.destroy()
    speecher_thread.join()


def Speecher(TEXT, stop_event):
    cmd = f'edge-tts --text \"{TEXT}\" --write-media result.wav'
    try:
        # 使用 subprocess.run 等待命令执行完毕，并添加超时机制
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, timeout=60)
        if result.returncode == 0:
            play_cmd = 'start /min mpv.exe result.wav'
            play_process = subprocess.Popen(play_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            
            while not stop_event.is_set():
                continue
            
            try:
                subprocess.run('taskkill /f /im mpv.exe', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, timeout=10)
            except subprocess.TimeoutExpired:
                print("Taskkill command timed out")
            except Exception as e:
                print(f"An error occurred while killing mpv.exe: {e}")
        else:
            print(f"Error executing command: {result.stderr.decode()}")
    except subprocess.TimeoutExpired:
        print("Command timed out")
    except Exception as e:
        print(f"An error occurred: {e}")

    if os.path.exists("result.wav"):
        os.remove("result.wav")


def correct_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gamma = 1.2
    corrected = np.power(gray/255.0, gamma) * 255.0
    # Increase sharpness using unsharp masking
    blurred = cv2.GaussianBlur(corrected, (0, 0), 3)
    corrected = cv2.addWeighted(corrected, 2, blurred, -1, 0)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    corrected = cv2.cvtColor(corrected, cv2.COLOR_GRAY2RGB)
    return corrected

def correct_color(img):
    b, g, r = cv2.split(img)
    b = b * 0.88
    corrected = cv2.merge([b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)])
    # Reduce brightness
    corrected = cv2.convertScaleAbs(corrected, alpha=0.9, beta=-5)
    # Increase sharpness using unsharp masking
    blurred = cv2.GaussianBlur(corrected, (0, 0), 3)
    corrected = cv2.addWeighted(corrected, 2.0, blurred, -1.0, 0)
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    #cv2.imshow("corrected1", corrected)
    # Increase contrast using histogram equalization
    lab = cv2.cvtColor(corrected, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16,16))  # 降低对比度：降低clipLimit并增大tileGridSize
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return corrected


# def Speecher(TEXT, stop_event):
#     cmd = f'edge-tts --text \"{TEXT}\" --write-media result.wav'
#     result = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
#     while result.poll() is None:
#         continue
#     play_cmd = 'start /min mpv.exe result.wav'
#     subprocess.Popen(play_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

#     while not stop_event.is_set():
#         continue
#     subprocess.run('taskkill /f /im mpv.exe', stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
#     if os.path.exists("result.wav"):
#         os.remove("result.wav")

