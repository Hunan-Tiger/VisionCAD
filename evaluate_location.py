from module_load import module_load
import time
import cv2
from tools import Tools
import os
import pyautogui

import warnings
warnings.filterwarnings("ignore", category=Warning)

modules_dict = module_load()
tools = Tools()

screenshot = "./screenshot"



prev_frame = None
while True:
    # root = tk.Tk()
    # root.title("Screenshot Button")
    # root.geometry("100x50+10+10")
    # root.attributes("-topmost", True)
    # def take_screenshot():
    #     root.destroy()
    # def on_closing():
    #     root.destroy()
    #     exit()
    # root.protocol("WM_DELETE_WINDOW", on_closing)
    # button = tk.Button(root, text="截取", command=take_screenshot)
    # button.pack()
    # root.mainloop()
    # ==================================================================


    tools.get_monitor_photo(screenshot, modules_dict['interactive_device'])
    time_1 = time.time()
    # ==================================================================
    monitor_photo = tools.localize_monitor(modules_dict['monitor_detect_module'], cv2.imread("screenshot/screenshot.png"), 62) # return ndarray, 62 represents TV index.
    
    # if prev_frame is None:
    #     prev_frame = monitor_photo
    # else:
    #     isSame, percent = tools.diff_two_frames(prev_frame, monitor_photo)
    #     if isSame:
    #         print(f"** SKIP **")
    #         print(f"percent: {percent:.2f}%")
    #         continue
    #     else:
    #         print(f"** NOT SMAE **")
    #         print(f"percent: {percent:.2f}%")
    #         prev_frame = monitor_photo

    rough_medical_img, flag = tools.localize_image(monitor_photo)
    if not flag:
        print(f"no medical image detected")
        continue

    output_dir = "evaluate_location_images/"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"medical_image_{int(time.time())}.png")
    cv2.imwrite(output_path, rough_medical_img)
    print(f"Saved medical image to {output_path}")


    # 下一张图像
    print('-'*70," Change to the next image ",'-'*70, '\n')
    pyautogui.moveTo(3440+1440+200, 720, duration=0.5) #3440+1440+200
    pyautogui.click()
    pyautogui.press("right")

    