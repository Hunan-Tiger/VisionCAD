from module_load import module_load, diagnosis_module_load
import time
import cv2
from tools import Tools
import tkinter as tk
import json
import os

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
    
    # real-time detection
    if prev_frame is None:
        prev_frame = monitor_photo
    else:
        isSame, percent = tools.diff_two_frames(prev_frame, monitor_photo)
        if isSame:
            print(f"** SKIP **")
            print(f"percent: {percent:.2f}%")
            continue
        else:
            print(f"** NOT SMAE **")
            print(f"percent: {percent:.2f}%")
            prev_frame = monitor_photo

    rough_medical_img, flag = tools.localize_image(monitor_photo)
    if not flag:
        print(f"no medical image detected")
        continue


    # 修复图像
    medical_img = tools.restorer(rough_medical_img, modules_dict['image_restoer_module'], modules_dict['clip_module'][0], modules_dict['clip_module'][1])

    time_2 = time.time()
    print(f'localization and Restore spent time: {int((time_2 - time_1)*1000)} ms')

    # ==================================================================
    PartType = tools.identification(medical_img, modules_dict['discrimination_module'][0], modules_dict['discrimination_module'][1], modules_dict['discrimination_module'][2])
    modules_dict = diagnosis_module_load(modules_dict, PartType) # 加载对应部位模态图像的诊断模型

    classifier_results = tools.diagnosis(medical_img, modules_dict['diagnosis_module'][PartType], PartType)
    cv2.imwrite(f'./medical_img/{time.strftime("%Y%m%d%H%M%S")}_{PartType}.png', medical_img)
    print(classifier_results)


    time_3 = time.time()
    print(f'Diagnosis spent time: {int((time_3 - time_2)*1000)} ms')

    # ==================================================================
    generated_report = tools.report_generation(medical_img, classifier_results)
    print('Report:', generated_report)
    time_4 = time.time()
    print(f'Report generation spent time: {int((time_4 - time_3)*1000)} ms')
    tools.multimedia_dispaly(generated_report)


    # ==================================================================
    generated_reports_path = f"generated_reports/{PartType}.json"
    dict_ = {f'{time.strftime("%Y-%m-%d %H:%M:%S")}': generated_report}
    with open(generated_reports_path, 'a', encoding='utf-8') as f:
        json.dump(dict_, f, ensure_ascii=False, indent=4)
        f.write(',')
        f.write('\n')

    # last_screenshot = "./screenshot/last_screenshot.png"
    # if os.path.exists(last_screenshot):
    #     os.remove(last_screenshot)
    # os.rename("./screenshot/screenshot.png", last_screenshot)